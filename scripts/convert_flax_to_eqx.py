#!/usr/bin/env python
"""
Convert Flax msgpack weight files to equinox-compatible serialised leaves.

Usage:
    python scripts/convert_flax_to_eqx.py --all --input-dir /path/to/msgpacks --output-dir /path/to/output
    python scripts/convert_flax_to_eqx.py --model mppTi --input /path/to/mppTi.msgpack --output mppTi_eqx.eqx
"""

import argparse
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1] / "repos"
for d in _REPO_ROOT.iterdir():
    if d.is_dir() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from flax.serialization import from_bytes
from flax.traverse_util import flatten_dict


# ──────────────────────── path normalisation ────────────────────────────

# Flax terminal → equinox terminal  (only unambiguous renames)
_TERMINAL_MAP = {'kernel': 'weight', 'scale': 'weight'}
# 'embedding' is NOT mapped globally: eqx.nn.Embedding uses 'weight' but
# relative_position_bias stores 'embedding' in both Flax and equinox.

# Model-specific: equinox attr name → Flax attr name
_POSEIDON_ALIASES = {
    'key_proj': 'key',
    'weight_dense': 'weight',
    'bias_dense': 'bias',
    'output_layer': 'output',
    'downsample_layer': 'downsample',
    'upsample_layer': 'upsample',
}
_PROSE_ALIASES = {
    'q_proj': 'linear_q',
    'k_proj': 'linear_k',
    'v_proj': 'linear_v',
    'dense1': 'linear1',
    'dense2': 'linear2',
    'final_norm': 'norm',
}
_WALRUS_ALIASES = {
    'encoders_2': 'embed_2',
    'encoders_3': 'embed_3',
    'decoders_2': 'debed_2',
    'decoders_3': 'debed_3',
}
MODEL_ALIASES = {
    "poseidon": _POSEIDON_ALIASES,
    "prose_fd": _PROSE_ALIASES,
    "walrus": _WALRUS_ALIASES,
}


def _eqx_path_to_parts(path_tuple):
    parts = []
    for p in path_tuple:
        s = str(p)
        if s.startswith('.'):
            parts.append(s[1:])
        elif s.startswith('[') and s.endswith(']'):
            parts.append(s[1:-1])
        else:
            parts.append(s)
    return parts


def _normalise_flax_key(fk):
    """Normalise a Flax flat-dict key tuple to canonical form.

    - Strip 'params' prefix
    - Dense_N → layers_N
    - layer_N → layers_N  (singular → plural)
    - block_N → blocks_N
    - Terminal rename: kernel → weight, scale → weight, embedding → weight
    """
    parts = list(fk)
    if parts and parts[0] == 'params':
        parts = parts[1:]

    result = []
    for p in parts:
        # Dense_N → layers_N
        m = re.match(r'^Dense_(\d+)$', p)
        if m:
            result.append(f"layers_{m.group(1)}")
            continue
        # layer_N → layers_N (singular to plural, for Poseidon)
        m = re.match(r'^layer_(\d+)$', p)
        if m:
            result.append(f"layers_{m.group(1)}")
            continue
        # block_N → blocks_N
        m = re.match(r'^block_(\d+)$', p)
        if m:
            result.append(f"blocks_{m.group(1)}")
            continue
        # residual_block_N → residual_blocks_N
        m = re.match(r'^residual_block_(\d+)$', p)
        if m:
            result.append(f"residual_blocks_{m.group(1)}")
            continue
        result.append(p)

    # Terminal rename
    if result:
        last = result[-1]
        if last in _TERMINAL_MAP:
            result[-1] = _TERMINAL_MAP[last]

    return tuple(result)


def _normalise_eqx_path(parts):
    """Normalise equinox path parts to canonical form.

    - Merge consecutive (name, int) → name_int
    """
    merged = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            merged.append(f"{parts[i]}_{parts[i + 1]}")
            i += 2
        else:
            merged.append(parts[i])
            i += 1
    return tuple(merged)


# ──────────────────────── matching engine ────────────────────────────

def _poseidon_remap(nk, num_stages, depth):
    """Remap Poseidon Flax normalised keys to match equinox list ordering.

    Flax names modules by ``i_layer`` / ``i`` from reversed iteration, while
    equinox stores them by list index.  The two orderings are complementary:
      decoder layers:  Flax layer_i → eqx layers[num_stages-1-i]
      blocks (Swin stages only): Flax block_i → eqx blocks[depth-1-i]
    ConvNext blocks inside residual_blocks are NOT reversed.
    """
    parts = list(nk)
    in_decoder = False
    in_swin_stage = False
    for i, p in enumerate(parts):
        if p == 'decoder':
            in_decoder = True
            in_swin_stage = False
            continue
        if p == 'encoder':
            in_decoder = False
            in_swin_stage = False
            continue
        if p.startswith('residual_blocks'):
            in_swin_stage = False
            continue
        # Detect entering a Swin stage
        m = re.match(r'^layers_(\d+)$', p)
        if m:
            if in_decoder:
                parts[i] = f"layers_{num_stages - 1 - int(m.group(1))}"
            in_swin_stage = True
            continue
        # Reverse block indices only within Swin encoder/decoder stages
        if in_swin_stage:
            m = re.match(r'^blocks_(\d+)$', p)
            if m:
                parts[i] = f"blocks_{depth - 1 - int(m.group(1))}"
    return tuple(parts)


def _prose_remap(nk):
    """Remap PROSE Flax keys: self_attn/multihead_attn → attn."""
    parts = list(nk)
    for i, p in enumerate(parts):
        if p in ('self_attn', 'multihead_attn'):
            parts[i] = 'attn'
    return tuple(parts)


def _build_flax_index(flat_flax, remap_fn=None):
    """Build normalised-key → original-key index."""
    idx = {}
    for fk in flat_flax:
        nk = _normalise_flax_key(fk)
        if remap_fn:
            nk = remap_fn(nk)
        idx[nk] = fk
    return idx


def _find_match(eqx_parts, flax_index, aliases=None):
    """Find matching Flax key for equinox parts. Returns (orig_key, tag) or (None, None)."""
    eqx_norm = _normalise_eqx_path(eqx_parts)

    # 1. Direct match
    if eqx_norm in flax_index:
        return flax_index[eqx_norm], _tag(flax_index[eqx_norm])

    # 1b. Try with 'weight' → 'embedding' fallback for eqx.nn.Embedding
    if eqx_norm[-1:] == ('weight',):
        alt = eqx_norm[:-1] + ('embedding',)
        if alt in flax_index:
            return flax_index[alt], 'embedding'

    # 2. With model-specific aliases (equinox name → Flax name)
    if aliases:
        for combo in _alias_combos(eqx_norm, aliases):
            if combo in flax_index:
                return flax_index[combo], _tag(flax_index[combo])

    # 3. Subsequence match (equinox is contraction of Flax, handles extra nesting)
    candidates = _subsequence_matches(eqx_norm, flax_index)
    if candidates:
        return candidates[0]

    # 4. Subsequence match with aliases
    if aliases:
        for combo in _alias_combos(eqx_norm, aliases):
            candidates = _subsequence_matches(combo, flax_index)
            if candidates:
                return candidates[0]

    # 5. Reverse subsequence (Flax is a contraction of equinox; equinox has
    #    extra nesting, e.g. projection/conv/weight vs projection/weight)
    candidates = _reverse_subseq_matches(eqx_norm, flax_index)
    if candidates:
        return candidates[0]
    if aliases:
        for combo in _alias_combos(eqx_norm, aliases):
            candidates = _reverse_subseq_matches(combo, flax_index)
            if candidates:
                return candidates[0]

    return None, None


def _alias_combos(parts, aliases):
    """Generate all single-substitution alias combinations."""
    for i, p in enumerate(parts):
        if p in aliases:
            yield parts[:i] + (aliases[p],) + parts[i + 1:]
    # Also try double substitutions
    subs = list(parts)
    changed = False
    for i, p in enumerate(subs):
        if p in aliases:
            subs[i] = aliases[p]
            changed = True
    if changed:
        yield tuple(subs)


def _subsequence_matches(needle, flax_index):
    """Find Flax keys where needle is a subsequence."""
    results = []
    for nk, fk in flax_index.items():
        if len(nk) > len(needle) and nk[-1] == needle[-1]:
            if _is_subseq(needle, nk):
                results.append((fk, _tag(fk)))
    return results


def _reverse_subseq_matches(eqx_norm, flax_index):
    """Find Flax keys that are a subsequence of eqx_norm (equinox has extra nesting)."""
    results = []
    for nk, fk in flax_index.items():
        if len(nk) < len(eqx_norm) and nk[0] == eqx_norm[0] and nk[-1] == eqx_norm[-1]:
            if _is_subseq(nk, eqx_norm):
                results.append((fk, _tag(fk)))
    return results


def _is_subseq(short, long):
    it = iter(long)
    return all(s in it for s in short)


def _tag(flax_key):
    last = flax_key[-1]
    if last == 'kernel':
        return 'kernel'
    elif last == 'scale':
        return 'scale'
    elif last == 'embedding':
        return 'embedding'
    return 'none'


# ──────────────────────── array transform ────────────────────────────

def _transform(flax_val, eqx_leaf, tag):
    fv = np.asarray(flax_val)
    target = eqx_leaf.shape

    if fv.shape == target:
        return jnp.array(fv, dtype=eqx_leaf.dtype)

    if tag in ('kernel', 'embedding'):
        nd = fv.ndim
        if nd == 2:
            fv = fv.T
        elif nd == 4:
            fv = fv.transpose(3, 2, 0, 1)
        elif nd == 5:
            fv = fv.transpose(4, 3, 0, 1, 2)
        elif nd == 3:
            fv = fv.transpose(2, 1, 0)

    if fv.shape != target:
        if fv.size == np.prod(target):
            fv = fv.reshape(target)
        else:
            raise ValueError(
                f"Shape mismatch: Flax {flax_val.shape} → {fv.shape}, expected {target}"
            )

    return jnp.array(fv, dtype=eqx_leaf.dtype)


# ──────────────────────── core loader ────────────────────────────

def load_flax_into_eqx(eqx_model, flax_params, aliases=None, flax_remap=None):
    flat_flax = flatten_dict(flax_params)
    flax_index = _build_flax_index(flat_flax, remap_fn=flax_remap)

    leaves_with_path = jax.tree_util.tree_leaves_with_path(eqx_model)
    mapped, missed, flax_used, replacements = 0, [], set(), {}

    for idx, (path, leaf) in enumerate(leaves_with_path):
        if not hasattr(leaf, 'shape'):
            continue
        eqx_parts = tuple(_eqx_path_to_parts(path))
        fk, tag = _find_match(eqx_parts, flax_index, aliases)
        if fk is None:
            missed.append(('/'.join(eqx_parts), leaf.shape))
            continue
        replacements[idx] = _transform(flat_flax[fk], leaf, tag)
        flax_used.add(fk)
        mapped += 1

    n_flax = sum(1 for v in flat_flax.values() if hasattr(v, 'shape'))
    n_eqx = sum(1 for _, l in leaves_with_path if hasattr(l, 'shape'))
    print(f"  Mapped {mapped}/{n_eqx} equinox leaves from {len(flax_used)}/{n_flax} Flax params")

    if missed:
        print(f"  WARNING: {len(missed)} equinox leaves NOT matched:")
        for p, s in missed[:15]:
            print(f"    {p}: {s}")
        if len(missed) > 15:
            print(f"    ... and {len(missed) - 15} more")

    unused = [k for k in set(flat_flax) - flax_used if hasattr(flat_flax[k], 'shape')]
    if unused:
        print(f"  INFO: {len(unused)} Flax params not consumed (OK if buffers/tables):")
        for k in sorted(unused)[:10]:
            print(f"    {'/'.join(k)}: {flat_flax[k].shape}")
        if len(unused) > 10:
            print(f"    ... and {len(unused) - 10} more")

    flat_leaves = list(jax.tree_util.tree_leaves(eqx_model))
    for idx, val in replacements.items():
        flat_leaves[idx] = val
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(eqx_model), flat_leaves)


# ──────────────────────── model factories ────────────────────────────

def _make_morph(variant, key):
    from jax_morph.configs import MORPH_CONFIGS
    from jax_morph.model_eqx import ViT3DRegression
    cfg = MORPH_CONFIGS[variant]
    return ViT3DRegression(
        patch_size=8, dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
        heads_xa=32, mlp_dim=cfg["mlp_dim"], max_components=3,
        conv_filter=cfg["conv_filter"], max_ar=cfg["max_ar"],
        max_patches=4096, max_fields=3, dropout=0.0, emb_dropout=0.0,
        model_size=cfg["model_size"], key=key,
    )


def _make_mpp(variant, key):
    from jax_mpp.configs import AVIT_CONFIGS
    from jax_mpp.avit_eqx import AViT
    cfg = AVIT_CONFIGS[variant]
    return AViT(
        patch_size=(4, 4), embed_dim=cfg["embed_dim"],
        processor_blocks=cfg["processor_blocks"],
        n_states=cfg["n_states"], num_heads=cfg["num_heads"], key=key,
    )


def _make_poseidon(variant, key):
    from jax_poseidon import ScOTConfig
    from jax_poseidon.scot_eqx import ScOT
    cfgs = {
        "T": dict(name="poseidonT", embed_dim=48, depths=(4, 4, 4, 4)),
        "B": dict(name="poseidonB", embed_dim=96, depths=(8, 8, 8, 8)),
        "L": dict(name="poseidonL", embed_dim=192, depths=(8, 8, 8, 8)),
    }
    c = cfgs[variant]
    config = ScOTConfig(
        name=c["name"], image_size=128, patch_size=4,
        num_channels=4, num_out_channels=4,
        embed_dim=c["embed_dim"], depths=c["depths"],
        num_heads=(3, 6, 12, 24), skip_connections=(2, 2, 2, 0),
        window_size=16, use_conditioning=True, residual_model="convnext",
        drop_path_rate=0.0,
    )
    return ScOT(config=config, use_conditioning=True, key=key)


def _make_walrus(key):
    from jax_walrus.model_eqx import IsotropicModel
    return IsotropicModel(
        hidden_dim=1408, intermediate_dim=352, n_states=67,
        processor_blocks=40, groups=16, num_heads=16, key=key,
    )


def _make_pdeformer2(variant, key):
    from jax_pdeformer2.model_eqx import PDEformer
    cfgs = {
        "small": dict(num_encoder_layers=9, embed_dim=512, ffn_embed_dim=1024, inr_dim_hidden=128),
        "base":  dict(num_encoder_layers=12, embed_dim=768, ffn_embed_dim=1536, inr_dim_hidden=768),
        "fast":  dict(num_encoder_layers=12, embed_dim=768, ffn_embed_dim=1536, inr_dim_hidden=256),
    }
    return PDEformer(**cfgs[variant], key=key)


def _make_prose_fd(key):
    from jax_prose.model_eqx import PROSE1to1
    return PROSE1to1(max_time_len=10, n_enc_layers=2, n_dec_layers=8, key=key)


MODELS = {
    "morphTi": ("morph", "Ti"), "morphS": ("morph", "S"),
    "morphM": ("morph", "M"), "morphL": ("morph", "L"),
    "mppTi": ("mpp", "Ti"), "mppS": ("mpp", "S"),
    "mppB": ("mpp", "B"), "mppL": ("mpp", "L"),
    "poseidonT": ("poseidon", "T"), "poseidonB": ("poseidon", "B"),
    "poseidonL": ("poseidon", "L"),
    "walrusL": ("walrus", None),
    "pdeformer2_small": ("pdeformer2", "small"),
    "pdeformer2_base": ("pdeformer2", "base"),
    "pdeformer2_fast": ("pdeformer2", "fast"),
    "prose_fd": ("prose_fd", None),
}


def create_model(family, variant, key):
    makers = {
        "morph": lambda: _make_morph(variant, key),
        "mpp": lambda: _make_mpp(variant, key),
        "poseidon": lambda: _make_poseidon(variant, key),
        "walrus": lambda: _make_walrus(key),
        "pdeformer2": lambda: _make_pdeformer2(variant, key),
        "prose_fd": lambda: _make_prose_fd(key),
    }
    return makers[family]()


# ──────────────────────── main ────────────────────────────

def convert_one(model_name, input_path, output_path):
    family, variant = MODELS[model_name]
    aliases = MODEL_ALIASES.get(family)
    print(f"\n{'='*60}")
    print(f"Converting {model_name} ({family}/{variant})")

    with open(input_path, 'rb') as f:
        flax_params = from_bytes(None, f.read())
    flat = flatten_dict(flax_params)
    n = sum(v.size for v in flat.values() if hasattr(v, 'shape'))
    print(f"  Flax: {sum(1 for v in flat.values() if hasattr(v, 'shape'))} leaves, {n:,} params")

    key = jax.random.PRNGKey(42)
    model = create_model(family, variant, key)
    el = jax.tree_util.tree_leaves(model)
    ne = sum(l.size for l in el if hasattr(l, 'size'))
    print(f"  Equinox: {sum(1 for l in el if hasattr(l, 'shape'))} leaves, {ne:,} params")

    flax_remap = None
    if family == "poseidon":
        poseidon_cfgs = {"T": (4, 4), "B": (4, 8), "L": (4, 8)}
        ns, dp = poseidon_cfgs[variant]
        flax_remap = lambda nk, _ns=ns, _dp=dp: _poseidon_remap(nk, _ns, _dp)
    elif family == "prose_fd":
        flax_remap = _prose_remap

    model = load_flax_into_eqx(model, flax_params, aliases=aliases, flax_remap=flax_remap)

    eqx.tree_serialise_leaves(output_path, model)
    mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--input", "-i", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()

    if args.all:
        os.makedirs(args.output_dir, exist_ok=True)
        results = {}
        for name in MODELS:
            inp = os.path.join(args.input_dir, f"{name}.msgpack")
            if not os.path.exists(inp):
                print(f"Skipping {name}: not found")
                continue
            out = os.path.join(args.output_dir, f"{name}.eqx")
            try:
                ok = convert_one(name, inp, out)
                results[name] = "OK" if ok else "FAIL"
            except Exception as e:
                import traceback; traceback.print_exc()
                results[name] = f"ERROR: {e}"
        print(f"\n{'='*60}\nSummary:")
        for n, s in sorted(results.items()):
            print(f"  {n}: {s}")
    elif args.model:
        convert_one(args.model, args.input, args.output)
    else:
        parser.error("Specify --model or --all")


if __name__ == "__main__":
    main()
