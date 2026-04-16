#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import torch
from flax.core import freeze, unfreeze
from flax.serialization import to_bytes

from jax_prose.prose_ode_pde_2to1 import PROSEPDE2to1, ProseTextData2to1Config


def _strip_prefix(k: str) -> str:
    for p in ("module._orig_mod.", "module."):
        if k.startswith(p):
            return k[len(p) :]
    return k


def _flatten_state(ckpt: dict) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return {_strip_prefix(k): v for k, v in ckpt["model"].items() if torch.is_tensor(v)}

    out: dict[str, torch.Tensor] = {}
    if isinstance(ckpt, dict) and all(isinstance(v, dict) for v in ckpt.values()):
        for mk, mv in ckpt.items():
            for k, v in mv.items():
                if torch.is_tensor(v):
                    out[f"{mk}.{_strip_prefix(k)}"] = v
        return out

    return {_strip_prefix(k): v for k, v in ckpt.items() if torch.is_tensor(v)}


def _set_param(params, path, value):
    d = params
    for p in path[:-1]:
        d = d[p]
    d[path[-1]] = value


def _map_attn(rest: str, base: list[str], params, a) -> bool:
    m = re.match(r"attentions\.(\d+)\.attn\.in_proj_weight$", rest)
    if m:
        i = int(m.group(1))
        q, k, v = a.reshape(3, a.shape[0] // 3, a.shape[1])
        _set_param(params, base + [f"attentions_{i}", "q_proj", "kernel"], jnp.asarray(q.T))
        _set_param(params, base + [f"attentions_{i}", "k_proj", "kernel"], jnp.asarray(k.T))
        _set_param(params, base + [f"attentions_{i}", "v_proj", "kernel"], jnp.asarray(v.T))
        return True

    m = re.match(r"attentions\.(\d+)\.attn\.in_proj_bias$", rest)
    if m:
        i = int(m.group(1))
        q, k, v = a.reshape(3, a.shape[0] // 3)
        _set_param(params, base + [f"attentions_{i}", "q_proj", "bias"], jnp.asarray(q))
        _set_param(params, base + [f"attentions_{i}", "k_proj", "bias"], jnp.asarray(k))
        _set_param(params, base + [f"attentions_{i}", "v_proj", "bias"], jnp.asarray(v))
        return True

    m = re.match(r"attentions\.(\d+)\.attn\.out_proj\.(weight|bias)$", rest)
    if m:
        i = int(m.group(1))
        field = m.group(2)
        _set_param(
            params,
            base + [f"attentions_{i}", "out_proj", "kernel" if field == "weight" else "bias"],
            jnp.asarray(a.T if field == "weight" else a),
        )
        return True

    return False


def _map_ffn(rest: str, base: list[str], params, a) -> bool:
    m = re.match(r"ffns\.(\d+)\.(lin1|lin2)\.(weight|bias)$", rest)
    if m:
        i = int(m.group(1))
        lin = m.group(2)
        field = m.group(3)
        _set_param(
            params,
            base + [f"ffns_{i}", lin, "kernel" if field == "weight" else "bias"],
            jnp.asarray(a.T if field == "weight" else a),
        )
        return True

    m = re.match(r"ffns\.(\d+)\.midlin\.(\d+)\.(weight|bias)$", rest)
    if m:
        i = int(m.group(1))
        j = int(m.group(2))
        field = m.group(3)
        _set_param(
            params,
            base + [f"ffns_{i}", f"midlin_{j}", "kernel" if field == "weight" else "bias"],
            jnp.asarray(a.T if field == "weight" else a),
        )
        return True

    return False


def _map_norms(rest: str, base: list[str], params, a) -> bool:
    if rest in ("layer_norm_emb.weight", "layer_norm_emb.bias"):
        _set_param(
            params,
            base + ["layer_norm_emb", "scale" if rest.endswith("weight") else "bias"],
            jnp.asarray(a),
        )
        return True

    m = re.match(r"layer_norm1\.(\d+)\.(weight|bias)$", rest)
    if m:
        i = int(m.group(1))
        field = m.group(2)
        _set_param(
            params,
            base + [f"layer_norm1_{i}", "scale" if field == "weight" else "bias"],
            jnp.asarray(a),
        )
        return True

    m = re.match(r"layer_norm2\.(\d+)\.(weight|bias)$", rest)
    if m:
        i = int(m.group(1))
        field = m.group(2)
        _set_param(
            params,
            base + [f"layer_norm2_{i}", "scale" if field == "weight" else "bias"],
            jnp.asarray(a),
        )
        return True

    return False


def main():
    ap = argparse.ArgumentParser(description="Convert PROSE-PDE checkpoint to JAX msgpack")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-words", type=int, default=-1)
    ap.add_argument("--pad-index", type=int, default=0)
    ap.add_argument("--max-output-dimension", type=int, default=1)
    ap.add_argument("--x-patch-size", type=int, default=1)
    ap.add_argument("--x-grid-size", type=int, default=128)
    ap.add_argument("--normalization", action="store_true", default=False)
    ap.add_argument("--input-len", type=int, default=10)
    ap.add_argument("--output-len", type=int, default=10)
    ap.add_argument("--text-len", type=int, default=48)
    args = ap.parse_args()

    ckpt = torch.load(args.input, map_location="cpu")
    state = _flatten_state(ckpt)

    if args.n_words > 0:
        n_words = args.n_words
    elif "text_encoder.embeddings.weight" in state:
        n_words = int(state["text_encoder.embeddings.weight"].shape[0])
    else:
        raise ValueError("Cannot infer n_words. Pass --n-words explicitly.")

    cfg = ProseTextData2to1Config(
        x_patch_size=args.x_patch_size,
        x_grid_size=args.x_grid_size,
        split_fused_feature_data=False,
        normalization=args.normalization,
    )
    model = PROSEPDE2to1(
        n_words=n_words,
        pad_index=args.pad_index,
        max_output_dimension=args.max_output_dimension,
        cfg=cfg,
    )

    in_dim = 1 + args.max_output_dimension * args.x_patch_size
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((args.input_len, 1, in_dim), dtype=jnp.float32)
    data_lengths = jnp.asarray([args.input_len], dtype=jnp.int32)
    query_times = jnp.linspace(0.0, 1.0, args.output_len, dtype=jnp.float32)
    text = jnp.zeros((args.text_len, 1), dtype=jnp.int32)
    text_lengths = jnp.asarray([args.text_len], dtype=jnp.int32)

    variables = model.init({"params": rng}, x, data_lengths, query_times, text, text_lengths)
    params = unfreeze(variables["params"])

    unknown = []
    handled = set()

    for k, v in state.items():
        a = v.detach().cpu().numpy()

        if k == "embedder.0.weight":
            _set_param(params, ["embedder_0", "kernel"], jnp.asarray(a.T))
            handled.add(k)
            continue
        if k == "embedder.0.bias":
            _set_param(params, ["embedder_0", "bias"], jnp.asarray(a))
            handled.add(k)
            continue
        if k == "embedder.2.weight":
            _set_param(params, ["embedder_2", "kernel"], jnp.asarray(a.T))
            handled.add(k)
            continue
        if k == "embedder.2.bias":
            _set_param(params, ["embedder_2", "bias"], jnp.asarray(a))
            handled.add(k)
            continue

        if k == "text_encoder.embeddings.weight":
            _set_param(params, ["text_encoder", "embeddings", "embedding"], jnp.asarray(a))
            handled.add(k)
            continue
        if k == "text_encoder.position_embeddings.weight":
            if "position_embeddings" in params["text_encoder"]:
                _set_param(params, ["text_encoder", "position_embeddings", "embedding"], jnp.asarray(a))
            handled.add(k)
            continue
        if k.startswith("text_encoder."):
            rest = k[len("text_encoder.") :]
            if _map_attn(rest, ["text_encoder"], params, a):
                handled.add(k)
                continue
            if _map_norms(rest, ["text_encoder"], params, a):
                handled.add(k)
                continue
            if _map_ffn(rest, ["text_encoder"], params, a):
                handled.add(k)
                continue

        if k == "data_encoder.position_embeddings.weight":
            if "position_embeddings" in params["data_encoder"]:
                _set_param(params, ["data_encoder", "position_embeddings", "embedding"], jnp.asarray(a))
            handled.add(k)
            continue
        if k.startswith("data_encoder."):
            rest = k[len("data_encoder.") :]
            if _map_attn(rest, ["data_encoder"], params, a):
                handled.add(k)
                continue
            if _map_norms(rest, ["data_encoder"], params, a):
                handled.add(k)
                continue
            if _map_ffn(rest, ["data_encoder"], params, a):
                handled.add(k)
                continue

        if k == "fusion.type_embeddings.weight":
            _set_param(params, ["fusion", "type_embeddings", "embedding"], jnp.asarray(a))
            handled.add(k)
            continue
        if k.startswith("fusion."):
            rest = k[len("fusion.") :]
            if _map_attn(rest, ["fusion"], params, a):
                handled.add(k)
                continue
            if _map_norms(rest, ["fusion"], params, a):
                handled.add(k)
                continue
            if _map_ffn(rest, ["fusion"], params, a):
                handled.add(k)
                continue

        if k == "data_decoder.query_embedder.weight":
            _set_param(params, ["data_decoder", "query_embedder", "kernel"], jnp.asarray(a.T))
            handled.add(k)
            continue
        if k == "data_decoder.query_embedder.bias":
            _set_param(params, ["data_decoder", "query_embedder", "bias"], jnp.asarray(a))
            handled.add(k)
            continue
        if k == "data_decoder.position_embeddings.weight":
            if "position_embeddings" in params["data_decoder"]:
                _set_param(params, ["data_decoder", "position_embeddings", "embedding"], jnp.asarray(a))
            handled.add(k)
            continue
        if k.startswith("data_decoder."):
            rest = k[len("data_decoder.") :]
            if _map_attn(rest, ["data_decoder"], params, a):
                handled.add(k)
                continue

            m = re.match(r"encoder_attn\.(\d+)\.attn\.in_proj_weight$", rest)
            if m:
                i = int(m.group(1))
                q, kk, vv = a.reshape(3, a.shape[0] // 3, a.shape[1])
                _set_param(params, ["data_decoder", f"encoder_attn_{i}", "q_proj", "kernel"], jnp.asarray(q.T))
                _set_param(params, ["data_decoder", f"encoder_attn_{i}", "k_proj", "kernel"], jnp.asarray(kk.T))
                _set_param(params, ["data_decoder", f"encoder_attn_{i}", "v_proj", "kernel"], jnp.asarray(vv.T))
                handled.add(k)
                continue
            m = re.match(r"encoder_attn\.(\d+)\.attn\.in_proj_bias$", rest)
            if m:
                i = int(m.group(1))
                q, kk, vv = a.reshape(3, a.shape[0] // 3)
                _set_param(params, ["data_decoder", f"encoder_attn_{i}", "q_proj", "bias"], jnp.asarray(q))
                _set_param(params, ["data_decoder", f"encoder_attn_{i}", "k_proj", "bias"], jnp.asarray(kk))
                _set_param(params, ["data_decoder", f"encoder_attn_{i}", "v_proj", "bias"], jnp.asarray(vv))
                handled.add(k)
                continue
            m = re.match(r"encoder_attn\.(\d+)\.attn\.out_proj\.(weight|bias)$", rest)
            if m:
                i = int(m.group(1))
                field = m.group(2)
                _set_param(
                    params,
                    ["data_decoder", f"encoder_attn_{i}", "out_proj", "kernel" if field == "weight" else "bias"],
                    jnp.asarray(a.T if field == "weight" else a),
                )
                handled.add(k)
                continue

            if _map_norms(rest, ["data_decoder"], params, a):
                handled.add(k)
                continue
            m = re.match(r"layer_norm15\.(\d+)\.(weight|bias)$", rest)
            if m:
                i = int(m.group(1))
                field = m.group(2)
                _set_param(
                    params,
                    ["data_decoder", f"layer_norm15_{i}", "scale" if field == "weight" else "bias"],
                    jnp.asarray(a),
                )
                handled.add(k)
                continue
            if _map_ffn(rest, ["data_decoder"], params, a):
                handled.add(k)
                continue

            if rest == "proj.0.weight":
                _set_param(params, ["data_decoder", "proj_0", "kernel"], jnp.asarray(a.T))
                handled.add(k)
                continue
            if rest == "proj.0.bias":
                _set_param(params, ["data_decoder", "proj_0", "bias"], jnp.asarray(a))
                handled.add(k)
                continue
            if rest == "proj.1.weight":
                _set_param(params, ["data_decoder", "proj_1", "kernel"], jnp.asarray(a.T))
                handled.add(k)
                continue
            if rest == "proj.1.bias":
                _set_param(params, ["data_decoder", "proj_1", "bias"], jnp.asarray(a))
                handled.add(k)
                continue

            if rest == "data_embedder.0.weight":
                _set_param(params, ["data_decoder", "data_embedder_0", "kernel"], jnp.asarray(a.T))
                handled.add(k)
                continue
            if rest == "data_embedder.0.bias":
                _set_param(params, ["data_decoder", "data_embedder_0", "bias"], jnp.asarray(a))
                handled.add(k)
                continue
            if rest == "data_embedder.2.weight":
                _set_param(params, ["data_decoder", "data_embedder_2", "kernel"], jnp.asarray(a.T))
                handled.add(k)
                continue
            if rest == "data_embedder.2.bias":
                _set_param(params, ["data_decoder", "data_embedder_2", "bias"], jnp.asarray(a))
                handled.add(k)
                continue
            if rest == "text_embedder.0.weight":
                _set_param(params, ["data_decoder", "text_embedder_0", "kernel"], jnp.asarray(a.T))
                handled.add(k)
                continue
            if rest == "text_embedder.0.bias":
                _set_param(params, ["data_decoder", "text_embedder_0", "bias"], jnp.asarray(a))
                handled.add(k)
                continue
            if rest == "text_embedder.2.weight":
                _set_param(params, ["data_decoder", "text_embedder_2", "kernel"], jnp.asarray(a.T))
                handled.add(k)
                continue
            if rest == "text_embedder.2.bias":
                _set_param(params, ["data_decoder", "text_embedder_2", "bias"], jnp.asarray(a))
                handled.add(k)
                continue

        if k == "normalizer.gamma":
            _set_param(params, ["normalizer", "gamma"], jnp.asarray(a))
            handled.add(k)
            continue
        if k == "normalizer.beta":
            _set_param(params, ["normalizer", "beta"], jnp.asarray(a))
            handled.add(k)
            continue

        unknown.append(k)

    args.output.write_bytes(to_bytes(freeze(params)))
    mapped = len(state) - len(unknown)
    print(f"Mapped {mapped} / {len(state)} tensors")
    if unknown:
        print("Unmapped keys:")
        for k in unknown:
            print(" -", k)
    print(f"Wrote msgpack: {args.output}")


if __name__ == "__main__":
    main()
