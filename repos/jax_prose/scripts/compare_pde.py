#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as tnn
from flax.serialization import from_bytes

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


class PTPDE(tnn.Module):
    def __init__(self, tr, params, n_words: int):
        super().__init__()
        id2word = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
        for i in range(3, n_words):
            id2word[i] = f"tok_{i}"

        self.embedder = tnn.Sequential(
            tnn.Linear(1 + params.max_output_dimension * params.x_patch_size, params.data_enc_emb_dim),
            tnn.GELU(),
            tnn.Linear(params.data_enc_emb_dim, params.data_enc_emb_dim),
        )
        self.text_encoder = tr.TextTransformerModel(
            params,
            id2word,
            is_encoder=True,
            with_output=False,
            use_prior_embeddings=False,
            positional_embeddings=params.text_enc_positional_embeddings,
        )
        self.data_encoder = tr.DataTransformerModel(
            params,
            is_encoder=True,
            with_output=False,
            positional_embeddings=params.data_enc_positional_embeddings,
        )
        self.fusion = tr.FusionTransformerModel(
            params,
            positional_embeddings=params.fusion_positional_embeddings,
        )
        self.data_decoder = tr.DataOperatorModel(
            params,
            with_output=True,
            positional_embeddings=params.data_dec_positional_embeddings,
        )
        self.normalizer = tr.RevIN(params) if getattr(params, "normalization", False) else None

    def forward(self, data_input, data_len, text_input, text_len, query_times):
        x = data_input
        mu = None
        var = None
        if self.normalizer is not None:
            x, mu, var = self.normalizer(x)

        x = self.embedder(x)
        d = self.data_encoder("fwd", x=x, lengths=data_len, causal=False)
        t = self.text_encoder("fwd", x=text_input, lengths=text_len, causal=False)
        fused = self.fusion(
            "fwd",
            x_data=d,
            x_text=t,
            lengths_data=data_len,
            lengths_text=text_len,
            causal=False,
        ).transpose(0, 1)
        q = self.data_decoder("query_emb", query_times=query_times)
        y = self.data_decoder.generate(src_enc=fused, src_len=(data_len, text_len), query_emb=q)
        if self.normalizer is not None:
            y = self.normalizer.reverse(y, mu, var)
        return y


def main():
    ap = argparse.ArgumentParser(description="Compare PROSE-PDE PyTorch and JAX outputs")
    ap.add_argument("--prose-root", type=Path, default=Path("/home/users/armbrust/projects/prose/prose_pde"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--msgpack", type=Path, default=None)
    ap.add_argument("--n-words", type=int, default=512)
    ap.add_argument("--input-len", type=int, default=10)
    ap.add_argument("--output-len", type=int, default=10)
    ap.add_argument("--text-len", type=int, default=48)
    ap.add_argument("--max-output-dimension", type=int, default=1)
    ap.add_argument("--x-patch-size", type=int, default=1)
    ap.add_argument("--x-grid-size", type=int, default=128)
    ap.add_argument("--normalization", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(args.prose_root))
    from parsers import get_parser
    from symbolicregression.model import transformer as tr

    params = get_parser().parse_args([])
    params.max_output_dimension = args.max_output_dimension
    params.x_patch_size = args.x_patch_size
    params.x_grid_size = args.x_grid_size
    params.normalization = args.normalization

    use_weights = (args.checkpoint is not None) or (args.msgpack is not None)
    if (args.checkpoint is None) != (args.msgpack is None):
        raise ValueError("Provide both --checkpoint and --msgpack, or neither.")

    if use_weights:
        state = _flatten_state(torch.load(args.checkpoint, map_location="cpu"))
        n_words = int(state["text_encoder.embeddings.weight"].shape[0])
    else:
        state = None
        n_words = args.n_words

    pt = PTPDE(tr, params, n_words=n_words)
    if use_weights:
        pt.load_state_dict(state, strict=False)
    pt.eval()

    rng = np.random.default_rng(args.seed)
    in_dim = 1 + args.max_output_dimension * args.x_patch_size
    out_dim = args.max_output_dimension * args.x_grid_size
    data_input = rng.normal(size=(args.input_len, 1, in_dim)).astype(np.float32)
    data_len = np.asarray([args.input_len], dtype=np.int64)
    text_input = rng.integers(3, n_words, size=(args.text_len, 1), dtype=np.int64)
    text_len = np.asarray([args.text_len], dtype=np.int64)
    query_times = np.linspace(0.0, 1.0, args.output_len, dtype=np.float32)

    with torch.no_grad():
        y_pt = pt(
            torch.from_numpy(data_input),
            torch.from_numpy(data_len),
            torch.from_numpy(text_input),
            torch.from_numpy(text_len),
            torch.from_numpy(query_times),
        ).cpu().numpy()

    cfg = ProseTextData2to1Config(
        x_patch_size=args.x_patch_size,
        x_grid_size=args.x_grid_size,
        split_fused_feature_data=False,
        normalization=args.normalization,
    )
    j_model = PROSEPDE2to1(
        n_words=n_words,
        pad_index=0,
        max_output_dimension=args.max_output_dimension,
        cfg=cfg,
    )
    init_vars = j_model.init(
        {"params": jax.random.PRNGKey(0)},
        jnp.asarray(data_input),
        jnp.asarray(data_len, dtype=jnp.int32),
        jnp.asarray(query_times),
        jnp.asarray(text_input, dtype=jnp.int32),
        jnp.asarray(text_len, dtype=jnp.int32),
    )
    params_j = (
        from_bytes(init_vars["params"], args.msgpack.read_bytes())
        if use_weights
        else init_vars["params"]
    )
    y_j = np.asarray(
        j_model.apply(
            {"params": params_j},
            jnp.asarray(data_input),
            jnp.asarray(data_len, dtype=jnp.int32),
            jnp.asarray(query_times),
            jnp.asarray(text_input, dtype=jnp.int32),
            jnp.asarray(text_len, dtype=jnp.int32),
        )
    )

    if y_pt.shape[-1] != out_dim:
        y_pt = y_pt.reshape(y_pt.shape[0], y_pt.shape[1], out_dim)

    if use_weights:
        diff = y_j - y_pt
        max_abs = np.max(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        rel_l2 = np.linalg.norm(diff.ravel()) / (np.linalg.norm(y_pt.ravel()) + 1e-12)

        print(f"max_abs: {max_abs:.6e}")
        print(f"rmse:    {rmse:.6e}")
        print(f"rel_l2:  {rel_l2:.6e}")
    else:
        print("Smoke mode (no weights):")
        print(f"pt_shape:  {y_pt.shape}, finite={np.isfinite(y_pt).all()}")
        print(f"jax_shape: {y_j.shape}, finite={np.isfinite(y_j).all()}")


if __name__ == "__main__":
    main()
