#!/usr/bin/env python3
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import torch
from flax.core import freeze, unfreeze
from flax.serialization import to_bytes

from jax_prose.prose_fd_2to1 import PROSE2to1


def _strip_prefix(k: str) -> str:
    for p in ("module._orig_mod.", "module."):
        if k.startswith(p):
            return k[len(p) :]
    return k


def _set_param(params, path, value):
    d = params
    for p in path[:-1]:
        d = d[p]
    d[path[-1]] = value


def main():
    ap = argparse.ArgumentParser(
        description="Convert PROSE-FD PyTorch checkpoint to JAX msgpack"
    )
    ap.add_argument("--input", type=Path, required=True, help="Path to prose_fd.pth")
    ap.add_argument("--output", type=Path, required=True, help="Output msgpack path")
    ap.add_argument("--x-num", type=int, default=128)
    ap.add_argument("--max-output-dim", type=int, default=4)
    ap.add_argument("--input-len", type=int, default=10)
    ap.add_argument("--output-len", type=int, default=10)
    args = ap.parse_args()

    ckpt = torch.load(args.input, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {_strip_prefix(k): v for k, v in state.items()}

    n_words = int(state["symbol_encoder.word_embeddings.weight"].shape[0])
    model = PROSE2to1(
        n_words=n_words, x_num=args.x_num, max_output_dim=args.max_output_dim
    )

    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.ones(
        (1, args.input_len, args.x_num, args.x_num, args.max_output_dim),
        dtype=jnp.float32,
    )
    tin = jnp.zeros((1, args.input_len, 1), dtype=jnp.float32)
    tout = jnp.zeros((1, args.output_len, 1), dtype=jnp.float32)
    sym = jnp.zeros((1, 32), dtype=jnp.int32)
    sym_mask = jnp.zeros((1, 32), dtype=bool)

    variables = model.init({"params": rng}, dummy_x, tin, tout, sym, sym_mask)
    params = unfreeze(variables["params"])
    handled = set()
    unknown = []

    for k, v in state.items():
        a = v.detach().cpu().numpy()

        if k == "embedder.patch_position_embeddings":
            handled.add(k)
            _set_param(
                params, ["embedder", "patch_position_embeddings"], jnp.asarray(a)
            )
        elif k == "embedder.time_embed":
            handled.add(k)
            _set_param(params, ["embedder", "time_embed"], jnp.asarray(a))
        elif k == "embedder.conv_proj.0.weight":
            handled.add(k)
            _set_param(
                params,
                ["embedder", "conv_proj_0", "kernel"],
                jnp.asarray(a.transpose(2, 3, 1, 0)),
            )
        elif k == "embedder.conv_proj.0.bias":
            handled.add(k)
            _set_param(params, ["embedder", "conv_proj_0", "bias"], jnp.asarray(a))
        elif k == "embedder.conv_proj.2.weight":
            handled.add(k)
            _set_param(
                params,
                ["embedder", "conv_proj_1", "kernel"],
                jnp.asarray(a.transpose(2, 3, 1, 0)),
            )
        elif k == "embedder.conv_proj.2.bias":
            handled.add(k)
            _set_param(params, ["embedder", "conv_proj_1", "bias"], jnp.asarray(a))
        elif k == "embedder.post_proj.1.weight":
            handled.add(k)
            k_deconv = a.transpose(2, 3, 0, 1)
            k_deconv = k_deconv[::-1, ::-1, :, :]
            _set_param(params, ["embedder", "deconv", "kernel"], jnp.asarray(k_deconv))
        elif k == "embedder.post_proj.1.bias":
            handled.add(k)
            _set_param(params, ["embedder", "deconv", "bias"], jnp.asarray(a))
        elif k == "embedder.post_proj.3.weight":
            handled.add(k)
            _set_param(
                params,
                ["embedder", "post_conv_0", "kernel"],
                jnp.asarray(a.transpose(2, 3, 1, 0)),
            )
        elif k == "embedder.post_proj.3.bias":
            handled.add(k)
            _set_param(params, ["embedder", "post_conv_0", "bias"], jnp.asarray(a))
        elif k == "embedder.post_proj.5.weight":
            handled.add(k)
            _set_param(
                params,
                ["embedder", "post_conv_1", "kernel"],
                jnp.asarray(a.transpose(2, 3, 1, 0)),
            )
        elif k == "embedder.post_proj.5.bias":
            handled.add(k)
            _set_param(params, ["embedder", "post_conv_1", "bias"], jnp.asarray(a))

        elif k.startswith("data_encoder.transformer_encoder.layers."):
            parts = k.split(".")
            li = int(parts[3])
            rest = ".".join(parts[4:])
            base = ["data_encoder", "layers_%d" % li]
            if rest.startswith("self_attn."):
                r = rest.split(".")
                layer = r[1]
                field = r[2]
                if field == "weight":
                    _set_param(
                        params, base + ["self_attn", layer, "kernel"], jnp.asarray(a.T)
                    )
                else:
                    _set_param(
                        params, base + ["self_attn", layer, "bias"], jnp.asarray(a)
                    )
            elif rest.startswith("linear1."):
                _set_param(
                    params,
                    base + ["linear1", "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a),
                )
            elif rest.startswith("linear2."):
                _set_param(
                    params,
                    base + ["linear2", "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a),
                )
            elif rest == "norm1.weight":
                _set_param(params, base + ["norm1", "scale"], jnp.asarray(a))
            elif rest == "norm2.weight":
                _set_param(params, base + ["norm2", "scale"], jnp.asarray(a))
        elif k == "data_encoder.transformer_encoder.norm.weight":
            handled.add(k)
            _set_param(params, ["data_encoder", "norm", "scale"], jnp.asarray(a))

        elif k.startswith("symbol_encoder.transformer_encoder.layers."):
            parts = k.split(".")
            li = int(parts[3])
            rest = ".".join(parts[4:])
            base = ["symbol_encoder", "transformer_encoder", "layers_%d" % li]
            if rest.startswith("self_attn."):
                r = rest.split(".")
                layer = r[1]
                field = r[2]
                if field == "weight":
                    _set_param(
                        params, base + ["self_attn", layer, "kernel"], jnp.asarray(a.T)
                    )
                else:
                    _set_param(
                        params, base + ["self_attn", layer, "bias"], jnp.asarray(a)
                    )
            elif rest.startswith("linear1."):
                _set_param(
                    params,
                    base + ["linear1", "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a),
                )
            elif rest.startswith("linear2."):
                _set_param(
                    params,
                    base + ["linear2", "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a),
                )
            elif rest == "norm1.weight":
                _set_param(params, base + ["norm1", "scale"], jnp.asarray(a))
            elif rest == "norm2.weight":
                _set_param(params, base + ["norm2", "scale"], jnp.asarray(a))
        elif k == "symbol_encoder.transformer_encoder.norm.weight":
            handled.add(k)
            _set_param(
                params,
                ["symbol_encoder", "transformer_encoder", "norm", "scale"],
                jnp.asarray(a),
            )
        elif k == "symbol_encoder.positional_embedding.pe":
            handled.add(k)
            _set_param(params, ["symbol_encoder", "pe"], jnp.asarray(a))
        elif k == "symbol_encoder.word_embeddings.weight":
            handled.add(k)
            _set_param(
                params,
                ["symbol_encoder", "word_embeddings", "embedding"],
                jnp.asarray(a),
            )

        elif k == "fusion.type_embeddings.weight":
            handled.add(k)
            _set_param(
                params, ["fusion", "type_embeddings", "embedding"], jnp.asarray(a)
            )
        elif k.startswith("fusion.transformer_encoder.layers."):
            parts = k.split(".")
            li = int(parts[3])
            rest = ".".join(parts[4:])
            base = ["fusion", "transformer_encoder", "layers_%d" % li]
            if rest.startswith("self_attn."):
                r = rest.split(".")
                layer = r[1]
                field = r[2]
                if field == "weight":
                    _set_param(
                        params, base + ["self_attn", layer, "kernel"], jnp.asarray(a.T)
                    )
                else:
                    _set_param(
                        params, base + ["self_attn", layer, "bias"], jnp.asarray(a)
                    )
            elif rest.startswith("linear1."):
                _set_param(
                    params,
                    base + ["linear1", "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a),
                )
            elif rest.startswith("linear2."):
                _set_param(
                    params,
                    base + ["linear2", "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a),
                )
            elif rest == "norm1.weight":
                _set_param(params, base + ["norm1", "scale"], jnp.asarray(a))
            elif rest == "norm2.weight":
                _set_param(params, base + ["norm2", "scale"], jnp.asarray(a))
        elif k == "fusion.transformer_encoder.norm.weight":
            handled.add(k)
            _set_param(
                params,
                ["fusion", "transformer_encoder", "norm", "scale"],
                jnp.asarray(a),
            )

        elif k == "data_decoder.time_embed":
            handled.add(k)
            _set_param(params, ["data_decoder", "time_embed"], jnp.asarray(a))
        elif k == "data_decoder.patch_position_embeddings":
            handled.add(k)
            _set_param(
                params, ["data_decoder", "patch_position_embeddings"], jnp.asarray(a)
            )
        elif k.startswith("data_decoder.transformer_decoder.layers."):
            parts = k.split(".")
            li = int(parts[3])
            rest = ".".join(parts[4:])
            base = ["data_decoder", "layers_%d" % li]
            if rest.startswith("multihead_attn."):
                r = rest.split(".")
                layer = r[1]
                field = r[2]
                if field == "weight":
                    _set_param(
                        params,
                        base + ["multihead_attn", layer, "kernel"],
                        jnp.asarray(a.T),
                    )
                else:
                    _set_param(
                        params, base + ["multihead_attn", layer, "bias"], jnp.asarray(a)
                    )
            elif rest.startswith("linear1."):
                _set_param(
                    params,
                    base + ["linear1", "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a),
                )
            elif rest.startswith("linear2."):
                _set_param(
                    params,
                    base + ["linear2", "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a),
                )
            elif rest == "norm1.weight":
                _set_param(params, base + ["norm1", "scale"], jnp.asarray(a))
            elif rest == "norm2.weight":
                _set_param(params, base + ["norm2", "scale"], jnp.asarray(a))
        elif k == "data_decoder.transformer_decoder.norm.weight":
            handled.add(k)
            _set_param(params, ["data_decoder", "norm", "scale"], jnp.asarray(a))
        else:
            unknown.append(k)

    params_frozen = freeze(params)
    args.output.write_bytes(to_bytes(params_frozen))
    mapped = len(state) - len(unknown)
    print(f"Mapped {mapped} / {len(state)} tensors")
    if unknown:
        print("Unmapped keys:")
        for k in unknown:
            print(" -", k)
    print(f"Wrote msgpack: {args.output}")


if __name__ == "__main__":
    main()
