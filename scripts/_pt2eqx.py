"""
Shared PyTorch → Equinox weight-transfer + comparison helpers.

Used by the per-architecture ``scripts/compare_<name>.py`` scripts to
copy parameters tensor-by-tensor from a seeded PyTorch reference model
into a foundax Equinox model and compare forward outputs.

Convention notes worth knowing before writing a new copier:

* foundax's custom ``Linear`` (``foundax/architectures/linear.py``) stores
  weights as ``(out_features, in_features)`` — same layout as PyTorch's
  ``nn.Linear``. No transpose is needed.
* foundax's NHWC ``Conv2d`` (``foundax/architectures/common.py``) stores
  weights as ``(kH, kW, in, out)``. PT ``nn.Conv2d`` is
  ``(out, in, kH, kW)`` — needs ``transpose(2, 3, 1, 0)``.
* foundax's ``Conv3dNHWC`` (``foundax/architectures/unet.py``) stores
  weights as ``(kD, kH, kW, in, out)``. PT ``nn.Conv3d`` is
  ``(out, in, kD, kH, kW)`` — needs ``transpose(2, 3, 4, 1, 0)``.
* ``eqx.nn.LayerNorm`` has ``weight`` + ``bias`` arrays (gamma + beta).
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


# ── core surgical setter ───────────────────────────────────────────────────


def set_eqx_array(module, path: List[Tuple[str, Any]], value):
    """Replace a single jnp array inside an Equinox module at the given path.

    ``path`` is a list of ``(attr_name, optional_index)`` tuples that
    walks from the module to the array. Use ``index=None`` for plain
    attribute access; pass an int to index into a list.
    """
    import equinox as eqx
    import jax.numpy as jnp

    def get(m):
        for attr, idx in path:
            m = getattr(m, attr)
            if idx is not None:
                m = m[idx]
        return m

    return eqx.tree_at(get, module, jnp.asarray(value))


# ── tensor-by-tensor copiers ───────────────────────────────────────────────


def copy_linear(eqx_module, prefix, pt_linear, *, has_bias=True):
    """Copy a ``torch.nn.Linear`` into a foundax ``Linear``.

    PT layout ``(out, in)`` matches foundax — no transpose.
    """
    eqx_module = set_eqx_array(
        eqx_module,
        prefix + [("weight", None)],
        pt_linear.weight.detach().cpu().numpy(),
    )
    if has_bias and pt_linear.bias is not None:
        eqx_module = set_eqx_array(
            eqx_module,
            prefix + [("bias", None)],
            pt_linear.bias.detach().cpu().numpy(),
        )
    return eqx_module


def copy_conv2d(eqx_module, prefix, pt_conv):
    """PT ``Conv2d`` weight ``(out, in, kH, kW)`` → foundax NHWC
    ``(kH, kW, in, out)``."""
    w = pt_conv.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)
    eqx_module = set_eqx_array(eqx_module, prefix + [("weight", None)], w)
    if pt_conv.bias is not None:
        eqx_module = set_eqx_array(
            eqx_module,
            prefix + [("bias", None)],
            pt_conv.bias.detach().cpu().numpy(),
        )
    return eqx_module


def copy_conv3d(eqx_module, prefix, pt_conv):
    """PT ``Conv3d`` weight ``(out, in, kD, kH, kW)`` → foundax NDHWC
    ``(kD, kH, kW, in, out)``."""
    w = pt_conv.weight.detach().cpu().numpy().transpose(2, 3, 4, 1, 0)
    eqx_module = set_eqx_array(eqx_module, prefix + [("weight", None)], w)
    if pt_conv.bias is not None:
        eqx_module = set_eqx_array(
            eqx_module,
            prefix + [("bias", None)],
            pt_conv.bias.detach().cpu().numpy(),
        )
    return eqx_module


def copy_layernorm(eqx_module, prefix, pt_ln):
    """Copy a ``torch.nn.LayerNorm`` into an ``eqx.nn.LayerNorm``."""
    eqx_module = set_eqx_array(
        eqx_module,
        prefix + [("weight", None)],
        pt_ln.weight.detach().cpu().numpy(),
    )
    eqx_module = set_eqx_array(
        eqx_module,
        prefix + [("bias", None)],
        pt_ln.bias.detach().cpu().numpy(),
    )
    return eqx_module


# ── output comparison ──────────────────────────────────────────────────────


def compare_arrays(name, pt_out, jax_out, atol=1e-4, rtol=1e-4):
    """Print PASS/FAIL with max abs, mean abs, and relative L2 diffs.

    Returns ``True`` on PASS.
    """
    pt_np = (
        pt_out.detach().cpu().numpy()
        if hasattr(pt_out, "detach")
        else np.asarray(pt_out)
    )
    jax_np = np.asarray(jax_out)
    assert pt_np.shape == jax_np.shape, f"{name}: shape {pt_np.shape} vs {jax_np.shape}"
    max_diff = float(np.max(np.abs(pt_np - jax_np)))
    mean_diff = float(np.mean(np.abs(pt_np - jax_np)))
    denom = float(np.linalg.norm(pt_np)) + 1e-12
    rel = float(np.linalg.norm(pt_np - jax_np)) / denom
    threshold = atol + rtol * float(np.max(np.abs(pt_np)))
    status = "PASS" if max_diff < threshold else "FAIL"
    print(
        f"  [{status}] {name:<32} shape={tuple(pt_np.shape)} "
        f"max_abs={max_diff:.3e}  mean_abs={mean_diff:.3e}  rel_l2={rel:.3e}"
    )
    return status == "PASS"


__all__ = [
    "set_eqx_array",
    "copy_linear",
    "copy_conv2d",
    "copy_conv3d",
    "copy_layernorm",
    "compare_arrays",
]
