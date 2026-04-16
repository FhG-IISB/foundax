from __future__ import annotations

import jax
import jax.numpy as jnp


def resize_pos_embed(
    old_embed: jnp.ndarray,
    new_h: int,
    new_w: int,
) -> jnp.ndarray:
    """Resize a spatial position embedding via bilinear interpolation.

    Works for the DPOT-style pos_embed stored in channels-last format:
      (1, H, W, C)  for 2D models
      (1, H, W, D, C) for 3D models
    """
    if old_embed.ndim == 4:
        # (1, H, W, C) -> resize spatial dims
        return jax.image.resize(
            old_embed, (1, new_h, new_w, old_embed.shape[-1]), method="bilinear"
        )
    elif old_embed.ndim == 5:
        # (1, H, W, D, C)
        new_d = new_h  # assume isotropic
        return jax.image.resize(
            old_embed,
            (1, new_h, new_w, new_d, old_embed.shape[-1]),
            method="bilinear",
        )
    else:
        raise ValueError(f"Expected pos_embed with 4 or 5 dims, got {old_embed.ndim}")
