from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class LReLuRegular(nn.Module):
    in_size: int
    out_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.leaky_relu(x)
        if self.in_size != self.out_size:
            b, h, w, c = x.shape
            x = jax.image.resize(
                x, (b, self.out_size, self.out_size, c), method="bilinear"
            )
        return x


class LReLuTorch(nn.Module):
    channels: int
    in_size: int
    out_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # NHWC throughout
        b, h, w, c = x.shape

        # Up-interpolate 2x
        x = jax.image.resize(
            x,
            (b, 2 * self.in_size, 2 * self.in_size, c),
            method="bilinear",
            antialias=True,
        )
        x = jax.nn.leaky_relu(x)

        # Down-interpolate back to in_size
        x = jax.image.resize(
            x,
            (b, self.in_size, self.in_size, c),
            method="bilinear",
            antialias=True,
        )

        # Optionally resize to out_size
        if self.in_size != self.out_size:
            x = jax.image.resize(
                x,
                (b, self.out_size, self.out_size, c),
                method="bilinear",
                antialias=True,
            )

        # Add bias
        bias = self.param("bias", nn.initializers.zeros_init(), (self.channels,))
        x = x + bias
        return x
