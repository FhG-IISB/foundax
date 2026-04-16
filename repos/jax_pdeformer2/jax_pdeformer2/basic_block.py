"""Basic building blocks for the JAX PDEformer implementation."""

from typing import Sequence, Optional
import math

import jax
import jax.numpy as jnp
from flax import linen as nn


class UniformInitDense(nn.Module):
    """Linear layer with uniform initialization similar to MindSpore's implementation."""

    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        dim_in = x.shape[-1]
        scale = math.sqrt(1.0 / dim_in)  # Kaiming uniform initialization
        kernel_init = nn.initializers.uniform(scale=scale)
        bias_init = nn.initializers.uniform(scale=scale) if self.use_bias else None

        return nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init if self.use_bias else None,
            dtype=self.dtype,
        )(x)


class MLP(nn.Module):
    """Multi-layer perceptron (MLP)."""

    dim_out: int
    dim_hidden: int
    num_layers: int = 3
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        dim_in = x.shape[-1]

        if self.num_layers > 1:
            scale = math.sqrt(1.0 / dim_in)
            x = nn.Dense(
                self.dim_hidden,
                kernel_init=nn.initializers.uniform(scale=scale),
                bias_init=nn.initializers.uniform(scale=scale),
                dtype=self.dtype,
            )(x)
            x = nn.relu(x)

            for _ in range(self.num_layers - 2):
                scale = math.sqrt(1.0 / self.dim_hidden)
                x = nn.Dense(
                    self.dim_hidden,
                    kernel_init=nn.initializers.uniform(scale=scale),
                    bias_init=nn.initializers.uniform(scale=scale),
                    dtype=self.dtype,
                )(x)
                x = nn.relu(x)

            scale = math.sqrt(1.0 / self.dim_hidden)
            x = nn.Dense(
                self.dim_out,
                kernel_init=nn.initializers.uniform(scale=scale),
                bias_init=nn.initializers.uniform(scale=scale),
                dtype=self.dtype,
            )(x)
        elif self.num_layers == 1:
            scale = math.sqrt(1.0 / dim_in)
            x = nn.Dense(
                self.dim_out,
                kernel_init=nn.initializers.uniform(scale=scale),
                bias_init=nn.initializers.uniform(scale=scale),
                dtype=self.dtype,
            )(x)
        else:
            raise ValueError(
                f"'num_layers' should be greater than 0, but got {self.num_layers}."
            )

        return x


class Sine(nn.Module):
    """Sine activation with scaling factor."""

    omega_0: float = 1.0

    @nn.compact
    def __call__(self, x):
        return jnp.sin(self.omega_0 * x)


class Scale(nn.Module):
    """Scale the input tensor."""

    a: float = 1.0

    @nn.compact
    def __call__(self, x):
        return self.a * x


class Clamp(nn.Module):
    """Clamp values within a fixed range."""

    threshold: float = 256.0

    @nn.compact
    def __call__(self, x):
        return jnp.clip(x, -self.threshold, self.threshold)
