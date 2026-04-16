"""Flax linen MLP for neural operator applications.

Port of jNO's Equinox MLP to Flax ``nn.Module``.  Uses ``nn.Dense``
instead of a custom Linear layer.
"""

from typing import Callable, Optional, Sequence

import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture.

    Attributes
    ----------
    output_dim : int
        Number of output features.
    hidden_dims : int | Sequence[int]
        Width(s) of hidden layers.  An ``int`` produces ``num_layers``
        identical hidden layers.
    num_layers : int
        Number of hidden layers (used when ``hidden_dims`` is an ``int``).
    activation : Callable
        Activation function for hidden layers (e.g. ``jax.nn.gelu``,
        ``jnp.tanh``, ``jax.nn.silu``).
    output_activation : Callable | None
        Optional activation applied to the output.
    use_bias : bool
        Whether hidden Dense layers use bias.
    final_layer_bias : bool
        Whether the final Dense layer uses bias.
    dropout_rate : float
        Dropout probability (applied after each hidden layer).
    batch_norm : bool
        Use batch normalization.
    layer_norm : bool
        Use layer normalization.
    """

    output_dim: int = 1
    hidden_dims: int | Sequence[int] = 64
    num_layers: int = 2
    activation: Callable = jnp.tanh
    output_activation: Optional[Callable] = None
    use_bias: bool = True
    final_layer_bias: bool = True
    dropout_rate: float = 0.0
    batch_norm: bool = False
    layer_norm: bool = False

    @nn.compact
    def __call__(self, *inputs: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        act_fn = self.activation
        out_act_fn = self.output_activation

        if isinstance(self.hidden_dims, int):
            widths = [self.hidden_dims] * self.num_layers
        else:
            widths = list(self.hidden_dims)

        # Concatenate all inputs along feature axis
        if len(inputs) == 1:
            h = inputs[0]
        else:
            h = jnp.concatenate(inputs, axis=-1)

        # Hidden layers
        for i, w in enumerate(widths):
            h = nn.Dense(w, use_bias=self.use_bias, name=f"hidden_{i}")(h)
            if self.batch_norm:
                h = nn.BatchNorm(use_running_average=deterministic, name=f"bn_{i}")(h)
            if self.layer_norm:
                h = nn.LayerNorm(name=f"ln_{i}")(h)
            h = act_fn(h)
            if self.dropout_rate > 0.0:
                h = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(h)

        # Output layer
        h = nn.Dense(self.output_dim, use_bias=self.final_layer_bias, name="output")(h)
        if out_act_fn is not None:
            h = out_act_fn(h)
        return h
