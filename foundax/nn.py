"""Unified factory facade for all foundax models.

Every factory returns a :class:`~foundax.model.FlaxModel` — an initialized
container holding ``(apply_fn, params, post_fn, default_kwargs)``.

Usage::

    import foundax

    model = foundax.mlp(in_features=2, output_dim=1)
    model = foundax.fno2d(in_features=2, hidden_channels=32, n_modes=16)
    model = foundax.unet2d(in_channels=1, out_channels=1)
    model = foundax.poseidonT()
    model = foundax.morph_Ti()

In jNO, wrap the result for training::

    import jno
    net = jno.nn.wrap(model)
    net.optimizer(optax.adam, lr=1e-3)
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from .model import FlaxModel


# ── helpers ──────────────────────────────────────────────────

def _default_rng(key):
    if key is None:
        return jax.random.PRNGKey(0)
    return key


# =====================================================================
# Non-foundation architectures
# =====================================================================


def mlp(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    activation: Callable = jnp.tanh,
    output_activation: Callable | None = None,
    use_bias: bool = True,
    final_layer_bias: bool = True,
    dropout_rate: float = 0.0,
    batch_norm: bool = False,
    layer_norm: bool = False,
    *,
    key: jax.Array | None = None,
) -> FlaxModel:
    """Create an initialized MLP.

    Args:
        in_features: Number of input features.
        output_dim: Number of output features.
        hidden_dims: Width(s) of hidden layers.
        num_layers: Number of hidden layers (when ``hidden_dims`` is ``int``).
        activation: Activation function (e.g. ``jax.nn.gelu``, ``jnp.tanh``).
        key: PRNG key for initialization.

    Returns:
        FlaxModel: Initialized MLP.
    """
    from .architectures.mlp import MLP

    model = MLP(
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        activation=activation,
        output_activation=output_activation,
        use_bias=use_bias,
        final_layer_bias=final_layer_bias,
        dropout_rate=dropout_rate,
        batch_norm=batch_norm,
        layer_norm=layer_norm,
    )

    rng = _default_rng(key)
    dummy = jnp.ones((1, in_features))
    params = model.init(rng, dummy, deterministic=True)

    return FlaxModel(model.apply, params, deterministic=True)


def fno2d(
    in_features: int,
    hidden_channels: int = 32,
    n_modes: int = 16,
    d_vars: int = 1,
    n_layers: int = 4,
    n_steps: int = 1,
    activation: str = "gelu",
    norm: str | None = "layer",
    use_positions: bool = False,
    linear_conv: bool = True,
    d_model: Tuple[int, int] = (64, 64),
    *,
    key: jax.Array | None = None,
) -> FlaxModel:
    """Create an initialized 2-D Fourier Neural Operator.

    Returns:
        FlaxModel: Initialized FNO2D.
    """
    from .architectures.fno import FNO2D

    model = FNO2D(
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        d_vars=d_vars,
        n_layers=n_layers,
        n_steps=n_steps,
        activation=activation,
        norm=norm,
        use_positions=use_positions,
        linear_conv=linear_conv,
    )

    rng = _default_rng(key)
    H, W = d_model
    dummy = jnp.ones((H, W, in_features))
    params = model.init(rng, dummy, deterministic=True)

    return FlaxModel(model.apply, params, deterministic=True)


def unet2d(
    in_channels: int = 1,
    out_channels: int = 1,
    depth: int = 4,
    wf: int = 6,
    norm: str | None = "batch",
    up_mode: str = "upconv",
    activation: str = "celu",
    padding_mode: str = "circular",
    d_model: Tuple[int, int] = (64, 64),
    *,
    key: jax.Array | None = None,
) -> FlaxModel:
    """Create an initialized 2-D UNet.

    Returns:
        FlaxModel: Initialized UNet2D.
    """
    from .architectures.unet import UNet2D

    model = UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        wf=wf,
        norm=norm,
        up_mode=up_mode,
        activation=activation,
        padding_mode=padding_mode,
    )

    rng = _default_rng(key)
    H, W = d_model
    dummy = jnp.ones((H, W, in_channels))
    variables = model.init(rng, dummy, deterministic=True)

    return FlaxModel(model.apply, variables, deterministic=True)


def transformer(
    encoder_num_layers: int = 4,
    decoder_num_layers: int = 4,
    embed_dim: int = 128,
    num_heads: int = 4,
    qkv_features: int = 128,
    mlp_features: int = 256,
    vocab_size: int = 256,
    max_len: int = 512,
    dropout_rate: float = 0.1,
    seq_len: int = 32,
    *,
    key: jax.Array | None = None,
) -> FlaxModel:
    """Create an initialized Transformer (encoder-decoder).

    Returns:
        FlaxModel: Initialized Transformer.
    """
    from .architectures.transformer import Transformer

    model = Transformer(
        encoder_num_layers=encoder_num_layers,
        decoder_num_layers=decoder_num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        qkv_features=qkv_features,
        mlp_features=mlp_features,
        vocab_size=vocab_size,
        max_len=max_len,
        dropout_rate=dropout_rate,
    )

    rng = _default_rng(key)
    dummy_enc = jnp.ones((seq_len,), dtype=jnp.int32)
    dummy_dec = jnp.ones((seq_len,), dtype=jnp.int32)
    params = model.init(rng, dummy_enc, dummy_dec, deterministic=True)

    return FlaxModel(model.apply, params, deterministic=True)


# =====================================================================
# Foundation models — delegate to per-model modules
# =====================================================================


def poseidonT(*args, **kwargs) -> FlaxModel:
    """Poseidon-T foundation model."""
    from . import poseidon
    return poseidon.T(*args, **kwargs)


def poseidonB(*args, **kwargs) -> FlaxModel:
    """Poseidon-B foundation model."""
    from . import poseidon
    return poseidon.B(*args, **kwargs)


def poseidonL(*args, **kwargs) -> FlaxModel:
    """Poseidon-L foundation model."""
    from . import poseidon
    return poseidon.L(*args, **kwargs)


def morph_Ti(*args, **kwargs) -> FlaxModel:
    """MORPH-Ti foundation model."""
    from . import morph
    return morph.Ti(*args, **kwargs)


def morph_S(*args, **kwargs) -> FlaxModel:
    """MORPH-S foundation model."""
    from . import morph
    return morph.S(*args, **kwargs)


def morph_M(*args, **kwargs) -> FlaxModel:
    """MORPH-M foundation model."""
    from . import morph
    return morph.M(*args, **kwargs)


def morph_L(*args, **kwargs) -> FlaxModel:
    """MORPH-L foundation model."""
    from . import morph
    return morph.L(*args, **kwargs)


def mpp_Ti(*args, **kwargs) -> FlaxModel:
    """MPP-Ti foundation model."""
    from . import mpp
    return mpp.Ti(*args, **kwargs)


def mpp_S(*args, **kwargs) -> FlaxModel:
    """MPP-S foundation model."""
    from . import mpp
    return mpp.S(*args, **kwargs)


def mpp_B(*args, **kwargs) -> FlaxModel:
    """MPP-B foundation model."""
    from . import mpp
    return mpp.B(*args, **kwargs)


def mpp_L(*args, **kwargs) -> FlaxModel:
    """MPP-L foundation model."""
    from . import mpp
    return mpp.L(*args, **kwargs)


def walrus(*args, **kwargs) -> FlaxModel:
    """Walrus foundation model."""
    from . import walrus as _walrus
    return _walrus.base(*args, **kwargs)


def bcat(*args, **kwargs) -> FlaxModel:
    """BCAT foundation model."""
    from . import bcat as _bcat
    return _bcat.base(*args, **kwargs)


def pdeformer2_small(*args, **kwargs) -> FlaxModel:
    """PDEformer2-small foundation model."""
    from . import pdeformer2
    return pdeformer2.small(*args, **kwargs)


def pdeformer2_base(*args, **kwargs) -> FlaxModel:
    """PDEformer2-base foundation model."""
    from . import pdeformer2
    return pdeformer2.base(*args, **kwargs)


def pdeformer2_fast(*args, **kwargs) -> FlaxModel:
    """PDEformer2-fast foundation model."""
    from . import pdeformer2
    return pdeformer2.fast(*args, **kwargs)


def dpot_Ti(*args, **kwargs) -> FlaxModel:
    """DPOT-Ti foundation model."""
    from . import dpot
    return dpot.Ti(*args, **kwargs)


def dpot_S(*args, **kwargs) -> FlaxModel:
    """DPOT-S foundation model."""
    from . import dpot
    return dpot.S(*args, **kwargs)


def dpot_M(*args, **kwargs) -> FlaxModel:
    """DPOT-M foundation model."""
    from . import dpot
    return dpot.M(*args, **kwargs)


def dpot_L(*args, **kwargs) -> FlaxModel:
    """DPOT-L foundation model."""
    from . import dpot
    return dpot.L(*args, **kwargs)


def dpot_H(*args, **kwargs) -> FlaxModel:
    """DPOT-H foundation model."""
    from . import dpot
    return dpot.H(*args, **kwargs)


def prose_fd_1to1(*args, **kwargs) -> FlaxModel:
    """PROSE fd_1to1 foundation model."""
    from . import prose
    return prose.fd_1to1(*args, **kwargs)


def prose_fd_2to1(*args, **kwargs) -> FlaxModel:
    """PROSE fd_2to1 foundation model."""
    from . import prose
    return prose.fd_2to1(*args, **kwargs)


def prose_ode_2to1(*args, **kwargs) -> FlaxModel:
    """PROSE ode_2to1 foundation model."""
    from . import prose
    return prose.ode_2to1(*args, **kwargs)


def prose_pde_2to1(*args, **kwargs) -> FlaxModel:
    """PROSE pde_2to1 foundation model."""
    from . import prose
    return prose.pde_2to1(*args, **kwargs)
