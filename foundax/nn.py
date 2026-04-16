"""Unified factory facade for all foundax models.

Non-foundation factories return a raw ``equinox.Module``.
Foundation-model factories return a :class:`~foundax.model.FlaxModel`.

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

from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from typing import Literal

import jax
import jax.numpy as jnp
import equinox as eqx

from .model import FlaxModel


# ── helpers ──────────────────────────────────────────────────

def _resolve_key(key):
    if key is None:
        return jax.random.PRNGKey(0)
    return key


# =====================================================================
# Non-foundation architectures  (return eqx.Module)
# =====================================================================


def linear(
    in_features: int,
    out_features: int,
    use_bias: bool = True,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a batched Linear layer."""
    from .architectures.linear import Linear
    return Linear(in_features, out_features, use_bias, key=_resolve_key(key))


def mlp(
    in_features: int,
    output_dim: int = 1,
    activation: Callable = jnp.tanh,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    output_activation: Callable | None = None,
    use_bias: bool = True,
    dropout_rate: float = 0.0,
    batch_norm: bool = False,
    layer_norm: bool = False,
    final_layer_bias: bool = True,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a Multi-Layer Perceptron."""
    from .architectures.mlp import MLP
    return MLP(
        in_features,
        output_dim,
        activation,
        hidden_dims,
        num_layers,
        output_activation,
        use_bias,
        dropout_rate,
        batch_norm,
        layer_norm,
        final_layer_bias,
        key=_resolve_key(key),
    )


def fno1d(
    in_features: int,
    hidden_channels: int,
    n_modes: int,
    d_vars: int = 1,
    linear_conv: bool = True,
    n_layers: int = 4,
    n_steps: int = 1,
    activation: Callable = jax.nn.gelu,
    norm: str | None = None,
    training: bool = True,
    dropout_rate: float = 0.0,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 1-D Fourier Neural Operator."""
    from .architectures.fno import FNO1D
    norm_type = norm[0] if isinstance(norm, tuple) else norm
    return FNO1D(
        in_features=in_features,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        d_vars=d_vars,
        linear_conv=linear_conv,
        n_layers=n_layers,
        n_steps=n_steps,
        activation=activation,
        norm=norm_type,
        training=training,
        dropout_rate=dropout_rate,
        key=_resolve_key(key),
    )


def fno2d(
    in_features: int,
    hidden_channels: int = 32,
    n_modes: int = 16,
    d_vars: int = 1,
    n_layers: int = 4,
    n_steps: int = 1,
    d_model: Tuple[int, int] = (64, 64),
    activation: Callable = jax.nn.gelu,
    norm: str | None = "layer",
    training: bool = True,
    use_positions: bool = False,
    linear_conv: bool = True,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 2-D Fourier Neural Operator."""
    from .architectures.fno import FNO2D
    return FNO2D(
        in_features=in_features,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        d_vars=d_vars,
        n_layers=n_layers,
        n_steps=n_steps,
        d_model=d_model,
        activation=activation,
        norm=norm,
        training=training,
        use_positions=use_positions,
        linear_conv=linear_conv,
        key=_resolve_key(key),
    )


def fno3d(
    in_features: int,
    hidden_channels: int = 32,
    n_modes: int = 12,
    d_vars: int = 1,
    n_layers: int = 4,
    n_steps: int = 1,
    d_model: Tuple[int, int, int] = (32, 32, 32),
    activation: Callable = jax.nn.gelu,
    norm: str | None = "layer",
    training: bool = True,
    use_positions: bool = False,
    linear_conv: bool = True,
    dropout_rate: float = 0.0,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 3-D Fourier Neural Operator."""
    from .architectures.fno import FNO3D
    return FNO3D(
        in_features=in_features,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        d_vars=d_vars,
        n_layers=n_layers,
        n_steps=n_steps,
        d_model=d_model,
        activation=activation,
        norm=norm,
        training=training,
        use_positions=use_positions,
        linear_conv=linear_conv,
        dropout_rate=dropout_rate,
        key=_resolve_key(key),
    )


def unet1d(
    in_channels: int = 1,
    out_channels: int = 1,
    depth: int = 4,
    wf: int = 6,
    norm: str = "batch",
    up_mode: str = "upconv",
    groups: int = 1,
    activation: Callable = jax.nn.celu,
    padding_mode: str = "circular",
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 1-D UNet."""
    from .architectures.unet import UNet1D
    return UNet1D(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        wf=wf,
        norm=norm,
        up_mode=up_mode,
        groups=groups,
        activation=activation,
        padding_mode=padding_mode,
        key=_resolve_key(key),
    )


def unet2d(
    in_channels: int = 1,
    out_channels: int = 1,
    depth: int = 4,
    wf: int = 6,
    norm: str = "layer",
    up_mode: str = "upconv",
    activation: Callable = jax.nn.gelu,
    padding_mode: str = "circular",
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 2-D UNet."""
    from .architectures.unet import UNet2D
    return UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        wf=wf,
        norm=norm,
        up_mode=up_mode,
        groups=1,
        activation=activation,
        padding_mode=padding_mode,
        key=_resolve_key(key),
    )


def unet3d(
    in_channels: int = 1,
    out_channels: int = 1,
    depth: int = 4,
    wf: int = 6,
    norm: str = "batch",
    up_mode: str = "upconv",
    activation: str = "celu",
    padding_mode: str = "circular",
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 3-D UNet."""
    from .architectures.unet import UNet3D
    return UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        wf=wf,
        norm=norm,
        up_mode=up_mode,
        groups=1,
        activation=activation,
        padding_mode=padding_mode,
        key=_resolve_key(key),
    )


def transformer(
    num_layers: int = 6,
    embed_dim: int = 512,
    num_heads: int = 8,
    mlp_features: int = 2048,
    dropout_rate: float = 0.1,
    vocab_size: int = 10000,
    max_len: int = 128,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a Transformer (encoder-decoder)."""
    from .architectures.transformer import Transformer
    return Transformer(
        encoder_num_layers=num_layers,
        decoder_num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        qkv_features=embed_dim,
        mlp_features=mlp_features,
        vocab_size=vocab_size,
        dropout_rate=dropout_rate,
        max_len=max_len,
        key=_resolve_key(key),
    )


def deeponet(
    branch_type: Literal["mlp", "resmlp", "conv1d", "transformer"] = "mlp",
    trunk_type: Literal["mlp", "resmlp", "siren"] = "mlp",
    combination_type: Literal["dot", "bilinear", "mlp", "attention"] = "dot",
    n_sensors: int = 100,
    sensor_channels: int = 1,
    coord_dim: int = 1,
    n_outputs: int = 1,
    basis_functions: int = 128,
    hidden_dim: int = 256,
    n_layers: int = 4,
    n_heads: int = 8,
    coord_embedding: Optional[Literal["fourier", "positional"]] = None,
    coord_embedding_dim: int = 64,
    coord_embedding_scale: float = 1.0,
    activation: Callable = jax.nn.gelu,
    norm: Optional[str] = None,
    dropout_rate: float = 0.0,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a Deep Operator Network (DeepONet)."""
    from .architectures.deeponet import DeepONet
    hidden_dims = tuple([hidden_dim] * n_layers)
    return DeepONet(
        branch_type=branch_type,
        trunk_type=trunk_type,
        combination_type=combination_type,
        n_sensors=n_sensors,
        sensor_channels=sensor_channels,
        coord_dim=coord_dim,
        n_outputs=n_outputs,
        basis_functions=basis_functions,
        branch_hidden_dims=hidden_dims,
        branch_hidden_dim=hidden_dim,
        branch_n_blocks=n_layers,
        branch_n_layers=n_layers,
        branch_n_heads=n_heads,
        trunk_hidden_dims=hidden_dims,
        trunk_hidden_dim=hidden_dim,
        trunk_n_blocks=n_layers,
        coord_embedding=coord_embedding,
        coord_embedding_dim=coord_embedding_dim,
        coord_embedding_scale=coord_embedding_scale,
        combination_hidden_dims=(hidden_dim, hidden_dim // 2),
        combination_d_model=hidden_dim,
        combination_n_heads=n_heads,
        activation=activation,
        norm=norm,
        dropout_rate=dropout_rate,
        key=_resolve_key(key),
    )


def cno2d(
    in_dim: int = 1,
    out_dim: int = 1,
    size: int = 64,
    N_layers: int = 3,
    N_res: int = 4,
    N_res_neck: int = 4,
    channel_multiplier: int = 16,
    use_bn: bool = True,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 2-D Continuous Neural Operator (CNO)."""
    from .architectures.cno import CNO2D
    return CNO2D(
        in_dim=in_dim,
        out_dim=out_dim,
        size=size,
        N_layers=N_layers,
        N_res=N_res,
        N_res_neck=N_res_neck,
        channel_multiplier=channel_multiplier,
        use_bn=use_bn,
        key=_resolve_key(key),
    )


def mgno1d(
    input_length: int,
    num_layer: int = 5,
    num_channel_u: int = 24,
    num_channel_f: int = 3,
    num_iteration: Optional[List[Tuple[int, int]]] = None,
    output_dim: int = 1,
    activation: str = "gelu",
    padding_mode: str = "CIRCULAR",
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 1-D Multigrid Neural Operator."""
    from .architectures.mgno import MgNO1D
    if num_iteration is None:
        num_iteration = [(1, 1)] * 5
    return MgNO1D(
        input_length=input_length,
        num_layer=num_layer,
        num_channel_u=num_channel_u,
        num_channel_f=num_channel_f,
        num_iteration=num_iteration,
        output_dim=output_dim,
        activation=activation,
        padding_mode=padding_mode,
        key=_resolve_key(key),
    )


def mgno2d(
    input_shape: Tuple[int, int],
    num_layer: int = 5,
    num_channel_u: int = 24,
    num_channel_f: int = 3,
    num_iteration: Optional[List[Tuple[int, int]]] = None,
    output_dim: int = 1,
    activation: str = "gelu",
    padding_mode: str = "CIRCULAR",
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 2-D Multigrid Neural Operator."""
    from .architectures.mgno import MgNO
    if num_iteration is None:
        num_iteration = [(1, 1)] * 5
    return MgNO(
        input_shape=input_shape,
        num_layer=num_layer,
        num_channel_u=num_channel_u,
        num_channel_f=num_channel_f,
        num_iteration=num_iteration,
        output_dim=output_dim,
        activation=activation,
        padding_mode=padding_mode,
        key=_resolve_key(key),
    )


def geofno(
    ndims: int,
    nks: Sequence[int],
    Ls: Sequence[float],
    layers: Sequence[int] = (64, 64, 64, 64),
    fc_dim: int = 128,
    in_dim: int = 3,
    out_dim: int = 1,
    act: str = "gelu",
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a Geometry-aware Fourier Neural Operator."""
    from .architectures.geofno import GeoFNO
    from .architectures.common import compute_Fourier_modes as _compute_modes
    modes = _compute_modes(ndims, list(nks), list(Ls))
    modes = jnp.array(modes)
    return GeoFNO(
        ndims=ndims,
        modes=modes,
        layers=list(layers),
        fc_dim=fc_dim,
        in_dim=in_dim,
        out_dim=out_dim,
        act=act,
        key=_resolve_key(key),
    )


def pcno(
    ndims: int,
    nks: Sequence[int],
    Ls: Sequence[float],
    layers: Sequence[int] = (64, 64, 64, 64),
    fc_dim: int = 128,
    in_dim: int = 3,
    out_dim: int = 1,
    inv_L_scale_min: float = 0.5,
    inv_L_scale_max: float = 2.0,
    train_inv_L_scale: bool = True,
    act: str = "gelu",
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a Point Cloud Neural Operator."""
    from .architectures.pcno import PCNO
    from .architectures.pcno import compute_Fourier_modes as _pcno_modes
    modes = _pcno_modes(ndims, nks, Ls)
    modes = jnp.array(modes)
    nmeasures = len(nks) // ndims
    return PCNO(
        ndims=ndims,
        modes=modes,
        nmeasures=nmeasures,
        layers=list(layers),
        fc_dim=fc_dim,
        in_dim=in_dim,
        out_dim=out_dim,
        inv_L_scale_min=inv_L_scale_min,
        inv_L_scale_max=inv_L_scale_max,
        train_inv_L_scale=train_inv_L_scale,
        act=act,
        key=_resolve_key(key),
    )


def cgptno(
    trunk_size: int,
    branch_sizes: Optional[List[int]] = None,
    output_size: int = 1,
    n_layers: int = 2,
    n_hidden: int = 64,
    n_head: int = 1,
    n_inner: int = 4,
    mlp_layers: int = 2,
    attn_type: str = "linear",
    act: str = "gelu",
    ffn_dropout: float = 0.0,
    attn_dropout: float = 0.0,
    horiz_fourier_dim: int = 0,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a Cross-attention GPT Neural Operator (CGPTNO)."""
    from .architectures.gnot import CGPTNO
    return CGPTNO(
        trunk_size=trunk_size,
        branch_sizes=branch_sizes,
        output_size=output_size,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_head=n_head,
        n_inner=n_inner,
        mlp_layers=mlp_layers,
        attn_type=attn_type,
        act=act,
        ffn_dropout=ffn_dropout,
        attn_dropout=attn_dropout,
        horiz_fourier_dim=horiz_fourier_dim,
        key=_resolve_key(key),
    )


def gnot(
    trunk_size: int,
    branch_sizes: List[int],
    space_dim: int = 2,
    output_size: int = 1,
    n_layers: int = 2,
    n_hidden: int = 64,
    n_head: int = 1,
    n_experts: int = 2,
    n_inner: int = 4,
    mlp_layers: int = 2,
    attn_type: str = "linear",
    act: str = "gelu",
    ffn_dropout: float = 0.0,
    attn_dropout: float = 0.0,
    horiz_fourier_dim: int = 0,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a General Neural Operator Transformer (GNOT)."""
    from .architectures.gnot import GNOT
    return GNOT(
        trunk_size=trunk_size,
        branch_sizes=branch_sizes,
        space_dim=space_dim,
        output_size=output_size,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_head=n_head,
        n_experts=n_experts,
        n_inner=n_inner,
        mlp_layers=mlp_layers,
        attn_type=attn_type,
        act=act,
        ffn_dropout=ffn_dropout,
        attn_dropout=attn_dropout,
        horiz_fourier_dim=horiz_fourier_dim,
        key=_resolve_key(key),
    )


def moegptno(
    trunk_size: int,
    branch_size: int,
    space_dim: int = 2,
    output_size: int = 1,
    n_layers: int = 2,
    n_hidden: int = 64,
    n_head: int = 1,
    n_experts: int = 2,
    mlp_layers: int = 2,
    attn_type: str = "linear",
    act: str = "gelu",
    ffn_dropout: float = 0.0,
    attn_dropout: float = 0.0,
    horiz_fourier_dim: int = 0,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a single-input MoE GPT Neural Operator."""
    from .architectures.gnot import MoEGPTNO
    return MoEGPTNO(
        trunk_size=trunk_size,
        branch_size=branch_size,
        space_dim=space_dim,
        output_size=output_size,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_head=n_head,
        n_experts=n_experts,
        mlp_layers=mlp_layers,
        attn_type=attn_type,
        act=act,
        ffn_dropout=ffn_dropout,
        attn_dropout=attn_dropout,
        horiz_fourier_dim=horiz_fourier_dim,
        key=_resolve_key(key),
    )


def pit(
    in_channels: int,
    out_channels: int,
    hid_channels: int = 256,
    n_head: int = 8,
    localities: Sequence[float] = (100, 50, 50, 50, 100),
    input_res: Optional[Tuple[int, int]] = (64, 64),
    latent_res: Optional[Tuple[int, int]] = (16, 16),
    output_res: Optional[Tuple[int, int]] = (64, 64),
    m_dists: Optional[Sequence[jnp.ndarray]] = None,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a Position-induced Transformer (PiT)."""
    from .architectures.pit import PiT as _PiT, PiTWithCoords
    if m_dists is not None:
        return _PiT(
            in_channels=in_channels,
            out_channels=out_channels,
            hid_channels=hid_channels,
            n_head=n_head,
            localities=list(localities),
            m_dists=m_dists,
            key=_resolve_key(key),
        )
    else:
        return PiTWithCoords(
            in_channels=in_channels,
            out_channels=out_channels,
            hid_channels=hid_channels,
            n_head=n_head,
            localities=list(localities),
            input_res=input_res,
            latent_res=latent_res,
            output_res=output_res,
            key=_resolve_key(key),
        )


def pointnet(
    in_features: int,
    output_dim: int,
    hidden_dims: List[int] = [32, 16, 8, 4, 2, 2, 4, 8, 8],
    dropout_rate: float = 0.0,
    feature_transform: Optional[Callable] = None,
    activation_function: Callable = jnp.tanh,
    use_bias: bool = True,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a PointNet-style network."""
    from .architectures.pointnet import PointNet
    return PointNet(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        feature_transform=feature_transform,
        act=activation_function,
        use_bias=use_bias,
        key=_resolve_key(key),
    )


# =====================================================================
# Foundation models — delegate to per-model modules
# =====================================================================


def poseidonT(
    rng: Optional[jax.Array] = None,
    weight_path: Optional[str] = None,
    verbose: bool = True,
) -> FlaxModel:
    """Poseidon-T (Tiny) foundation model (~20.8 M params).

    Scalable Operator Transformer (ScOT) with Swin-Transformer backbone.

    Parameters
    ----------
    rng : jax.random.PRNGKey, optional
        PRNG key.  Needed together with *weight_path* to load weights.
    weight_path : str, optional
        Path to ``.msgpack`` checkpoint.
    verbose : bool, default True
        Print model / loading info.

    Returns
    -------
    ScOT or tuple[ScOT, dict]
        Model (and loaded params when *weight_path* + *rng* are given).

    References
    ----------
    Herde et al., "Poseidon: Efficient Foundation Models for PDEs", 2024.
    https://arxiv.org/abs/2405.19101
    """
    from . import poseidon
    return poseidon.T(rng=rng, weight_path=weight_path, verbose=verbose)


def poseidonB(
    rng: Optional[jax.Array] = None,
    weight_path: Optional[str] = None,
    verbose: bool = True,
) -> FlaxModel:
    """Poseidon-B (Base) foundation model (~157.7 M params).

    Scalable Operator Transformer (ScOT) with Swin-Transformer backbone.

    Parameters
    ----------
    rng : jax.random.PRNGKey, optional
        PRNG key.
    weight_path : str, optional
        Path to ``.msgpack`` checkpoint.
    verbose : bool, default True
        Print model / loading info.

    Returns
    -------
    ScOT or tuple[ScOT, dict]

    References
    ----------
    Herde et al., "Poseidon", 2024.  https://arxiv.org/abs/2405.19101
    """
    from . import poseidon
    return poseidon.B(rng=rng, weight_path=weight_path, verbose=verbose)


def poseidonL(
    rng: Optional[jax.Array] = None,
    weight_path: Optional[str] = None,
    verbose: bool = True,
) -> FlaxModel:
    """Poseidon-L (Large) foundation model (~628.6 M params).

    Scalable Operator Transformer (ScOT) with Swin-Transformer backbone.

    Parameters
    ----------
    rng : jax.random.PRNGKey, optional
        PRNG key.
    weight_path : str, optional
        Path to ``.msgpack`` checkpoint.
    verbose : bool, default True
        Print model / loading info.

    Returns
    -------
    ScOT or tuple[ScOT, dict]

    References
    ----------
    Herde et al., "Poseidon", 2024.  https://arxiv.org/abs/2405.19101
    """
    from . import poseidon
    return poseidon.L(rng=rng, weight_path=weight_path, verbose=verbose)


def morph_Ti(**overrides) -> FlaxModel:
    """MORPH-Ti (Tiny) — ~9.9 M params.

    ViT3DRegression: embed=256, depth=4, heads=4, mlp=1024.

    Parameters
    ----------
    **overrides
        Override any ``ViT3DRegression`` attribute (e.g. ``dropout``,
        ``max_patches``).

    References
    ----------
    Rautela et al., "MORPH", 2025.  https://arxiv.org/abs/2509.21670
    """
    from . import morph
    return morph.Ti(**overrides)


def morph_S(**overrides) -> FlaxModel:
    """MORPH-S (Small) — ~32.8 M params.

    ViT3DRegression: embed=512, depth=4, heads=8, mlp=2048.

    Parameters
    ----------
    **overrides
        Override any ``ViT3DRegression`` attribute.

    References
    ----------
    Rautela et al., "MORPH", 2025.  https://arxiv.org/abs/2509.21670
    """
    from . import morph
    return morph.S(**overrides)


def morph_M(**overrides) -> FlaxModel:
    """MORPH-M (Medium) — ~125.6 M params.

    ViT3DRegression: embed=768, depth=8, heads=12, mlp=3072.

    Parameters
    ----------
    **overrides
        Override any ``ViT3DRegression`` attribute.

    References
    ----------
    Rautela et al., "MORPH", 2025.  https://arxiv.org/abs/2509.21670
    """
    from . import morph
    return morph.M(**overrides)


def morph_L(**overrides) -> FlaxModel:
    """MORPH-L (Large) — ~483.3 M params.

    ViT3DRegression: embed=1024, depth=16, heads=16, mlp=4096, max_ar=16.

    Parameters
    ----------
    **overrides
        Override any ``ViT3DRegression`` attribute.

    References
    ----------
    Rautela et al., "MORPH", 2025.  https://arxiv.org/abs/2509.21670
    """
    from . import morph
    return morph.L(**overrides)


def mpp_Ti(**overrides) -> FlaxModel:
    """MPP AViT-Tiny — ~5.5 M params.

    AViT: embed=192, heads=3, blocks=12, n_states=12.

    Parameters
    ----------
    **overrides
        Override any ``AViT`` attribute.

    References
    ----------
    McCabe et al., NeurIPS 2024.
    https://openreview.net/forum?id=DKSI3bULiZ
    """
    from . import mpp
    return mpp.Ti(**overrides)


def mpp_S(**overrides) -> FlaxModel:
    """MPP AViT-Small — ~21 M params.

    AViT: embed=384, heads=6, blocks=12, n_states=12.

    Parameters
    ----------
    **overrides
        Override any ``AViT`` attribute.

    References
    ----------
    McCabe et al., NeurIPS 2024.
    https://openreview.net/forum?id=DKSI3bULiZ
    """
    from . import mpp
    return mpp.S(**overrides)


def mpp_B(**overrides) -> FlaxModel:
    """MPP AViT-Base — ~83 M params.

    AViT: embed=768, heads=12, blocks=12, n_states=12.

    Parameters
    ----------
    **overrides
        Override any ``AViT`` attribute.

    References
    ----------
    McCabe et al., NeurIPS 2024.
    https://openreview.net/forum?id=DKSI3bULiZ
    """
    from . import mpp
    return mpp.B(**overrides)


def mpp_L(**overrides) -> FlaxModel:
    """MPP AViT-Large — ~300 M params.

    AViT: embed=1024, heads=16, blocks=24, n_states=12.

    Parameters
    ----------
    **overrides
        Override any ``AViT`` attribute.

    References
    ----------
    McCabe et al., NeurIPS 2024.
    https://openreview.net/forum?id=DKSI3bULiZ
    """
    from . import mpp
    return mpp.L(**overrides)


def walrus(**kwargs) -> FlaxModel:
    """Walrus foundation model (~1.29 B params).

    ``IsotropicModel``: hidden=768, blocks=12, heads=12, n_states=4.

    Parameters
    ----------
    **kwargs
        Override any ``IsotropicModel`` attribute (see
        ``foundax.walrus.base`` for the full list).

    References
    ----------
    https://github.com/nubskr/walrus
    """
    from . import walrus as _walrus
    return _walrus.base(**kwargs)


def bcat() -> FlaxModel:
    """BCAT foundation model (Block Causal Transformer).

    Default config: 12 layers, dim_emb=1024, 8 heads, SwiGLU, RMSNorm.

    Takes no arguments — uses default ``BCATConfig``.

    Forward call: ``model(data, times, input_len=10)``
    - data: ``(bs, input_len+output_len, x_num, x_num, data_dim)``
    - times: ``(bs, input_len+output_len, 1)``

    References
    ----------
    https://arxiv.org/abs/2501.18972
    """
    from . import bcat as _bcat
    return _bcat.base()


def pdeformer2_small(dtype=None) -> FlaxModel:
    """PDEformer-2 Small (~27.7 M params).

    Graphormer(9 layers, embed=512) + PolyINR(hidden=128).

    Parameters
    ----------
    dtype : jnp.dtype, optional
        JAX dtype.  Defaults to ``jnp.float32``.

    References
    ----------
    Shi et al., "PDEformer-2", 2025.  https://arxiv.org/abs/2502.14844
    """
    from . import pdeformer2
    return pdeformer2.small(dtype=dtype)


def pdeformer2_base(dtype=None) -> FlaxModel:
    """PDEformer-2 Base.

    Graphormer(12 layers, embed=768) + PolyINR(hidden=768).

    Parameters
    ----------
    dtype : jnp.dtype, optional
        JAX dtype.  Defaults to ``jnp.float32``.

    References
    ----------
    Shi et al., "PDEformer-2", 2025.  https://arxiv.org/abs/2502.14844
    """
    from . import pdeformer2
    return pdeformer2.base(dtype=dtype)


def pdeformer2_fast(dtype=None) -> FlaxModel:
    """PDEformer-2 Fast.

    Graphormer(12 layers, embed=768) + PolyINR(hidden=256) — smaller
    INR than Base for faster inference.

    Parameters
    ----------
    dtype : jnp.dtype, optional
        JAX dtype.  Defaults to ``jnp.float32``.

    References
    ----------
    Shi et al., "PDEformer-2", 2025.  https://arxiv.org/abs/2502.14844
    """
    from . import pdeformer2
    return pdeformer2.fast(dtype=dtype)


def dpot_Ti() -> FlaxModel:
    """DPOTNet-Ti (Tiny).

    Config: embed=512, depth=4, n_blocks=4, img_size=128, patch_size=8.
    """
    from . import dpot
    return dpot.Ti()


def dpot_S() -> FlaxModel:
    """DPOTNet-S (Small).

    Config: embed=1024, depth=6, n_blocks=8, img_size=128, patch_size=8.
    """
    from . import dpot
    return dpot.S()


def dpot_M() -> FlaxModel:
    """DPOTNet-M (Medium).

    Config: embed=1024, depth=12, n_blocks=8, img_size=128, patch_size=8.
    """
    from . import dpot
    return dpot.M()


def dpot_L() -> FlaxModel:
    """DPOTNet-L (Large).

    Config: embed=1536, depth=24, n_blocks=16, img_size=128, patch_size=8.
    """
    from . import dpot
    return dpot.L()


def dpot_H() -> FlaxModel:
    """DPOTNet-H (Huge).

    Config: embed=2048, depth=27, n_blocks=8, img_size=128, patch_size=8.
    """
    from . import dpot
    return dpot.H()


def prose_fd_1to1(
    config=None,
    x_num: int = 128,
    max_output_dim: int = 4,
    input_len: int = 10,
    output_len: int = 10,
) -> FlaxModel:
    """PROSE finite-difference 1-to-1 model.

    Parameters
    ----------
    config : PROSE1to1Config, optional
        Full config (internal defaults if ``None``).
    x_num : int, default 128
        Spatial grid size.
    max_output_dim : int, default 4
        Output dimensionality.
    input_len : int, default 10
        Input time-steps.
    output_len : int, default 10
        Output time-steps.

    Returns
    -------
    tuple[PROSE1to1, dict]
        ``(model, params)`` — already initialised.
    """
    from . import prose
    return prose.fd_1to1(
        config=config,
        x_num=x_num,
        max_output_dim=max_output_dim,
        input_len=input_len,
        output_len=output_len,
    )


def prose_fd_2to1(
    n_words: int,
    x_num: int = 128,
    max_output_dim: int = 4,
    input_len: int = 10,
    output_len: int = 10,
    symbol_len: int = 48,
) -> FlaxModel:
    """PROSE finite-difference 2-to-1 model.

    Parameters
    ----------
    n_words : int
        Vocabulary size (**required**).
    x_num : int, default 128
        Spatial grid size.
    max_output_dim : int, default 4
        Output dimensionality.
    input_len : int, default 10
        Input time-steps.
    output_len : int, default 10
        Output time-steps.
    symbol_len : int, default 48
        Symbol sequence length.

    Returns
    -------
    tuple[PROSE2to1, dict]
        ``(model, params)`` — already initialised.
    """
    from . import prose
    return prose.fd_2to1(
        n_words=n_words,
        x_num=x_num,
        max_output_dim=max_output_dim,
        input_len=input_len,
        output_len=output_len,
        symbol_len=symbol_len,
    )


def prose_ode_2to1(
    n_words: int,
    pad_index: int,
    max_output_dimension: int = 3,
    input_len: int = 50,
    output_len: int = 50,
    text_len: int = 48,
    cfg=None,
) -> FlaxModel:
    """PROSE ODE 2-to-1 model.

    Parameters
    ----------
    n_words : int
        Vocabulary size (**required**).
    pad_index : int
        Padding token index (**required**).
    max_output_dimension : int, default 3
        Max output dim.
    input_len : int, default 50
        Input sequence length.
    output_len : int, default 50
        Output sequence length.
    text_len : int, default 48
        Text / symbol sequence length.
    cfg : ProseTextData2to1Config, optional
        Full config.

    Returns
    -------
    tuple[PROSEODE2to1, dict]
        ``(model, params)`` — already initialised.
    """
    from . import prose
    return prose.ode_2to1(
        n_words=n_words,
        pad_index=pad_index,
        max_output_dimension=max_output_dimension,
        input_len=input_len,
        output_len=output_len,
        text_len=text_len,
        cfg=cfg,
    )


def prose_pde_2to1(
    n_words: int,
    pad_index: int,
    max_output_dimension: int = 1,
    x_patch_size: int = 1,
    x_grid_size: int = 128,
    input_len: int = 10,
    output_len: int = 10,
    text_len: int = 48,
    cfg=None,
) -> FlaxModel:
    """PROSE PDE 2-to-1 model.

    Parameters
    ----------
    n_words : int
        Vocabulary size (**required**).
    pad_index : int
        Padding token index (**required**).
    max_output_dimension : int, default 1
        Max output dim.
    x_patch_size : int, default 1
        Spatial patch size.
    x_grid_size : int, default 128
        Spatial grid size.
    input_len : int, default 10
        Input sequence length.
    output_len : int, default 10
        Output sequence length.
    text_len : int, default 48
        Text / symbol sequence length.
    cfg : ProseTextData2to1Config, optional
        Full config.

    Returns
    -------
    tuple[PROSEPDE2to1, dict]
        ``(model, params)`` — already initialised.
    """
    from . import prose
    return prose.pde_2to1(
        n_words=n_words,
        pad_index=pad_index,
        max_output_dimension=max_output_dimension,
        x_patch_size=x_patch_size,
        x_grid_size=x_grid_size,
        input_len=input_len,
        output_len=output_len,
        text_len=text_len,
        cfg=cfg,
    )
