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
