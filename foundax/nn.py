"""Unified factory facade for all foundax models.

Non-foundation factories return a raw ``equinox.Module``.
Foundation-model factories return an :class:`equinox.Module`.

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
    """Create a batched Linear layer.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Add a learnable bias vector.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (Linear).
    """
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
    """Create a Multi-Layer Perceptron.

    Args:
        in_features: Number of input features.
        output_dim: Dimensionality of the output.
        activation: Activation function applied after each hidden layer.
        hidden_dims: Width(s) of hidden layers. An ``int`` uses the same
            width for every hidden layer.
        num_layers: Number of hidden layers.
        output_activation: Activation applied to the final output
            (``None`` = identity).
        use_bias: Add learnable biases in hidden layers.
        dropout_rate: Dropout probability (``0`` = disabled).
        batch_norm: Apply batch normalisation after each hidden layer.
        layer_norm: Apply layer normalisation after each hidden layer.
        final_layer_bias: Add a bias to the output projection.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (MLP).
    """
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
    """Create a 1-D Fourier Neural Operator.

    Learns operators between function spaces on 1-D domains by
    combining spectral convolutions (in Fourier space) with pointwise
    nonlinearities in physical space.

    Args:
        in_features: Number of input features per grid point.
        hidden_channels: Width of the latent representation.
        n_modes: Number of Fourier modes retained.
        d_vars: Number of dependent variables (output channels).
        linear_conv: Use a linear (1×1) convolution bypass.
        n_layers: Number of Fourier layers.
        n_steps: Number of auto-regressive forward steps.
        activation: Pointwise activation function.
        norm: Normalisation type (``None``, ``"instance"``,
            ``"batch"``, …).
        training: Whether the model is in training mode (affects
            dropout / batchnorm).
        dropout_rate: Dropout probability (``0`` = disabled).
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (FNO1D).
    """
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
    """Create a 2-D Fourier Neural Operator.

    Learns operators between function spaces on 2-D domains by
    combining spectral convolutions (in Fourier space) with pointwise
    nonlinearities in physical space.

    Args:
        in_features: Number of input features per grid point.
        hidden_channels: Width of the latent representation.
        n_modes: Number of Fourier modes retained per dimension.
        d_vars: Number of dependent variables (output channels).
        n_layers: Number of Fourier layers.
        n_steps: Number of auto-regressive forward steps.
        d_model: Spatial resolution of the grid ``(H, W)``.
        activation: Pointwise activation function.
        norm: Normalisation type (``None``, ``"layer"``, …).
        training: Whether the model is in training mode.
        use_positions: Append grid coordinates to the input.
        linear_conv: Use a linear (1×1) convolution bypass.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (FNO2D).
    """
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
    """Create a 3-D Fourier Neural Operator.

    Learns operators between function spaces on 3-D domains by
    combining spectral convolutions (in Fourier space) with pointwise
    nonlinearities in physical space.

    Args:
        in_features: Number of input features per grid point.
        hidden_channels: Width of the latent representation.
        n_modes: Number of Fourier modes retained per dimension.
        d_vars: Number of dependent variables (output channels).
        n_layers: Number of Fourier layers.
        n_steps: Number of auto-regressive forward steps.
        d_model: Spatial resolution ``(D, H, W)``.
        activation: Pointwise activation function.
        norm: Normalisation type (``None``, ``"layer"``, …).
        training: Whether the model is in training mode.
        use_positions: Append grid coordinates to the input.
        linear_conv: Use a linear (1×1) convolution bypass.
        dropout_rate: Dropout probability (``0`` = disabled).
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (FNO3D).
    """
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
    """Create a 1-D UNet.

    Encoder-decoder architecture with skip connections for 1-D
    operator learning.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        depth: Number of down-/up-sampling stages.
        wf: Width factor — the first encoder stage has ``2**wf``
            filters (doubles per stage).
        norm: Normalisation type (``"batch"``, ``"layer"``,
            ``"instance"``, ``"none"``).
        up_mode: Upsampling method (``"upconv"`` = transposed
            convolution, ``"upsample"`` = nearest + conv).
        groups: Number of groups for grouped convolutions.
        activation: Activation function.
        padding_mode: Padding mode (``"circular"``, ``"zeros"``,
            ``"reflect"``).
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (UNet1D).
    """
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
    """Create a 2-D UNet.

    Encoder-decoder architecture with skip connections for 2-D
    operator learning.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        depth: Number of down-/up-sampling stages.
        wf: Width factor — the first encoder stage has ``2**wf``
            filters (doubles per stage).
        norm: Normalisation type (``"batch"``, ``"layer"``,
            ``"instance"``, ``"none"``).
        up_mode: Upsampling method (``"upconv"`` = transposed
            convolution, ``"upsample"`` = nearest + conv).
        activation: Activation function.
        padding_mode: Padding mode (``"circular"``, ``"zeros"``,
            ``"reflect"``).
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (UNet2D).
    """
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
    """Create a 3-D UNet.

    Encoder-decoder architecture with skip connections for 3-D
    operator learning.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        depth: Number of down-/up-sampling stages.
        wf: Width factor — the first encoder stage has ``2**wf``
            filters (doubles per stage).
        norm: Normalisation type (``"batch"``, ``"layer"``,
            ``"instance"``, ``"none"``).
        up_mode: Upsampling method (``"upconv"`` = transposed
            convolution, ``"upsample"`` = nearest + conv).
        activation: Activation function or name string.
        padding_mode: Padding mode (``"circular"``, ``"zeros"``,
            ``"reflect"``).
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (UNet3D).
    """
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
    """Create a Transformer (encoder-decoder).

    Standard encoder-decoder Transformer for structured sequence
    prediction.

    Args:
        num_layers: Number of encoder *and* decoder layers.
        embed_dim: Model embedding dimension.
        num_heads: Number of attention heads.
        mlp_features: Feedforward hidden dimension.
        dropout_rate: Dropout probability.
        vocab_size: Size of the input/output vocabulary.
        max_len: Maximum sequence length for positional encoding.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (Transformer).
    """
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
    """Create a Deep Operator Network (DeepONet).

    Learns nonlinear operators by decomposing them into a *branch*
    network (encodes the input function) and a *trunk* network
    (encodes the query coordinates).

    Args:
        branch_type: Branch network architecture
            (``"mlp"``, ``"resmlp"``, ``"conv1d"``, ``"transformer"``).
        trunk_type: Trunk network architecture
            (``"mlp"``, ``"resmlp"``, ``"siren"``).
        combination_type: How branch and trunk are combined
            (``"dot"``, ``"bilinear"``, ``"mlp"``, ``"attention"``).
        n_sensors: Number of sensor (input) locations.
        sensor_channels: Number of channels per sensor.
        coord_dim: Dimensionality of the query coordinates.
        n_outputs: Number of output fields.
        basis_functions: Number of basis functions (rank).
        hidden_dim: Hidden dimension shared across branch/trunk/MLP.
        n_layers: Depth of branch and trunk sub-networks.
        n_heads: Number of attention heads (for transformer branch or
            attention combiner).
        coord_embedding: Optional coordinate encoding
            (``"fourier"``, ``"positional"``, ``None``).
        coord_embedding_dim: Dimension of the coordinate embedding.
        coord_embedding_scale: Scale factor for Fourier features.
        activation: Activation function.
        norm: Normalisation type (``None``, ``"layer"``, …).
        dropout_rate: Dropout probability.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (DeepONet).
    """
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
    """Create a 2-D Continuous Neural Operator (CNO).

    Resolution-invariant operator that uses continuous-domain
    convolutions with learnable bandwidth.

    Args:
        in_dim: Number of input channels.
        out_dim: Number of output channels.
        size: Base spatial resolution.
        N_layers: Number of encoder/decoder layers.
        N_res: Number of residual blocks per layer.
        N_res_neck: Number of residual blocks in the bottleneck.
        channel_multiplier: Channel width multiplier.
        use_bn: Apply batch normalisation.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (CNO2D).
    """
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
    """Create a 1-D Multigrid Neural Operator.

    Multi-resolution operator that applies iterative smoothing across
    grids of different resolutions.

    Args:
        input_length: Length of the 1-D input grid.
        num_layer: Number of multigrid layers.
        num_channel_u: Number of channels in the solution variable.
        num_channel_f: Number of channels in the forcing / input.
        num_iteration: Per-layer iteration counts as
            ``[(pre, post), ...]``.
        output_dim: Number of output channels.
        activation: Activation function name.
        padding_mode: Padding mode (``"CIRCULAR"``, ``"CONSTANT"``).
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (MgNO1D).
    """
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
    """Create a 2-D Multigrid Neural Operator.

    Multi-resolution operator that applies iterative smoothing across
    grids of different resolutions.

    Args:
        input_shape: Spatial shape of the 2-D input ``(H, W)``.
        num_layer: Number of multigrid layers.
        num_channel_u: Number of channels in the solution variable.
        num_channel_f: Number of channels in the forcing / input.
        num_iteration: Per-layer iteration counts as
            ``[(pre, post), ...]``.
        output_dim: Number of output channels.
        activation: Activation function name.
        padding_mode: Padding mode (``"CIRCULAR"``, ``"CONSTANT"``).
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (MgNO).
    """
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
    """Create a Geometry-aware Fourier Neural Operator.

    FNO variant that handles irregular geometries by mapping them
    to a reference domain before applying spectral convolutions.

    Args:
        ndims: Spatial dimensionality (1, 2, or 3).
        nks: Grid sizes per dimension (used for Fourier-mode
            computation).
        Ls: Physical domain lengths per dimension.
        layers: Hidden channel widths per Fourier layer.
        fc_dim: Width of the final fully-connected projection.
        in_dim: Number of input channels (incl. coordinates).
        out_dim: Number of output channels.
        act: Activation function name.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (GeoFNO).
    """
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
    """Create a Point Cloud Neural Operator.

    Operator learning on unstructured point clouds via
    non-uniform Fourier features.

    Args:
        ndims: Spatial dimensionality.
        nks: Grid sizes per dimension (Fourier-mode computation).
        Ls: Physical domain lengths per dimension.
        layers: Hidden channel widths per layer.
        fc_dim: Width of the final fully-connected projection.
        in_dim: Number of input channels (incl. coordinates).
        out_dim: Number of output channels.
        inv_L_scale_min: Minimum inverse-length scale.
        inv_L_scale_max: Maximum inverse-length scale.
        train_inv_L_scale: Make the inverse-length scale trainable.
        act: Activation function name.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (PCNO).
    """
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
    """Create a Cross-attention GPT Neural Operator (CGPTNO).

    Combines cross-attention between trunk (query coordinates) and
    branch (input function) with auto-regressive GPT-style blocks.

    Args:
        trunk_size: Dimension of the trunk (query) input.
        branch_sizes: Dimensions of each branch input (``None`` =
            single branch of size ``trunk_size``).
        output_size: Number of output channels.
        n_layers: Number of transformer layers.
        n_hidden: Hidden dimension.
        n_head: Number of attention heads.
        n_inner: Inner dimension multiplier for the FFN.
        mlp_layers: Number of MLP layers in the output head.
        attn_type: Attention type (``"linear"``, ``"softmax"``).
        act: Activation function name.
        ffn_dropout: FFN dropout probability.
        attn_dropout: Attention dropout probability.
        horiz_fourier_dim: Fourier feature dimension for horizontal
            encoding (``0`` = disabled).
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (CGPTNO).
    """
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
    """Create a General Neural Operator Transformer (GNOT).

    Multi-input neural operator using mixture-of-experts and
    cross-attention for heterogeneous PDE problems.

    Args:
        trunk_size: Dimension of the trunk (query) input.
        branch_sizes: Dimensions of each branch input.
        space_dim: Spatial dimensionality.
        output_size: Number of output channels.
        n_layers: Number of transformer layers.
        n_hidden: Hidden dimension.
        n_head: Number of attention heads.
        n_experts: Number of mixture-of-experts in the FFN.
        n_inner: Inner dimension multiplier for the FFN.
        mlp_layers: Number of MLP layers in the output head.
        attn_type: Attention type (``"linear"``, ``"softmax"``).
        act: Activation function name.
        ffn_dropout: FFN dropout probability.
        attn_dropout: Attention dropout probability.
        horiz_fourier_dim: Fourier feature dimension for horizontal
            encoding (``0`` = disabled).
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (GNOT).
    """
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
    """Create a single-input MoE GPT Neural Operator.

    Single-branch variant of GNOT with mixture-of-experts.

    Args:
        trunk_size: Dimension of the trunk (query) input.
        branch_size: Dimension of the single branch input.
        space_dim: Spatial dimensionality.
        output_size: Number of output channels.
        n_layers: Number of transformer layers.
        n_hidden: Hidden dimension.
        n_head: Number of attention heads.
        n_experts: Number of mixture-of-experts in the FFN.
        mlp_layers: Number of MLP layers in the output head.
        attn_type: Attention type (``"linear"``, ``"softmax"``).
        act: Activation function name.
        ffn_dropout: FFN dropout probability.
        attn_dropout: Attention dropout probability.
        horiz_fourier_dim: Fourier feature dimension for horizontal
            encoding (``0`` = disabled).
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (MoEGPTNO).
    """
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
    """Create a Position-induced Transformer (PiT).

    Transformer for operator learning with position-induced
    cross-attention between input and latent grid resolutions.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        hid_channels: Hidden dimension.
        n_head: Number of attention heads.
        localities: Locality radii per transformer layer.
        input_res: Input spatial resolution ``(H, W)``.
        latent_res: Latent spatial resolution ``(H, W)``.
        output_res: Output spatial resolution ``(H, W)``.
        m_dists: Pre-computed pairwise distance matrices (if provided,
            ``input_res`` / ``latent_res`` / ``output_res`` are ignored).
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (PiT).
    """
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
    """Create a PointNet-style network.

    Point-based architecture for unstructured inputs using shared MLPs
    and global feature aggregation.

    Args:
        in_features: Number of input features per point.
        output_dim: Dimensionality of the output.
        hidden_dims: Hidden layer widths.
        dropout_rate: Dropout probability.
        feature_transform: Optional learnable feature transform
            (``None`` = disabled).
        activation_function: Pointwise activation function.
        use_bias: Add learnable biases.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (PointNet).
    """
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

# =====================================================================
# Foundation models — delegate to per-model modules
# =====================================================================


# -- Poseidon (ScOT) -------------------------------------------------


def poseidonT(
    num_channels=1,
    num_out_channels=1,
    embed_dim=48,
    depths=(4, 4, 4, 4),
    num_heads=(3, 6, 12, 24),
    window_size=8,
    drop_path_rate=0.0,
    patch_size=1,
    in_channels=1,
    out_channels=1,
    padding_mode="zeros",
    cls_token_per_channel=True,
    channel_slice_list_normalized_loss=None,
    use_rope=True,
    rope_theta=100.0,
    rope_mixed=True,
    use_channel_pe=True,
    use_ch_info_tokens=True,
    num_cls_tokens=0,
    num_extra_output_tokens=0,
    processor_type="swin",
    mlp_ratio=4.0,
):
    """Poseidon-T (Tiny) ~20.8 M params.

    Scalable Operator Transformer (ScOT) with Swin-Transformer backbone
    and U-Net-style skip connections.

    Args:
        num_channels: Number of input physical channels.
        num_out_channels: Number of output physical channels.
        embed_dim: Base embedding dimension (doubled per stage).
        depths: Number of Swin-Transformer blocks per stage.
        num_heads: Number of attention heads per stage.
        window_size: Local Swin-Transformer window size.
        drop_path_rate: Stochastic depth rate.
        patch_size: Patch size for input tokenisation.
        in_channels: Alias for ``num_channels``.
        out_channels: Alias for ``num_out_channels``.
        padding_mode: Convolution padding mode.
        cls_token_per_channel: Use separate CLS tokens per channel.
        channel_slice_list_normalized_loss: Channel indices for the
            normalised loss (``None`` → default).
        use_rope: Use Rotary Position Embeddings.
        rope_theta: Base frequency for RoPE.
        rope_mixed: Use mixed-frequency RoPE.
        use_channel_pe: Add per-channel positional encoding.
        use_ch_info_tokens: Use channel-info tokens.
        num_cls_tokens: Number of CLS tokens.
        num_extra_output_tokens: Extra output tokens.
        processor_type: Processor backbone (``"swin"``).
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.

    Returns:
        An ``equinox.Module`` (ScOT).
    """
    from . import poseidon

    return poseidon.T(**{k: v for k, v in locals().items() if k != "poseidon"})


def poseidonB(
    num_channels=1,
    num_out_channels=1,
    embed_dim=96,
    depths=(8, 8, 8, 8),
    num_heads=(3, 6, 12, 24),
    window_size=8,
    drop_path_rate=0.0,
    patch_size=1,
    in_channels=1,
    out_channels=1,
    padding_mode="zeros",
    cls_token_per_channel=True,
    channel_slice_list_normalized_loss=None,
    use_rope=True,
    rope_theta=100.0,
    rope_mixed=True,
    use_channel_pe=True,
    use_ch_info_tokens=True,
    num_cls_tokens=0,
    num_extra_output_tokens=0,
    processor_type="swin",
    mlp_ratio=4.0,
):
    """Poseidon-B (Base) ~157.7 M params.

    Scalable Operator Transformer (ScOT) with Swin-Transformer backbone
    and U-Net-style skip connections.

    Args:
        num_channels: Number of input physical channels.
        num_out_channels: Number of output physical channels.
        embed_dim: Base embedding dimension (doubled per stage).
        depths: Number of Swin-Transformer blocks per stage.
        num_heads: Number of attention heads per stage.
        window_size: Local Swin-Transformer window size.
        drop_path_rate: Stochastic depth rate.
        patch_size: Patch size for input tokenisation.
        in_channels: Alias for ``num_channels``.
        out_channels: Alias for ``num_out_channels``.
        padding_mode: Convolution padding mode.
        cls_token_per_channel: Use separate CLS tokens per channel.
        channel_slice_list_normalized_loss: Channel indices for the
            normalised loss (``None`` → default).
        use_rope: Use Rotary Position Embeddings.
        rope_theta: Base frequency for RoPE.
        rope_mixed: Use mixed-frequency RoPE.
        use_channel_pe: Add per-channel positional encoding.
        use_ch_info_tokens: Use channel-info tokens.
        num_cls_tokens: Number of CLS tokens.
        num_extra_output_tokens: Extra output tokens.
        processor_type: Processor backbone (``"swin"``).
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.

    Returns:
        An ``equinox.Module`` (ScOT).
    """
    from . import poseidon

    return poseidon.B(**{k: v for k, v in locals().items() if k != "poseidon"})


def poseidonL(
    num_channels=1,
    num_out_channels=1,
    embed_dim=192,
    depths=(8, 8, 8, 8),
    num_heads=(3, 6, 12, 24),
    window_size=8,
    drop_path_rate=0.0,
    patch_size=1,
    in_channels=1,
    out_channels=1,
    padding_mode="zeros",
    cls_token_per_channel=True,
    channel_slice_list_normalized_loss=None,
    use_rope=True,
    rope_theta=100.0,
    rope_mixed=True,
    use_channel_pe=True,
    use_ch_info_tokens=True,
    num_cls_tokens=0,
    num_extra_output_tokens=0,
    processor_type="swin",
    mlp_ratio=4.0,
):
    """Poseidon-L (Large) ~628.6 M params.

    Scalable Operator Transformer (ScOT) with Swin-Transformer backbone
    and U-Net-style skip connections.

    Args:
        num_channels: Number of input physical channels.
        num_out_channels: Number of output physical channels.
        embed_dim: Base embedding dimension (doubled per stage).
        depths: Number of Swin-Transformer blocks per stage.
        num_heads: Number of attention heads per stage.
        window_size: Local Swin-Transformer window size.
        drop_path_rate: Stochastic depth rate.
        patch_size: Patch size for input tokenisation.
        in_channels: Alias for ``num_channels``.
        out_channels: Alias for ``num_out_channels``.
        padding_mode: Convolution padding mode.
        cls_token_per_channel: Use separate CLS tokens per channel.
        channel_slice_list_normalized_loss: Channel indices for the
            normalised loss (``None`` → default).
        use_rope: Use Rotary Position Embeddings.
        rope_theta: Base frequency for RoPE.
        rope_mixed: Use mixed-frequency RoPE.
        use_channel_pe: Add per-channel positional encoding.
        use_ch_info_tokens: Use channel-info tokens.
        num_cls_tokens: Number of CLS tokens.
        num_extra_output_tokens: Extra output tokens.
        processor_type: Processor backbone (``"swin"``).
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.

    Returns:
        An ``equinox.Module`` (ScOT).
    """
    from . import poseidon

    return poseidon.L(**{k: v for k, v in locals().items() if k != "poseidon"})


# -- MORPH (ViT3DRegression) -----------------------------------------


def morph_Ti(
    embed_dim=256,
    depth=4,
    heads=4,
    mlp_dim=1024,
    channels=1,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0,
    max_patches=256,
    max_ar=4,
    num_cls_tokens=0,
    out_channels=1,
    image_size=64,
    patch_size=8,
    num_frames=10,
    output_frames=10,
    temporal_patch_size=1,
    positional_embedding="fourier",
):
    """MORPH-Ti (Tiny) ~9.9 M params.

    3-D Vision Transformer for regression over arbitrary-modality PDE
    data with LoRA fine-tuning support.

    Args:
        embed_dim: Model embedding dimension.
        depth: Number of self-attention transformer blocks.
        heads: Number of self-attention heads.
        mlp_dim: MLP hidden dimension.
        channels: Number of input channels.
        dim_head: Dimension per attention head.
        dropout: General dropout rate.
        emb_dropout: Dropout on patch embeddings.
        max_patches: Maximum number of patches.
        max_ar: Maximum autoregressive rollout order.
        num_cls_tokens: Number of CLS tokens.
        out_channels: Number of output channels.
        image_size: Spatial resolution of the input.
        patch_size: Spatial size of 3-D patches.
        num_frames: Number of input time frames.
        output_frames: Number of output time frames.
        temporal_patch_size: Patch size along the time axis.
        positional_embedding: Positional embedding type
            (``"fourier"``, ``"learned"``).

    Returns:
        An ``equinox.Module`` (ViT3DRegression).
    """
    from . import morph

    return morph.Ti(**{k: v for k, v in locals().items() if k != "morph"})


def morph_S(
    embed_dim=512,
    depth=4,
    heads=8,
    mlp_dim=2048,
    channels=1,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0,
    max_patches=256,
    max_ar=4,
    num_cls_tokens=0,
    out_channels=1,
    image_size=64,
    patch_size=8,
    num_frames=10,
    output_frames=10,
    temporal_patch_size=1,
    positional_embedding="fourier",
):
    """MORPH-S (Small) ~32.8 M params.

    3-D Vision Transformer for regression over arbitrary-modality PDE
    data with LoRA fine-tuning support.

    Args:
        embed_dim: Model embedding dimension.
        depth: Number of self-attention transformer blocks.
        heads: Number of self-attention heads.
        mlp_dim: MLP hidden dimension.
        channels: Number of input channels.
        dim_head: Dimension per attention head.
        dropout: General dropout rate.
        emb_dropout: Dropout on patch embeddings.
        max_patches: Maximum number of patches.
        max_ar: Maximum autoregressive rollout order.
        num_cls_tokens: Number of CLS tokens.
        out_channels: Number of output channels.
        image_size: Spatial resolution of the input.
        patch_size: Spatial size of 3-D patches.
        num_frames: Number of input time frames.
        output_frames: Number of output time frames.
        temporal_patch_size: Patch size along the time axis.
        positional_embedding: Positional embedding type
    Returns:
        An ``equinox.Module`` (ViT3DRegression).
    """
    from . import morph

    return morph.S(**{k: v for k, v in locals().items() if k != "morph"})


def morph_M(
    embed_dim=768,
    depth=8,
    heads=12,
    mlp_dim=3072,
    channels=1,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0,
    max_patches=256,
    max_ar=4,
    num_cls_tokens=0,
    out_channels=1,
    image_size=64,
    patch_size=8,
    num_frames=10,
    output_frames=10,
    temporal_patch_size=1,
    positional_embedding="fourier",
):
    """MORPH-M (Medium) ~125.6 M params.

    3-D Vision Transformer for regression over arbitrary-modality PDE
    data with LoRA fine-tuning support.

    Args:
        embed_dim: Model embedding dimension.
        depth: Number of self-attention transformer blocks.
        heads: Number of self-attention heads.
        mlp_dim: MLP hidden dimension.
        channels: Number of input channels.
        dim_head: Dimension per attention head.
        dropout: General dropout rate.
        emb_dropout: Dropout on patch embeddings.
        max_patches: Maximum number of patches.
        max_ar: Maximum autoregressive rollout order.
        num_cls_tokens: Number of CLS tokens.
        out_channels: Number of output channels.
        image_size: Spatial resolution of the input.
        patch_size: Spatial size of 3-D patches.
        num_frames: Number of input time frames.
        output_frames: Number of output time frames.
        temporal_patch_size: Patch size along the time axis.
        positional_embedding: Positional embedding type
            (``"fourier"``, ``"learned"``).

    Returns:
        An ``equinox.Module`` (ViT3DRegression).
    """
    from . import morph

    return morph.M(**{k: v for k, v in locals().items() if k != "morph"})


def morph_L(
    embed_dim=1024,
    depth=16,
    heads=16,
    mlp_dim=4096,
    channels=1,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0,
    max_patches=256,
    max_ar=16,
    num_cls_tokens=0,
    out_channels=1,
    image_size=64,
    patch_size=8,
    num_frames=10,
    output_frames=10,
    temporal_patch_size=1,
    positional_embedding="fourier",
):
    """MORPH-L (Large) ~483.3 M params.

    3-D Vision Transformer for regression over arbitrary-modality PDE
    data with LoRA fine-tuning support.

    Args:
        embed_dim: Model embedding dimension.
        depth: Number of self-attention transformer blocks.
        heads: Number of self-attention heads.
        mlp_dim: MLP hidden dimension.
        channels: Number of input channels.
        dim_head: Dimension per attention head.
        dropout: General dropout rate.
        emb_dropout: Dropout on patch embeddings.
        max_patches: Maximum number of patches.
        max_ar: Maximum autoregressive rollout order.
        num_cls_tokens: Number of CLS tokens.
        out_channels: Number of output channels.
        image_size: Spatial resolution of the input.
        patch_size: Spatial size of 3-D patches.
        num_frames: Number of input time frames.
        output_frames: Number of output time frames.
        temporal_patch_size: Patch size along the time axis.
        positional_embedding: Positional embedding type
            (``"fourier"``, ``"learned"``).

    Returns:
        An ``equinox.Module`` (ViT3DRegression).
    """
    from . import morph

    return morph.L(**{k: v for k, v in locals().items() if k != "morph"})


# -- MPP (AViT) ------------------------------------------------------


def mpp_Ti(
    embed_dim=192,
    num_heads=3,
    processor_blocks=12,
    n_states=12,
    patch_size=(2, 8, 8),
    decoder_depth=2,
    max_num_patches=512,
):
    """MPP AViT-Tiny ~5.5 M params.

    Adaptive Vision Transformer with variable-resolution patching for
    multi-physics operator learning.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        processor_blocks: Number of space-time processor blocks.
        n_states: Number of active physical state variables (channels).
        patch_size: Patch dimensions ``(T, H, W)``.
        decoder_depth: Number of decoder layers.
        max_num_patches: Maximum number of patches.

    Returns:
        An ``equinox.Module`` (AViT).
    """
    from . import mpp

    return mpp.Ti(**{k: v for k, v in locals().items() if k != "mpp"})


def mpp_S(
    embed_dim=384,
    num_heads=6,
    processor_blocks=12,
    n_states=12,
    patch_size=(2, 8, 8),
    decoder_depth=2,
    max_num_patches=512,
):
    """MPP AViT-Small ~21 M params.

    Adaptive Vision Transformer with variable-resolution patching for
    multi-physics operator learning.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        processor_blocks: Number of space-time processor blocks.
        n_states: Number of active physical state variables (channels).
        patch_size: Patch dimensions ``(T, H, W)``.
        decoder_depth: Number of decoder layers.
        max_num_patches: Maximum number of patches.

    Returns:
        An ``equinox.Module`` (AViT).
    """
    from . import mpp

    return mpp.S(**{k: v for k, v in locals().items() if k != "mpp"})


def mpp_B(
    embed_dim=768,
    num_heads=12,
    processor_blocks=12,
    n_states=12,
    patch_size=(2, 8, 8),
    decoder_depth=2,
    max_num_patches=512,
):
    """MPP AViT-Base ~83 M params.

    Adaptive Vision Transformer with variable-resolution patching for
    multi-physics operator learning.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        processor_blocks: Number of space-time processor blocks.
        n_states: Number of active physical state variables (channels).
        patch_size: Patch dimensions ``(T, H, W)``.
        decoder_depth: Number of decoder layers.
        max_num_patches: Maximum number of patches.

    Returns:
        An ``equinox.Module`` (AViT).
    """
    from . import mpp

    return mpp.B(**{k: v for k, v in locals().items() if k != "mpp"})


def mpp_L(
    embed_dim=1024,
    num_heads=16,
    processor_blocks=24,
    n_states=12,
    patch_size=(2, 8, 8),
    decoder_depth=2,
    max_num_patches=512,
):
    """MPP AViT-Large ~300 M params.

    Adaptive Vision Transformer with variable-resolution patching for
    multi-physics operator learning.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        processor_blocks: Number of space-time processor blocks.
        n_states: Number of active physical state variables (channels).
        patch_size: Patch dimensions ``(T, H, W)``.
        decoder_depth: Number of decoder layers.
        max_num_patches: Maximum number of patches.

    Returns:
        An ``equinox.Module`` (AViT).
    """
    from . import mpp

    return mpp.L(**{k: v for k, v in locals().items() if k != "mpp"})


# -- Walrus (IsotropicModel) -----------------------------------------


def walrus(
    hidden_dim=768,
    intermediate_dim=192,
    n_states=4,
    processor_blocks=12,
    groups=16,
    num_heads=12,
    mlp_dim=0,
    max_d=3,
    causal_in_time=False,
    drop_path=0.05,
    input_field_drop=0.1,
    bias_type="rel",
    base_kernel_size=((8, 4), (8, 4), (8, 4)),
    use_spacebag=True,
    use_silu=True,
    include_d=(2, 3),
    encoder_groups=16,
    jitter_patches=True,
    learned_pad=True,
    remat=True,
):
    """Walrus ~1.29 B params.

    Isotropic encoder-processor-decoder with grouped convolutions,
    windowed multi-head attention, and SpaceBag input for unified
    1-D / 2-D / 3-D operator learning (PolymathicAI).

    Args:
        hidden_dim: Hidden feature dimension throughout the processor.
        intermediate_dim: Intermediate dimension in encoder / decoder.
        n_states: Number of active physical state variables (channels).
        processor_blocks: Number of space-time processor blocks.
        groups: Number of groups for grouped convolutions.
        num_heads: Number of attention heads.
        mlp_dim: MLP hidden dimension (``0`` = disable MLP branch).
        max_d: Maximum spatial dimensionality (1–3).
        causal_in_time: Apply causal masking along the time axis.
        drop_path: Stochastic depth rate.
        input_field_drop: Dropout rate on input fields.
        bias_type: Attention bias type (``"rel"`` = relative position).
        base_kernel_size: Base convolution kernel sizes per dimension.
        use_spacebag: Use SpaceBag encoder variant.
        use_silu: Use SiLU activation in encoder / decoder.
        include_d: Which spatial dimensionalities to include.
        encoder_groups: Groups for encoder convolutions.
        jitter_patches: Apply random patch jitter augmentation.
        learned_pad: Learn padding values instead of zeros.
        remat: Enable ``jax.checkpoint`` (rematerialisation) to save
            memory.

    Returns:
        An ``equinox.Module`` (IsotropicModel).
    """
    from . import walrus as _walrus

    return _walrus.base(**{k: v for k, v in locals().items() if k != "_walrus"})


# -- BCAT -------------------------------------------------------------


def bcat(
    n_layers=12,
    dim_emb=1024,
    n_emb_channels=4,
    n_head=8,
    dim_ffn=None,
    max_output_dim=4,
    x_num=128,
    norm_type="rmsnorm",
    activation="swiglu",
    output_activation="tanh",
    dropout=0.0,
    attention_dropout=0.0,
    patch_num=16,
    carry_last_frame=False,
    time_embed="learnable",
    num_registers=0,
    compile_model=True,
):
    """BCAT foundation model (Block Causal Transformer).

    Block Causal Transformer with patched spatio-temporal input,
    RMSNorm, SwiGLU activation, and learnable time embeddings for
    PDE fluid dynamics.

    Args:
        n_layers: Number of transformer layers.
        dim_emb: Token embedding dimension.
        n_emb_channels: Number of embedding channels.
        n_head: Number of attention heads.
        dim_ffn: Feedforward hidden dimension (``None`` → auto).
        max_output_dim: Maximum number of output physical channels.
        x_num: Spatial grid resolution (number of grid points).
        norm_type: Normalisation type (``"rmsnorm"``, ``"layernorm"``).
        activation: Activation function (``"swiglu"``, ``"gelu"``, …).
        output_activation: Final output activation (``"tanh"``,
            ``None``, …).
        dropout: General dropout rate.
        attention_dropout: Attention dropout rate.
        patch_num: Number of patches per spatial dimension.
        carry_last_frame: Carry the last frame as context.
        time_embed: Time embedding type (``"learnable"``,
            ``"continuous"``).
        num_registers: Number of register tokens.
        compile_model: JIT-compile the model.

    Returns:
        An ``equinox.Module`` (BCAT).
    """
    from . import bcat as _bcat

    return _bcat.base(**{k: v for k, v in locals().items() if k != "_bcat"})


# -- PDEformer-2 ------------------------------------------------------


def pdeformer2_small(
    num_node_type=128,
    num_in_degree=32,
    num_out_degree=32,
    num_spatial=16,
    num_encoder_layers=9,
    embed_dim=512,
    ffn_embed_dim=1024,
    num_heads=32,
    pre_layernorm=True,
    scalar_dim_hidden=256,
    scalar_num_layers=3,
    func_enc_type="cnn2dv3",
    func_enc_num_branches=4,
    func_enc_resolution=128,
    func_enc_input_txyz=False,
    func_enc_keep_nchw=True,
    inr_dim_hidden=128,
    inr_num_layers=12,
    enable_affine=False,
    enable_shift=True,
    enable_scale=True,
    activation_fn="sin",
    affine_act_fn="identity",
    hyper_dim_hidden=512,
    hyper_num_layers=2,
    share_hypernet=False,
    multi_inr=False,
    separate_latent=False,
    dtype=None,
):
    """PDEformer-2 Small ~27.7 M params.

    Graphormer encoder + INR decoder with hyper-network bridge for
    two-dimensional PDE solving.

    Args:
        num_node_type: Number of node-type categories in the PDE graph.
        num_in_degree: Number of in-degree bins for degree encoding.
        num_out_degree: Number of out-degree bins for degree encoding.
        num_spatial: Spatial embedding dimension.
        num_encoder_layers: Number of Graphormer encoder layers.
        embed_dim: Graphormer embedding dimension.
        ffn_embed_dim: Feedforward hidden dimension.
        num_heads: Number of attention heads.
        pre_layernorm: Apply LayerNorm before attention (pre-norm).
        scalar_dim_hidden: Scalar encoder MLP hidden dimension.
        scalar_num_layers: Scalar encoder MLP depth.
        func_enc_type: Function encoder architecture.
        func_enc_num_branches: Number of function encoder branches.
        func_enc_resolution: Input resolution for function encoder.
        func_enc_input_txyz: Include time in function-encoder input.
        func_enc_keep_nchw: Keep NCHW layout after function encoding.
        inr_dim_hidden: INR MLP hidden dimension.
        inr_num_layers: INR MLP depth.
        enable_affine: Enable affine transformation in INR.
        enable_shift: Enable additive shift modulation in INR.
        enable_scale: Enable multiplicative scale modulation in INR.
        activation_fn: INR activation (``"sin"``, ``"relu"``, …).
        affine_act_fn: Hypernet output activation.
        hyper_dim_hidden: Hyper-network hidden dimension.
        hyper_num_layers: Hyper-network depth.
        share_hypernet: Share hyper-network across INR layers.
        multi_inr: Use a second INR for multi-output prediction.
        separate_latent: Separate latent for the second INR head.
        dtype: Computation dtype (``None`` = default).

    Returns:
        An ``equinox.Module`` (PDEformer).
    """
    from . import pdeformer2

    return pdeformer2.small(**{k: v for k, v in locals().items() if k != "pdeformer2"})


def pdeformer2_base(
    num_node_type=128,
    num_in_degree=32,
    num_out_degree=32,
    num_spatial=16,
    num_encoder_layers=12,
    embed_dim=768,
    ffn_embed_dim=1536,
    num_heads=32,
    pre_layernorm=True,
    scalar_dim_hidden=256,
    scalar_num_layers=3,
    func_enc_type="cnn2dv3",
    func_enc_num_branches=4,
    func_enc_resolution=128,
    func_enc_input_txyz=False,
    func_enc_keep_nchw=True,
    inr_dim_hidden=768,
    inr_num_layers=12,
    enable_affine=False,
    enable_shift=True,
    enable_scale=True,
    activation_fn="sin",
    affine_act_fn="identity",
    hyper_dim_hidden=512,
    hyper_num_layers=2,
    share_hypernet=False,
    multi_inr=False,
    separate_latent=False,
    dtype=None,
):
    """PDEformer-2 Base.

    Graphormer encoder + INR decoder with hyper-network bridge for
    two-dimensional PDE solving.

    Args:
        num_node_type: Number of node-type categories in the PDE graph.
        num_in_degree: Number of in-degree bins for degree encoding.
        num_out_degree: Number of out-degree bins for degree encoding.
        num_spatial: Spatial embedding dimension.
        num_encoder_layers: Number of Graphormer encoder layers.
        embed_dim: Graphormer embedding dimension.
        ffn_embed_dim: Feedforward hidden dimension.
        num_heads: Number of attention heads.
        pre_layernorm: Apply LayerNorm before attention (pre-norm).
        scalar_dim_hidden: Scalar encoder MLP hidden dimension.
        scalar_num_layers: Scalar encoder MLP depth.
        func_enc_type: Function encoder architecture.
        func_enc_num_branches: Number of function encoder branches.
        func_enc_resolution: Input resolution for function encoder.
        func_enc_input_txyz: Include time in function-encoder input.
        func_enc_keep_nchw: Keep NCHW layout after function encoding.
        inr_dim_hidden: INR MLP hidden dimension.
        inr_num_layers: INR MLP depth.
        enable_affine: Enable affine transformation in INR.
        enable_shift: Enable additive shift modulation in INR.
        enable_scale: Enable multiplicative scale modulation in INR.
        activation_fn: INR activation (``"sin"``, ``"relu"``, …).
        affine_act_fn: Hypernet output activation.
        hyper_dim_hidden: Hyper-network hidden dimension.
        hyper_num_layers: Hyper-network depth.
        share_hypernet: Share hyper-network across INR layers.
        multi_inr: Use a second INR for multi-output prediction.
        separate_latent: Separate latent for the second INR head.
        dtype: Computation dtype (``None`` = default).

    Returns:
        An ``equinox.Module`` (PDEformer).
    """
    from . import pdeformer2

    return pdeformer2.base(**{k: v for k, v in locals().items() if k != "pdeformer2"})


def pdeformer2_fast(
    num_node_type=128,
    num_in_degree=32,
    num_out_degree=32,
    num_spatial=16,
    num_encoder_layers=12,
    embed_dim=768,
    ffn_embed_dim=1536,
    num_heads=32,
    pre_layernorm=True,
    scalar_dim_hidden=256,
    scalar_num_layers=3,
    func_enc_type="cnn2dv3",
    func_enc_num_branches=4,
    func_enc_resolution=128,
    func_enc_input_txyz=False,
    func_enc_keep_nchw=True,
    inr_dim_hidden=256,
    inr_num_layers=12,
    enable_affine=False,
    enable_shift=True,
    enable_scale=True,
    activation_fn="sin",
    affine_act_fn="identity",
    hyper_dim_hidden=512,
    hyper_num_layers=2,
    share_hypernet=False,
    multi_inr=False,
    separate_latent=False,
    dtype=None,
):
    """PDEformer-2 Fast -- smaller INR than Base.

    Graphormer encoder + INR decoder with hyper-network bridge for
    two-dimensional PDE solving.  Uses a narrower INR (256 vs 768)
    for faster inference.

    Args:
        num_node_type: Number of node-type categories in the PDE graph.
        num_in_degree: Number of in-degree bins for degree encoding.
        num_out_degree: Number of out-degree bins for degree encoding.
        num_spatial: Spatial embedding dimension.
        num_encoder_layers: Number of Graphormer encoder layers.
        embed_dim: Graphormer embedding dimension.
        ffn_embed_dim: Feedforward hidden dimension.
        num_heads: Number of attention heads.
        pre_layernorm: Apply LayerNorm before attention (pre-norm).
        scalar_dim_hidden: Scalar encoder MLP hidden dimension.
        scalar_num_layers: Scalar encoder MLP depth.
        func_enc_type: Function encoder architecture.
        func_enc_num_branches: Number of function encoder branches.
        func_enc_resolution: Input resolution for function encoder.
        func_enc_input_txyz: Include time in function-encoder input.
        func_enc_keep_nchw: Keep NCHW layout after function encoding.
        inr_dim_hidden: INR MLP hidden dimension.
        inr_num_layers: INR MLP depth.
        enable_affine: Enable affine transformation in INR.
        enable_shift: Enable additive shift modulation in INR.
        enable_scale: Enable multiplicative scale modulation in INR.
        activation_fn: INR activation (``"sin"``, ``"relu"``, …).
        affine_act_fn: Hypernet output activation.
        hyper_dim_hidden: Hyper-network hidden dimension.
        hyper_num_layers: Hyper-network depth.
        share_hypernet: Share hyper-network across INR layers.
        multi_inr: Use a second INR for multi-output prediction.
        separate_latent: Separate latent for the second INR head.
        dtype: Computation dtype (``None`` = default).

    Returns:
        An ``equinox.Module`` (PDEformer).
    """
    from . import pdeformer2

    return pdeformer2.fast(**{k: v for k, v in locals().items() if k != "pdeformer2"})


# -- DPOT (DPOTNet) ---------------------------------------------------


def dpot_Ti(
    img_size=128,
    patch_size=8,
    embed_dim=512,
    depth=4,
    n_blocks=4,
    out_layer_dim=32,
    mlp_ratio=1.0,
    n_cls=1,
    n_head=None,
    act="gelu",
    dropout=0.0,
    kernel_size=3,
    padding=1,
    padding_mode="zeros",
    use_ln=True,
    unified_pos=False,
    use_codebook=False,
):
    """DPOTNet-Ti (Tiny).

    Patched AFNO-based neural operator with temporal aggregation for
    2-D PDE surrogate modelling.

    Args:
        img_size: Spatial resolution of the input image.
        patch_size: Patch size for input tokenisation.
        embed_dim: Token embedding dimension.
        depth: Number of transformer layers.
        n_blocks: Number of AFNO mixing blocks per layer.
        out_layer_dim: Hidden dimension of the output projection head.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        n_cls: Number of classification / task-embedding tokens.
        n_head: Number of attention heads (``None`` = auto).
        act: Activation function (``"gelu"``, ``"relu"``, …).
        dropout: Dropout probability.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        padding_mode: Padding mode (``"zeros"``, ``"circular"``).
        use_ln: Use layer normalisation.
        unified_pos: Use unified positional encoding.
        use_codebook: Use a learnable codebook.

    Returns:
        An ``equinox.Module`` (DPOTNet).
    """
    from . import dpot

    return dpot.Ti(**{k: v for k, v in locals().items() if k != "dpot"})


def dpot_S(
    img_size=128,
    patch_size=8,
    embed_dim=1024,
    depth=6,
    n_blocks=8,
    out_layer_dim=32,
    mlp_ratio=1.0,
    n_cls=1,
    n_head=None,
    act="gelu",
    dropout=0.0,
    kernel_size=3,
    padding=1,
    padding_mode="zeros",
    use_ln=True,
    unified_pos=False,
    use_codebook=False,
):
    """DPOTNet-S (Small).

    Patched AFNO-based neural operator with temporal aggregation for
    2-D PDE surrogate modelling.

    Args:
        img_size: Spatial resolution of the input image.
        patch_size: Patch size for input tokenisation.
        embed_dim: Token embedding dimension.
        depth: Number of transformer layers.
        n_blocks: Number of AFNO mixing blocks per layer.
        out_layer_dim: Hidden dimension of the output projection head.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        n_cls: Number of classification / task-embedding tokens.
        n_head: Number of attention heads (``None`` = auto).
        act: Activation function (``"gelu"``, ``"relu"``, …).
        dropout: Dropout probability.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        padding_mode: Padding mode (``"zeros"``, ``"circular"``).
        use_ln: Use layer normalisation.
        unified_pos: Use unified positional encoding.
        use_codebook: Use a learnable codebook.

    Returns:
        An ``equinox.Module`` (DPOTNet).
    """
    from . import dpot

    return dpot.S(**{k: v for k, v in locals().items() if k != "dpot"})


def dpot_M(
    img_size=128,
    patch_size=8,
    embed_dim=1024,
    depth=12,
    n_blocks=8,
    out_layer_dim=32,
    mlp_ratio=4.0,
    n_cls=1,
    n_head=None,
    act="gelu",
    dropout=0.0,
    kernel_size=3,
    padding=1,
    padding_mode="zeros",
    use_ln=True,
    unified_pos=False,
    use_codebook=False,
):
    """DPOTNet-M (Medium).

    Patched AFNO-based neural operator with temporal aggregation for
    2-D PDE surrogate modelling.

    Args:
        img_size: Spatial resolution of the input image.
        patch_size: Patch size for input tokenisation.
        embed_dim: Token embedding dimension.
        depth: Number of transformer layers.
        n_blocks: Number of AFNO mixing blocks per layer.
        out_layer_dim: Hidden dimension of the output projection head.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        n_cls: Number of classification / task-embedding tokens.
        n_head: Number of attention heads (``None`` = auto).
        act: Activation function (``"gelu"``, ``"relu"``, …).
        dropout: Dropout probability.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        padding_mode: Padding mode (``"zeros"``, ``"circular"``).
        use_ln: Use layer normalisation.
        unified_pos: Use unified positional encoding.
        use_codebook: Use a learnable codebook.

    Returns:
        An ``equinox.Module`` (DPOTNet).
    """
    from . import dpot

    return dpot.M(**{k: v for k, v in locals().items() if k != "dpot"})


def dpot_L(
    img_size=128,
    patch_size=8,
    embed_dim=1536,
    depth=24,
    n_blocks=16,
    out_layer_dim=32,
    mlp_ratio=1.0,
    n_cls=1,
    n_head=None,
    act="gelu",
    dropout=0.0,
    kernel_size=3,
    padding=1,
    padding_mode="zeros",
    use_ln=True,
    unified_pos=False,
    use_codebook=False,
):
    """DPOTNet-L (Large).

    Patched AFNO-based neural operator with temporal aggregation for
    2-D PDE surrogate modelling.

    Args:
        img_size: Spatial resolution of the input image.
        patch_size: Patch size for input tokenisation.
        embed_dim: Token embedding dimension.
        depth: Number of transformer layers.
        n_blocks: Number of AFNO mixing blocks per layer.
        out_layer_dim: Hidden dimension of the output projection head.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        n_cls: Number of classification / task-embedding tokens.
        n_head: Number of attention heads (``None`` = auto).
        act: Activation function (``"gelu"``, ``"relu"``, …).
        dropout: Dropout probability.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        padding_mode: Padding mode (``"zeros"``, ``"circular"``).
        use_ln: Use layer normalisation.
        unified_pos: Use unified positional encoding.
        use_codebook: Use a learnable codebook.

    Returns:
        An ``equinox.Module`` (DPOTNet).
    """
    from . import dpot

    return dpot.L(**{k: v for k, v in locals().items() if k != "dpot"})


def dpot_H(
    img_size=128,
    patch_size=8,
    embed_dim=2048,
    depth=27,
    n_blocks=8,
    out_layer_dim=32,
    mlp_ratio=1.0,
    n_cls=1,
    n_head=None,
    act="gelu",
    dropout=0.0,
    kernel_size=3,
    padding=1,
    padding_mode="zeros",
    use_ln=True,
    unified_pos=False,
    use_codebook=False,
):
    """DPOTNet-H (Huge).

    Patched AFNO-based neural operator with temporal aggregation for
    2-D PDE surrogate modelling.

    Args:
        img_size: Spatial resolution of the input image.
        patch_size: Patch size for input tokenisation.
        embed_dim: Token embedding dimension.
        depth: Number of transformer layers.
        n_blocks: Number of AFNO mixing blocks per layer.
        out_layer_dim: Hidden dimension of the output projection head.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        n_cls: Number of classification / task-embedding tokens.
        n_head: Number of attention heads (``None`` = auto).
        act: Activation function (``"gelu"``, ``"relu"``, …).
        dropout: Dropout probability.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        padding_mode: Padding mode (``"zeros"``, ``"circular"``).
        use_ln: Use layer normalisation.
        unified_pos: Use unified positional encoding.
        use_codebook: Use a learnable codebook.

    Returns:
        An ``equinox.Module`` (DPOTNet).
    """
    from . import dpot

    return dpot.H(**{k: v for k, v in locals().items() if k != "dpot"})


# -- PROSE -------------------------------------------------------------


def prose_fd_1to1(
    x_num=128,
    max_output_dim=4,
    output_len=10,
    *,
    key=None,
):
    """PROSE finite-difference 1-to-1.

    Transformer that maps a single finite-difference input trajectory
    to an output trajectory.

    Args:
        x_num: Spatial grid resolution (number of grid points).
        max_output_dim: Maximum number of output physical channels.
        output_len: Length of the output sequence.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (PROSE1to1).
    """
    from . import prose

    return prose.fd_1to1(**{k: v for k, v in locals().items() if k != "prose"})


def prose_fd_2to1(
    n_words,
    x_num=128,
    max_output_dim=4,
    *,
    key=None,
):
    """PROSE finite-difference 2-to-1.

    Transformer that fuses a symbolic equation description and a
    finite-difference data trajectory into a single output trajectory.

    Args:
        n_words: Size of the symbol vocabulary.
        x_num: Spatial grid resolution (number of grid points).
        max_output_dim: Maximum number of output physical channels.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (PROSE2to1).
    """
    from . import prose

    return prose.fd_2to1(**{k: v for k, v in locals().items() if k != "prose"})


def prose_ode_2to1(
    n_words,
    pad_index,
    max_output_dimension=3,
    *,
    key=None,
):
    """PROSE ODE 2-to-1.

    Transformer that fuses a symbolic ODE description and observed
    data into a predicted trajectory.

    Args:
        n_words: Size of the symbol vocabulary.
        pad_index: Index used for padding tokens in the vocabulary.
        max_output_dimension: Maximum number of output state dimensions.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (PROSEODE2to1).
    """
    from . import prose

    return prose.ode_2to1(**{k: v for k, v in locals().items() if k != "prose"})


def prose_pde_2to1(
    n_words,
    pad_index,
    max_output_dimension=1,
    x_patch_size=1,
    x_grid_size=128,
    *,
    key=None,
):
    """PROSE PDE 2-to-1.

    Transformer that fuses a symbolic PDE description and observed
    data into a predicted spatio-temporal field.

    Args:
        n_words: Size of the symbol vocabulary.
        pad_index: Index used for padding tokens in the vocabulary.
        max_output_dimension: Maximum number of output state dimensions.
        x_patch_size: Spatial patch size for the data tokeniser.
        x_grid_size: Spatial grid resolution.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (PROSEPDE2to1).
    """
    from . import prose

    return prose.pde_2to1(**{k: v for k, v in locals().items() if k != "prose"})
