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
        batch_norm: Not supported (raises): use ``layer_norm``.
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
        norm: Normalisation after each spectral layer: ``None``, ``"layer"``, ``"rms"``, ``"instance"`` or ``"group"`` (see ``make_norm``).
        training: Whether the model is in training mode (affects
            dropout).
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
        norm: Normalisation after each spectral layer: ``None``, ``"layer"``, ``"rms"``, ``"instance"`` or ``"group"`` (see ``make_norm``).
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
        norm: Normalisation after each spectral layer: ``None``, ``"layer"``, ``"rms"``, ``"instance"`` or ``"group"`` (see ``make_norm``).
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
    norm: str | None = "instance",
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
        norm: Normalisation after each convolution: ``None``, ``"layer"``,
            ``"rms"``, ``"instance"`` or ``"group"`` (see ``make_norm``).
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
        norm: Normalisation after each convolution: ``None``, ``"layer"``,
            ``"rms"``, ``"instance"`` or ``"group"`` (see ``make_norm``).
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
    norm: str | None = "instance",
    up_mode: str = "upconv",
    activation: str | Callable = "celu",
    padding_mode: str = "circular",
    groups: int = 1,
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
        norm: Normalisation after each convolution: ``None``, ``"layer"``,
            ``"rms"``, ``"instance"`` or ``"group"`` (see ``make_norm``).
        up_mode: Upsampling method (``"upconv"`` = transposed
            convolution, ``"upsample"`` = nearest + conv).
        activation: Activation function or name string.
        padding_mode: Padding mode (``"circular"``, ``"zeros"``,
            ``"reflect"``).
        groups: Number of groups for grouped convolutions.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (UNet3D).
    """
    from .architectures.common import get_activation
    from .architectures.unet import UNet3D

    return UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        wf=wf,
        norm=norm,
        up_mode=up_mode,
        groups=groups,
        activation=get_activation(activation),
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
    n_tokens: int = 8,
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
        n_tokens: Number of tokens the branch output is split into for the
            attention combiner; each query point attends over them.
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
        combination_n_tokens=n_tokens,
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
    modes1: int = 12,
    modes2: int = 12,
    width: int = 32,
    in_channels: int = 2,
    out_channels: int = 1,
    s1: int = 40,
    s2: int = 40,
    n_layers: int = 3,
    fc_dim: int = 128,
    act: str = "gelu",
    deform: bool = True,
    iphi_width: int = 32,
    iphi_depth: int = 3,
    iphi_act: str = "tanh",
    iphi_center: Sequence[float] = (1e-4, 1e-4),
    code_dim: int = 0,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a Geo-FNO (Li et al. 2022, arXiv:2207.05209).

    A learned deformation ``iphi`` maps 2-D physical points into a latent
    unit square; a non-uniform Fourier transform moves the point values
    onto a uniform ``s1 x s2`` latent grid, FFT-based Fourier layers act
    there, and the last layer evaluates the Fourier series back at the
    query points. Call as ``model(coords, x=None, out_coords=None,
    code=None)`` with ``coords`` of shape ``(N, 2)``.

    Defaults reproduce the paper's elasticity model. The deformation
    network is part of the module; to train it with its own learning rate
    (as upstream does), split the optimizer with ``optax.multi_transform``
    on the ``iphi`` subtree.

    Args:
        modes1, modes2: Fourier modes kept per latent axis.
        width: Channel width of the Fourier layers.
        in_channels: Point feature channels (2 when the features are the
            coordinates themselves).
        out_channels: Output channels.
        s1, s2: Latent grid size.
        n_layers: Number of latent-grid Fourier layers between the
            point-to-grid and grid-to-point layers.
        fc_dim: Hidden width of the output projection.
        act: Activation (``"gelu"`` is the exact erf GELU, as upstream).
        deform: Learn the deformation ``iphi``; ``False`` uses the
            coordinates as latent coordinates (they should lie in
            ``[0, 1]^2``).
        iphi_width: Width of the deformation network (divisible by 4).
        iphi_depth: Hidden layers of the deformation network.
        iphi_act: Activation of the deformation network.
        iphi_center: Centre for the angle/radius features of ``iphi``.
        code_dim: Length of an optional design-parameter vector fed to
            ``iphi`` (0 = none).
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (GeoFNO).
    """
    from .architectures.geofno import GeoFNO

    return GeoFNO(
        modes1=modes1,
        modes2=modes2,
        width=width,
        in_channels=in_channels,
        out_channels=out_channels,
        s1=s1,
        s2=s2,
        n_layers=n_layers,
        fc_dim=fc_dim,
        act=act,
        deform=deform,
        iphi_width=iphi_width,
        iphi_depth=iphi_depth,
        iphi_act=iphi_act,
        iphi_center=tuple(iphi_center),
        code_dim=code_dim,
        key=_resolve_key(key),
    )


def geofno_nodal(
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
    """Create a node-basis geometry-aware Fourier Neural Operator.

    Projects node features onto cos/sin Fourier bases evaluated directly
    at the mesh nodes (weighted by integration weights), so it runs on
    irregular point sets without a latent grid. This is the variant from
    the PKU-CMEGroup NeuralOperator code; for the Geo-FNO of Li et al.
    2022 with a learned deformation, use :func:`geofno`.

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
        An ``equinox.Module`` (GeoFNONodal).
    """
    from .architectures.geofno_nodal import GeoFNONodal
    from .architectures.common import compute_Fourier_modes as _compute_modes

    modes = _compute_modes(ndims, list(nks), list(Ls))
    modes = jnp.array(modes)
    return GeoFNONodal(
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


def gaot(
    input_size: int = 2,
    output_size: int = 1,
    coord_dim: int = 2,
    radius: float = 0.033,
    hidden_size_mlp: int = 64,
    mlp_layers: int = 3,
    lifting_channels: int = 32,
    use_attention: bool = True,
    attention_type: str = "cosine",
    use_geoembed: bool = True,
    embedding_method: str = "statistical",
    pooling: str = "max",
    transform_type: str = "linear",
    transformer_hidden_size: int = 256,
    num_heads: int = 8,
    num_kv_heads: Optional[int] = None,
    num_layers: int = 3,
    patch_size: int = 2,
    ffn_multiplier: int = 4,
    positional_embedding: str = "absolute",
    use_long_range_skip: bool = True,
    latent_tokens_size: Tuple[int, int] = (32, 32),
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a Geometry-Aware Operator Transformer (GAOT).

    Faithful JAX/Equinox port of ``camlab-ethz/GAOT``. Precompute neighbour
    graphs with :func:`foundax.architectures.gaot.compute_neighbors_csr`.

    Args:
        input_size: Number of input physical channels.
        output_size: Number of output channels.
        coord_dim: Spatial coordinate dimension (2 or 3).
        radius: MAGNO neighbour search radius.
        hidden_size_mlp: Hidden width of kernel + lifting MLPs.
        mlp_layers: Kernel MLP depth.
        lifting_channels: Latent token feature dimension.
        use_attention: Enable AGNO attention reweighting.
        attention_type: ``"cosine"`` or ``"dot_product"``.
        use_geoembed: Enable per-query geometric embedding.
        embedding_method: ``"statistical"`` or ``"pointnet"``.
        pooling: PointNet pooling (``"max"`` or ``"mean"``).
        transform_type: AGNO transform (``"linear"`` / ``"nonlinear"``).
        transformer_hidden_size: UViT hidden width.
        num_heads: UViT attention heads.
        num_kv_heads: UViT key/value heads (None → ``num_heads``).
        num_layers: UViT depth.
        patch_size: Latent-grid patchification stride.
        ffn_multiplier: SwiGLU FFN width multiplier.
        positional_embedding: ``"absolute"`` or ``"rope"``.
        use_long_range_skip: Enable UViT encoder↔decoder skips.
        latent_tokens_size: ``(H, W)`` of the latent grid.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (GAOT).
    """
    from .architectures.gaot import (
        gaot as _gaot,
        MAGNOConfig,
        TransformerConfig,
        AttentionConfig,
    )

    if num_kv_heads is None:
        num_kv_heads = num_heads

    mc = MAGNOConfig(
        coord_dim=coord_dim,
        radius=radius,
        hidden_size=hidden_size_mlp,
        mlp_layers=mlp_layers,
        lifting_channels=lifting_channels,
        use_attention=use_attention,
        attention_type=attention_type,
        use_geoembed=use_geoembed,
        embedding_method=embedding_method,
        pooling=pooling,
        transform_type=transform_type,
    )
    tc = TransformerConfig(
        patch_size=patch_size,
        hidden_size=transformer_hidden_size,
        num_layers=num_layers,
        ffn_multiplier=ffn_multiplier,
        positional_embedding=positional_embedding,
        use_long_range_skip=use_long_range_skip,
        attn_config=AttentionConfig(num_heads=num_heads, num_kv_heads=num_kv_heads),
    )
    return _gaot(
        input_size=input_size,
        output_size=output_size,
        magno_config=mc,
        transformer_config=tc,
        latent_tokens_size=latent_tokens_size,
        key=_resolve_key(key),
    )


def gaot_S(
    input_size: int = 2,
    output_size: int = 1,
    *,
    key: jax.Array | None = None,
    **kwargs,
) -> eqx.Module:
    """GAOT-S (Small) ~12 M params.

    Args:
        input_size: Number of input physical channels.
        output_size: Number of output channels.
        key: JAX PRNG key.
        **kwargs: Override any :func:`gaot` hyperparameter.

    Returns:
        An ``equinox.Module`` (GAOT).
    """
    from .architectures.gaot import S

    return S(input_size, output_size, key=_resolve_key(key), **kwargs)


def gaot_M(
    input_size: int = 2,
    output_size: int = 1,
    *,
    key: jax.Array | None = None,
    **kwargs,
) -> eqx.Module:
    """GAOT-M (Medium) ~55 M params.

    Args:
        input_size: Number of input physical channels.
        output_size: Number of output channels.
        key: JAX PRNG key.
        **kwargs: Override any :func:`gaot` hyperparameter.

    Returns:
        An ``equinox.Module`` (GAOT).
    """
    from .architectures.gaot import M

    return M(input_size, output_size, key=_resolve_key(key), **kwargs)


def gaot_L(
    input_size: int = 2,
    output_size: int = 1,
    *,
    key: jax.Array | None = None,
    **kwargs,
) -> eqx.Module:
    """GAOT-L (Large) ~200 M params.

    Args:
        input_size: Number of input physical channels.
        output_size: Number of output channels.
        key: JAX PRNG key.
        **kwargs: Override any :func:`gaot` hyperparameter.

    Returns:
        An ``equinox.Module`` (GAOT).
    """
    from .architectures.gaot import L

    return L(input_size, output_size, key=_resolve_key(key), **kwargs)


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
# Kolmogorov-Arnold Networks
# =====================================================================


def kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    grid_size: int = 5,
    spline_order: int = 3,
    grid_range: tuple = (-1.0, 1.0),
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a B-spline Kolmogorov-Arnold Network (Liu et al. 2024 original)."""
    from .architectures.kan import KAN

    return KAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=grid_range,
        key=_resolve_key(key),
    )


def efficient_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    grid_size: int = 5,
    spline_order: int = 3,
    grid_range: tuple = (-1.0, 1.0),
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create an EfficientKAN (Blealtan/efficient-kan B-spline reformulation)."""
    from .architectures.kan import EfficientKAN

    return EfficientKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=grid_range,
        key=_resolve_key(key),
    )


def fastkan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    grid_size: int = 8,
    grid_range: tuple = (-2.0, 2.0),
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a FastKAN (Gaussian-RBF KAN, ZiyaoLi/fast-kan)."""
    from .architectures.kan import FastKAN

    return FastKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        grid_size=grid_size,
        grid_range=grid_range,
        key=_resolve_key(key),
    )


def fourier_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    num_frequencies: int = 8,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a FourierKAN (sin/cos series edges)."""
    from .architectures.kan import FourierKAN

    return FourierKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        num_frequencies=num_frequencies,
        key=_resolve_key(key),
    )


def chebyshev_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    degree: int = 5,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a ChebyshevKAN (Chebyshev T_n polynomial edges)."""
    from .architectures.kan import ChebyshevKAN

    return ChebyshevKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        degree=degree,
        key=_resolve_key(key),
    )


def jacobi_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    degree: int = 5,
    alpha: float = 1.0,
    beta: float = 1.0,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a JacobiKAN (Jacobi P_n^(alpha,beta) polynomial edges)."""
    from .architectures.kan import JacobiKAN

    return JacobiKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        degree=degree,
        alpha=alpha,
        beta=beta,
        key=_resolve_key(key),
    )


def legendre_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    degree: int = 5,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a LegendreKAN (Legendre P_n polynomial edges)."""
    from .architectures.kan import LegendreKAN

    return LegendreKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        degree=degree,
        key=_resolve_key(key),
    )


def wavelet_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    num_scales: int = 6,
    wavelet_type: str = "mexican_hat",
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a WaveletKAN.

    ``wavelet_type`` ∈ ``mexican_hat``, ``morlet``, ``shannon``, ``dog``.
    """
    from .architectures.kan import WaveletKAN

    return WaveletKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        num_scales=num_scales,
        wavelet_type=wavelet_type,
        key=_resolve_key(key),
    )


def taylor_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    degree: int = 4,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a TaylorKAN (truncated power-series edges)."""
    from .architectures.kan import TaylorKAN

    return TaylorKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        degree=degree,
        key=_resolve_key(key),
    )


def kan_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    basis: str = "bspline",
    *,
    key: jax.Array | None = None,
    **basis_kwargs,
) -> eqx.Module:
    """Create a single KAN-convolutional 2D layer (channel-last ``(H, W, C)``).

    ``basis`` ∈ any of the registered KAN bases (``bspline``, ``rbf``,
    ``fourier``, ``chebyshev``, ``jacobi``, ``legendre``, ``wavelet``,
    ``taylor``, ``hermite``, ``laguerre``, ``bernstein``, ``relu``,
    ``rational``, ``sinc``, ``gram``, ``bsrbf``). Extra kwargs are forwarded
    to the basis (e.g. ``grid_size``, ``degree``, ``num_frequencies``).
    """
    from .architectures.kan import KANConv2d

    return KANConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        basis=basis,
        key=_resolve_key(key),
        **basis_kwargs,
    )


def hermite_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    degree: int = 5,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a HermiteKAN (probabilist Hermite polynomial edges)."""
    from .architectures.kan import HermiteKAN

    return HermiteKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        degree=degree,
        key=_resolve_key(key),
    )


def laguerre_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    degree: int = 5,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a LaguerreKAN (Laguerre polynomial edges)."""
    from .architectures.kan import LaguerreKAN

    return LaguerreKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        degree=degree,
        key=_resolve_key(key),
    )


def bernstein_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    degree: int = 5,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a BernsteinKAN (Bernstein polynomial edges on [0, 1])."""
    from .architectures.kan import BernsteinKAN

    return BernsteinKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        degree=degree,
        key=_resolve_key(key),
    )


def relu_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    grid_size: int = 8,
    order: int = 2,
    grid_range: tuple = (-1.0, 1.0),
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a ReLU-KAN / FasterKAN ((relu*relu)^order basis)."""
    from .architectures.kan import ReLUKAN

    return ReLUKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        grid_size=grid_size,
        order=order,
        grid_range=grid_range,
        key=_resolve_key(key),
    )


def rational_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    degree: int = 5,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a RationalKAN (rational-Chebyshev / Padé-style edges)."""
    from .architectures.kan import RationalKAN

    return RationalKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        degree=degree,
        key=_resolve_key(key),
    )


def sinc_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    grid_size: int = 8,
    grid_range: tuple = (-2.0, 2.0),
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a SincKAN (sinc basis on a uniform grid)."""
    from .architectures.kan import SincKAN

    return SincKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        grid_size=grid_size,
        grid_range=grid_range,
        key=_resolve_key(key),
    )


def gram_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    degree: int = 5,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a GramKAN (orthonormal Legendre / continuous-Gram basis)."""
    from .architectures.kan import GramKAN

    return GramKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        degree=degree,
        key=_resolve_key(key),
    )


def bsrbf_kan(
    in_features: int,
    output_dim: int = 1,
    hidden_dims: int | Sequence[int] = 64,
    num_layers: int = 2,
    grid_size: int = 5,
    spline_order: int = 3,
    rbf_grid_size: int = 8,
    grid_range: tuple = (-1.0, 1.0),
    rbf_grid_range: tuple = (-2.0, 2.0),
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a BSRBFKAN (concatenated B-spline + Gaussian-RBF basis)."""
    from .architectures.kan import BSRBFKAN

    return BSRBFKAN(
        in_features=in_features,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        grid_size=grid_size,
        spline_order=spline_order,
        rbf_grid_size=rbf_grid_size,
        grid_range=grid_range,
        rbf_grid_range=rbf_grid_range,
        key=_resolve_key(key),
    )


def kan_conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    basis: str = "bspline",
    *,
    key: jax.Array | None = None,
    **basis_kwargs,
) -> eqx.Module:
    """Create a KAN-convolutional 1D layer (channel-last ``(W, C)``)."""
    from .architectures.kan import KANConv1d

    return KANConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        basis=basis,
        key=_resolve_key(key),
        **basis_kwargs,
    )


def kan_conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    basis: str = "bspline",
    *,
    key: jax.Array | None = None,
    **basis_kwargs,
) -> eqx.Module:
    """Create a KAN-convolutional 3D layer (channel-last ``(D, H, W, C)``)."""
    from .architectures.kan import KANConv3d

    return KANConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        basis=basis,
        key=_resolve_key(key),
        **basis_kwargs,
    )


def kan_res_block(
    features: int,
    basis: str = "bspline",
    activation: Callable = jax.nn.silu,
    use_layer_norm: bool = False,
    *,
    key: jax.Array | None = None,
    **basis_kwargs,
) -> eqx.Module:
    """Create a residual KAN block: ``out = x + kan2(act(kan1(x)))``."""
    from .architectures.kan import KANResBlock

    return KANResBlock(
        features=features,
        basis=basis,
        activation=activation,
        use_layer_norm=use_layer_norm,
        key=_resolve_key(key),
        **basis_kwargs,
    )


def kan_spectral_block1d(
    in_channels: int,
    out_channels: int,
    n_modes: int,
    basis: str = "rbf",
    activation: Callable = jax.nn.gelu,
    *,
    key: jax.Array | None = None,
    **basis_kwargs,
) -> eqx.Module:
    """FNO-style 1-D block with KAN channel mixing."""
    from .architectures.kan import KANSpectralBlock1d

    return KANSpectralBlock1d(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        basis=basis,
        activation=activation,
        key=_resolve_key(key),
        **basis_kwargs,
    )


def kan_spectral_block2d(
    in_channels: int,
    out_channels: int,
    n_modes: int,
    basis: str = "rbf",
    activation: Callable = jax.nn.gelu,
    *,
    key: jax.Array | None = None,
    **basis_kwargs,
) -> eqx.Module:
    """FNO-style 2-D block with KAN channel mixing."""
    from .architectures.kan import KANSpectralBlock2d

    return KANSpectralBlock2d(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        basis=basis,
        activation=activation,
        key=_resolve_key(key),
        **basis_kwargs,
    )


def kan_spectral_block3d(
    in_channels: int,
    out_channels: int,
    n_modes: int,
    basis: str = "rbf",
    activation: Callable = jax.nn.gelu,
    *,
    key: jax.Array | None = None,
    **basis_kwargs,
) -> eqx.Module:
    """FNO-style 3-D block with KAN channel mixing."""
    from .architectures.kan import KANSpectralBlock3d

    return KANSpectralBlock3d(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        basis=basis,
        activation=activation,
        key=_resolve_key(key),
        **basis_kwargs,
    )


def kan_attention_block(
    features: int,
    num_heads: int = 4,
    basis: str = "rbf",
    *,
    key: jax.Array | None = None,
    **basis_kwargs,
) -> eqx.Module:
    """Transformer-style block with KAN feed-forward sublayer."""
    from .architectures.kan import KANAttentionBlock

    return KANAttentionBlock(
        features=features,
        num_heads=num_heads,
        basis=basis,
        key=_resolve_key(key),
        **basis_kwargs,
    )


# =====================================================================
# Foundation models: the per-model factories themselves
# =====================================================================
# The flat names are the same functions as the namespace factories (poseidonT is poseidon.T),
# so their signatures cannot drift apart. foundax/__init__.py imports these modules first anyway.

from . import bcat as _bcat  # noqa: E402
from . import dpot as _dpot  # noqa: E402
from . import morph as _morph  # noqa: E402
from . import mpp as _mpp  # noqa: E402
from . import pdeformer2 as _pdeformer2  # noqa: E402
from . import poseidon as _poseidon  # noqa: E402
from . import prose as _prose  # noqa: E402
from . import walrus as _walrus  # noqa: E402

poseidonT, poseidonB, poseidonL = _poseidon.T, _poseidon.B, _poseidon.L
morph_Ti, morph_S, morph_M, morph_L = _morph.Ti, _morph.S, _morph.M, _morph.L
mpp_Ti, mpp_S, mpp_B, mpp_L = _mpp.Ti, _mpp.S, _mpp.B, _mpp.L
walrus = _walrus.base
bcat = _bcat.base
pdeformer2_small, pdeformer2_base, pdeformer2_fast = (
    _pdeformer2.small,
    _pdeformer2.base,
    _pdeformer2.fast,
)
dpot_Ti, dpot_S, dpot_M, dpot_L, dpot_H = _dpot.Ti, _dpot.S, _dpot.M, _dpot.L, _dpot.H
prose_fd_1to1, prose_fd_2to1 = _prose.fd_1to1, _prose.fd_2to1
prose_ode_2to1, prose_pde_2to1 = _prose.ode_2to1, _prose.pde_2to1


# =====================================================================
# Flow-matching / diffusion backbones
# =====================================================================


def dit2d(
    in_channels: int,
    patch_size: int = 4,
    hidden_size: int = 256,
    depth: int = 6,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 2-D Diffusion Transformer (DiT).

    Processes unbatched 2-D PDE fields ``(H, W, C)`` conditioned on a
    scalar timestep *t*.  H and W must be divisible by *patch_size*.

    Args:
        in_channels: Number of input (and output) channels.
        patch_size: Spatial patch size (H and W must be divisible by this).
        hidden_size: Transformer hidden dimension (must be divisible by 4).
        depth: Number of DiT transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension multiplier inside each block.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (DiT2d).
    """
    from .architectures.dit import DiT2d

    return DiT2d(
        in_channels,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio,
        key=_resolve_key(key),
    )


def dit3d(
    in_channels: int,
    patch_size: int = 4,
    hidden_size: int = 252,
    depth: int = 4,
    num_heads: int = 6,
    mlp_ratio: float = 4.0,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 3-D Diffusion Transformer (DiT).

    Processes unbatched volumetric fields ``(D, H, W, C)`` conditioned on
    a scalar timestep *t*.  All spatial dims must be divisible by *patch_size*.

    Args:
        in_channels: Number of input (and output) channels.
        patch_size: Cubic patch size (D, H, W must be divisible by this).
        hidden_size: Transformer hidden dimension (must be divisible by 6).
        depth: Number of DiT transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension multiplier.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (DiT3d).
    """
    from .architectures.dit import DiT3d

    return DiT3d(
        in_channels,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio,
        key=_resolve_key(key),
    )


def ffno2d(
    in_channels: int,
    hidden_channels: int = 32,
    out_channels: int | None = None,
    n_modes: int = 16,
    n_layers: int = 4,
    use_film: bool = False,
    emb_dim: int = 64,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 2-D Factorized FNO (F-FNO).

    Factorized spectral conv uses two 1-D spectral ops instead of one 2-D
    op, reducing parameter count ~16× while preserving expressivity.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Width of the latent representation.
        out_channels: Output channels (defaults to ``in_channels``).
        n_modes: Number of Fourier modes retained per axis.
        n_layers: Number of FactorizedSpectralBlock2d layers.
        use_film: If True, blocks accept ``t_emb`` for FiLM time conditioning.
        emb_dim: Embedding dimension expected by FiLM (only used if use_film).
        key: JAX PRNG key.

    Returns:
        An ``equinox.Module`` (FFNO2d).
    """
    from .architectures.ffno import FFNO2d

    return FFNO2d(
        in_channels,
        hidden_channels,
        out_channels or in_channels,
        n_modes,
        n_layers,
        use_film,
        emb_dim,
        key=_resolve_key(key),
    )


def ffno3d(
    in_channels: int,
    hidden_channels: int = 16,
    out_channels: int | None = None,
    n_modes: int = 8,
    n_layers: int = 4,
    use_film: bool = False,
    emb_dim: int = 64,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 3-D Factorized FNO (F-FNO).

    Args:
        in_channels: Number of input channels.
        hidden_channels: Width of the latent representation.
        out_channels: Output channels (defaults to ``in_channels``).
        n_modes: Number of Fourier modes retained per axis.
        n_layers: Number of FactorizedSpectralBlock3d layers.
        use_film: If True, blocks accept ``t_emb`` for FiLM time conditioning.
        emb_dim: Embedding dimension expected by FiLM (only used if use_film).
        key: JAX PRNG key.

    Returns:
        An ``equinox.Module`` (FFNO3d).
    """
    from .architectures.ffno import FFNO3d

    return FFNO3d(
        in_channels,
        hidden_channels,
        out_channels or in_channels,
        n_modes,
        n_layers,
        use_film,
        emb_dim,
        key=_resolve_key(key),
    )


def wno1d(
    in_channels: int,
    hidden_channels: int = 32,
    out_channels: int | None = None,
    n_scales: int = 2,
    depth: int = 4,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 1-D Wavelet Neural Operator (WNO).

    Uses Daubechies-8 wavelet decomposition in a multi-scale architecture.
    Spatial dimension W must be ≥ 16 and divisible by 2^n_scales.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Width of the latent representation.
        out_channels: Output channels (defaults to ``in_channels``).
        n_scales: Number of wavelet decomposition levels.
        depth: Number of WaveletBlock1d layers.
        key: JAX PRNG key.

    Returns:
        An ``equinox.Module`` (WNO1d).
    """
    from .architectures.wno import WNO1d

    return WNO1d(
        in_channels,
        hidden_channels,
        out_channels or in_channels,
        n_scales,
        depth,
        key=_resolve_key(key),
    )


def wno2d(
    in_channels: int,
    hidden_channels: int = 32,
    out_channels: int | None = None,
    n_scales: int = 2,
    depth: int = 4,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 2-D Wavelet Neural Operator (WNO).

    H and W must each be ≥ 16 and divisible by 2^n_scales.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Width of the latent representation.
        out_channels: Output channels (defaults to ``in_channels``).
        n_scales: Number of wavelet decomposition levels.
        depth: Number of WaveletBlock2d layers.
        key: JAX PRNG key.

    Returns:
        An ``equinox.Module`` (WNO2d).
    """
    from .architectures.wno import WNO2d

    return WNO2d(
        in_channels,
        hidden_channels,
        out_channels or in_channels,
        n_scales,
        depth,
        key=_resolve_key(key),
    )


def wno3d(
    in_channels: int,
    hidden_channels: int = 16,
    out_channels: int | None = None,
    n_scales: int = 2,
    depth: int = 4,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 3-D Wavelet Neural Operator (WNO).

    D, H, W must each be ≥ 16 and divisible by 2^n_scales.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Width of the latent representation.
        out_channels: Output channels (defaults to ``in_channels``).
        n_scales: Number of wavelet decomposition levels.
        depth: Number of WaveletBlock3d layers.
        key: JAX PRNG key.

    Returns:
        An ``equinox.Module`` (WNO3d).
    """
    from .architectures.wno import WNO3d

    return WNO3d(
        in_channels,
        hidden_channels,
        out_channels or in_channels,
        n_scales,
        depth,
        key=_resolve_key(key),
    )


def sfno2d(
    in_channels: int,
    out_channels: int | None = None,
    hidden_channels: int = 32,
    L: int = 32,
    nlat: int = 64,
    nlon: int = 128,
    n_layers: int = 4,
    grid: str = "legendre-gauss",
    activation: str = "gelu",
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a 2-D Spherical Fourier Neural Operator (SFNO).

    Replaces FNO's FFT-based spectral conv with a real-valued spherical
    harmonic transform, giving rotational equivariance on the sphere and
    long-horizon stability for global weather / climate fields.

    Reference:
        Bonev et al. *Spherical Fourier Neural Operators: Learning
        Stable Dynamics on the Sphere* (ICML 2023).
        https://arxiv.org/abs/2306.03838

    Input shape ``(nlat, nlon, in_channels)``, channel-last unbatched.
    Batch with ``jax.vmap`` externally.

    Args:
        in_channels: Number of input channels.
        out_channels: Output channels (defaults to ``in_channels``).
        hidden_channels: Width of the latent representation.
        L: Spherical-harmonic bandlimit (number of l-modes kept).
            Must satisfy ``L ≤ nlat`` and ``L ≤ nlon//2 + 1``.
        nlat: Number of latitude points on the grid.
        nlon: Number of longitude points on the grid.
        n_layers: Number of SphericalBlock2d layers.
        grid: Latitude grid: ``"legendre-gauss"`` (default, exact
            quadrature) or ``"equiangular"`` (ERA5-style, approximate).
        activation: Activation function name (e.g. ``"gelu"``,
            ``"silu"``).
        key: JAX PRNG key.

    Returns:
        An ``equinox.Module`` (SFNO2d).
    """
    from .architectures.sfno import SFNO2d

    return SFNO2d(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels if out_channels is not None else in_channels,
        L=L,
        nlat=nlat,
        nlon=nlon,
        n_layers=n_layers,
        grid=grid,
        activation=activation,
        key=_resolve_key(key),
    )


def transolver(
    space_dim: int,
    fun_dim: int,
    out_features: int = 1,
    hidden_dim: int = 128,
    n_layers: int = 8,
    n_heads: int = 8,
    n_slices: int = 64,
    mlp_ratio: int = 4,
    act: str = "gelu",
    dropout: float = 0.0,
    time_input: bool = False,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a Transolver for unstructured 1-D / 2-D / 3-D point clouds.

    Physics-Attention transformer with O(N·M·D + M²·D) cost via learnable
    slicing into M physical groups per head.

    Reference:
        Wu et al. *Transolver: A Fast Transformer Solver for PDEs on
        General Geometries* (ICML 2024). https://arxiv.org/abs/2402.02366

    Call signature: ``model(x_coords, x_func=None, t=None)`` where
    ``x_coords`` has shape ``(N, space_dim)`` and ``x_func`` has shape
    ``(N, fun_dim)``.

    Args:
        space_dim: Spatial dimensionality of the coordinates.
        fun_dim: Number of input function channels.
        out_features: Number of output channels.
        hidden_dim: Hidden dimension (must be divisible by ``n_heads``).
        n_layers: Number of Transolver blocks.
        n_heads: Number of attention heads.
        n_slices: Number of physical slices per head.
        mlp_ratio: FFN hidden ratio.
        act: Activation function name.
        dropout: Dropout probability inside attention and FFN.
        time_input: If True, model accepts a scalar timestep ``t``.
        key: JAX PRNG key.

    Returns:
        An ``equinox.Module`` (TransolverIrregular).
    """
    from .architectures.transolver import TransolverIrregular

    return TransolverIrregular(
        space_dim=space_dim,
        fun_dim=fun_dim,
        out_features=out_features,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_slices=n_slices,
        mlp_ratio=mlp_ratio,
        act=act,
        dropout=dropout,
        time_input=time_input,
        key=_resolve_key(key),
    )


def transolver2d(
    space_dim: int = 2,
    fun_dim: int = 1,
    out_features: int = 1,
    hidden_dim: int = 128,
    n_layers: int = 8,
    n_heads: int = 8,
    n_slices: int = 64,
    mlp_ratio: int = 4,
    act: str = "gelu",
    dropout: float = 0.0,
    kernel: int = 3,
    time_input: bool = False,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a Transolver for structured 2-D grids.

    Same as :func:`transolver`, but uses ``Conv2d`` projections inside
    Physics-Attention so each pixel sees a local ``kernel × kernel``
    neighbourhood before being assigned to slices. Inputs are channel-last
    grids: ``(H, W, space_dim)`` and ``(H, W, fun_dim)``.

    Reference:
        Wu et al. *Transolver: A Fast Transformer Solver for PDEs on
        General Geometries* (ICML 2024). https://arxiv.org/abs/2402.02366

    Returns:
        An ``equinox.Module`` (TransolverStructured2D).
    """
    from .architectures.transolver import TransolverStructured2D

    return TransolverStructured2D(
        space_dim=space_dim,
        fun_dim=fun_dim,
        out_features=out_features,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_slices=n_slices,
        mlp_ratio=mlp_ratio,
        act=act,
        dropout=dropout,
        kernel=kernel,
        time_input=time_input,
        key=_resolve_key(key),
    )


def transolver3d(
    space_dim: int = 3,
    fun_dim: int = 1,
    out_features: int = 1,
    hidden_dim: int = 128,
    n_layers: int = 8,
    n_heads: int = 8,
    n_slices: int = 32,
    mlp_ratio: int = 4,
    act: str = "gelu",
    dropout: float = 0.0,
    kernel: int = 3,
    time_input: bool = False,
    *,
    key: jax.Array | None = None,
) -> eqx.Module:
    """Create a Transolver for structured 3-D grids.

    Same as :func:`transolver2d` but with ``Conv3d`` projections. Inputs:
    ``(D, H, W, space_dim)`` and ``(D, H, W, fun_dim)``.

    Reference:
        Wu et al. *Transolver: A Fast Transformer Solver for PDEs on
        General Geometries* (ICML 2024). https://arxiv.org/abs/2402.02366

    Returns:
        An ``equinox.Module`` (TransolverStructured3D).
    """
    from .architectures.transolver import TransolverStructured3D

    return TransolverStructured3D(
        space_dim=space_dim,
        fun_dim=fun_dim,
        out_features=out_features,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_slices=n_slices,
        mlp_ratio=mlp_ratio,
        act=act,
        dropout=dropout,
        kernel=kernel,
        time_input=time_input,
        key=_resolve_key(key),
    )
