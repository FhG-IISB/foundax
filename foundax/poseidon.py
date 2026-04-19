"""Poseidon -- Efficient Foundation Models for PDEs.

**Paper:** Herde et al., *"Poseidon: Efficient Foundation Models for PDEs"* (2024)
https://arxiv.org/abs/2405.19101

Architecture: Scalable Operator Transformer (ScOT) -- Swin-Transformer
backbone with U-Net-style skip connections for multi-scale operator learning.

Usage::

    model = foundax.poseidon.T(num_channels=1, num_out_channels=1)
    model = foundax.poseidon(embed_dim=96, depths=(8, 8, 8, 8))
"""

import importlib
from typing import Any, Callable, List, Optional, Tuple

import jax

from ._vendors import ensure_repo_on_path
from . import _callable_module


def _build(
    name="poseidonT",
    image_size=128,
    patch_size=4,
    num_channels=4,
    num_out_channels=4,
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    skip_connections=(2, 2, 2, 0),
    window_size=16,
    mlp_ratio=4.0,
    qkv_bias=True,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    drop_path_rate=0.0,
    hidden_act: Callable = jax.nn.gelu,
    use_absolute_embeddings=False,
    initializer_range=0.02,
    layer_norm_eps=1e-5,
    p=1,
    channel_slice_list_normalized_loss=None,
    residual_model="convnext",
    use_conditioning=True,
    learn_residual=False,
    pretrained_window_sizes=(0, 0, 0, 0),
    chunk_size_feed_forward=0,
    output_attentions=False,
    output_hidden_states=False,
    use_return_dict=True,
    *,
    key=None,
):
    import jax

    ensure_repo_on_path("jax_poseidon")
    mod = importlib.import_module("jax_poseidon")
    config = mod.ScOTConfig(
        name=name,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        num_out_channels=num_out_channels,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        skip_connections=skip_connections,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        drop_path_rate=drop_path_rate,
        hidden_act=hidden_act,
        use_absolute_embeddings=use_absolute_embeddings,
        initializer_range=initializer_range,
        layer_norm_eps=layer_norm_eps,
        p=p,
        channel_slice_list_normalized_loss=channel_slice_list_normalized_loss,
        residual_model=residual_model,
        use_conditioning=use_conditioning,
        learn_residual=learn_residual,
        pretrained_window_sizes=pretrained_window_sizes,
        chunk_size_feed_forward=chunk_size_feed_forward,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        use_return_dict=use_return_dict,
    )
    eqx_mod = importlib.import_module("jax_poseidon.scot_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    return eqx_mod.ScOT(config=config, use_conditioning=use_conditioning, key=key)


def T(
    name="poseidonT",
    image_size=128,
    patch_size=4,
    num_channels=4,
    num_out_channels=4,
    embed_dim=48,
    depths=(4, 4, 4, 4),
    num_heads=(3, 6, 12, 24),
    skip_connections=(2, 2, 2, 0),
    window_size=16,
    mlp_ratio=4.0,
    qkv_bias=True,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    drop_path_rate=0.0,
    hidden_act: Callable = jax.nn.gelu,
    use_absolute_embeddings=False,
    initializer_range=0.02,
    layer_norm_eps=1e-5,
    p=1,
    channel_slice_list_normalized_loss=None,
    residual_model="convnext",
    use_conditioning=True,
    learn_residual=False,
    pretrained_window_sizes=(0, 0, 0, 0),
    chunk_size_feed_forward=0,
    output_attentions=False,
    output_hidden_states=False,
    use_return_dict=True,
):
    """Poseidon-T (Tiny) ~20.8 M params.

    Scalable Operator Transformer (ScOT) with Swin-Transformer backbone and
    U-Net-style skip connections for multi-scale operator learning.

    Reference:
        Herde et al., *Poseidon: Efficient Foundation Models for PDEs*
        (2024). https://arxiv.org/abs/2405.19101

    Example::

        model = foundax.poseidon.T(num_channels=1, num_out_channels=1)

    Shape:
        - Input: ``(B, H, W, num_channels)``; optional time ``(B,)``
        - Output: ``(B, H, W, num_out_channels)``

    See Also:
        :func:`B`, :func:`L`

    Args:
        name: Model identifier.
        image_size: Spatial resolution of the input grid.
        patch_size: Patch size for input tokenisation.
        num_channels: Number of input physical channels.
        num_out_channels: Number of output physical channels.
        embed_dim: Base embedding dimension (doubled per stage).
        depths: Number of Swin-Transformer blocks per encoder stage.
        num_heads: Number of attention heads per stage.
        skip_connections: Number of skip-connection blocks between
            encoder and decoder at each stage (0 = none).
        window_size: Swin-Transformer local window size.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        qkv_bias: Add learnable bias to query/key/value projections.
        hidden_dropout_prob: Dropout probability on hidden states.
        attention_probs_dropout_prob: Dropout on attention weights.
        drop_path_rate: Stochastic depth rate (linearly increases).
        hidden_act: Activation callable (e.g. ``jax.nn.gelu``,
            ``jax.nn.relu``).
        use_absolute_embeddings: Use absolute instead of relative
            positional embeddings.
        initializer_range: Std-dev for weight initialisation.
        layer_norm_eps: Epsilon for layer normalisation.
        p: Conditioning parameter ``p`` passed to the residual block.
        channel_slice_list_normalized_loss: Channel indices used for
            the normalised loss term (``None`` → ``[0, 1, 3, 4]``).
        residual_model: Residual block type (``"convnext"``).
        use_conditioning: Enable input conditioning via the residual
            pathway.
        learn_residual: Learn an additive residual correction.
        pretrained_window_sizes: Per-stage window sizes from a
            pre-trained checkpoint (``0`` = not pre-trained).
        chunk_size_feed_forward: Chunk the FFN for lower peak memory
            (``0`` = no chunking).
        output_attentions: Return attention weight tensors.
        output_hidden_states: Return hidden states from every layer.
        use_return_dict: Return a dataclass instead of a plain tuple.

    Returns:
        An ``equinox.Module`` (ScOT).
    """
    if channel_slice_list_normalized_loss is None:
        channel_slice_list_normalized_loss = [0, 1, 3, 4]
    return _build(**{k: v for k, v in locals().items()})


def B(
    name="poseidonB",
    image_size=128,
    patch_size=4,
    num_channels=4,
    num_out_channels=4,
    embed_dim=96,
    depths=(8, 8, 8, 8),
    num_heads=(3, 6, 12, 24),
    skip_connections=(2, 2, 2, 0),
    window_size=16,
    mlp_ratio=4.0,
    qkv_bias=True,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    drop_path_rate=0.0,
    hidden_act: Callable = jax.nn.gelu,
    use_absolute_embeddings=False,
    initializer_range=0.02,
    layer_norm_eps=1e-5,
    p=1,
    channel_slice_list_normalized_loss=None,
    residual_model="convnext",
    use_conditioning=True,
    learn_residual=False,
    pretrained_window_sizes=(0, 0, 0, 0),
    chunk_size_feed_forward=0,
    output_attentions=False,
    output_hidden_states=False,
    use_return_dict=True,
):
    """Poseidon-B (Base) ~157.7 M params.

    Scalable Operator Transformer (ScOT) with Swin-Transformer backbone and
    U-Net-style skip connections for multi-scale operator learning.

    Reference:
        Herde et al., *Poseidon: Efficient Foundation Models for PDEs*
        (2024). https://arxiv.org/abs/2405.19101

    Example::

        model = foundax.poseidon.B(num_channels=1, num_out_channels=1)

    Shape:
        - Input: ``(B, H, W, num_channels)``; optional time ``(B,)``
        - Output: ``(B, H, W, num_out_channels)``

    See Also:
        :func:`T`, :func:`L`

    Args:
        name: Model identifier.
        image_size: Spatial resolution of the input grid.
        patch_size: Patch size for input tokenisation.
        num_channels: Number of input physical channels.
        num_out_channels: Number of output physical channels.
        embed_dim: Base embedding dimension (doubled per stage).
        depths: Number of Swin-Transformer blocks per encoder stage.
        num_heads: Number of attention heads per stage.
        skip_connections: Number of skip-connection blocks between
            encoder and decoder at each stage (0 = none).
        window_size: Swin-Transformer local window size.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        qkv_bias: Add learnable bias to query/key/value projections.
        hidden_dropout_prob: Dropout probability on hidden states.
        attention_probs_dropout_prob: Dropout on attention weights.
        drop_path_rate: Stochastic depth rate (linearly increases).
        hidden_act: Activation callable (e.g. ``jax.nn.gelu``,
            ``jax.nn.relu``).
        use_absolute_embeddings: Use absolute instead of relative
            positional embeddings.
        initializer_range: Std-dev for weight initialisation.
        layer_norm_eps: Epsilon for layer normalisation.
        p: Conditioning parameter ``p`` passed to the residual block.
        channel_slice_list_normalized_loss: Channel indices used for
            the normalised loss term (``None`` → ``[0, 1, 3, 4]``).
        residual_model: Residual block type (``"convnext"``).
        use_conditioning: Enable input conditioning via the residual
            pathway.
        learn_residual: Learn an additive residual correction.
        pretrained_window_sizes: Per-stage window sizes from a
            pre-trained checkpoint (``0`` = not pre-trained).
        chunk_size_feed_forward: Chunk the FFN for lower peak memory
            (``0`` = no chunking).
        output_attentions: Return attention weight tensors.
        output_hidden_states: Return hidden states from every layer.
        use_return_dict: Return a dataclass instead of a plain tuple.

    Returns:
        An ``equinox.Module`` (ScOT).
    """
    if channel_slice_list_normalized_loss is None:
        channel_slice_list_normalized_loss = [0, 1, 3, 4]
    return _build(**{k: v for k, v in locals().items()})


def L(
    name="poseidonL",
    image_size=128,
    patch_size=4,
    num_channels=4,
    num_out_channels=4,
    embed_dim=192,
    depths=(8, 8, 8, 8),
    num_heads=(3, 6, 12, 24),
    skip_connections=(2, 2, 2, 0),
    window_size=16,
    mlp_ratio=4.0,
    qkv_bias=True,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    drop_path_rate=0.0,
    hidden_act: Callable = jax.nn.gelu,
    use_absolute_embeddings=False,
    initializer_range=0.02,
    layer_norm_eps=1e-5,
    p=1,
    channel_slice_list_normalized_loss=None,
    residual_model="convnext",
    use_conditioning=True,
    learn_residual=False,
    pretrained_window_sizes=(0, 0, 0, 0),
    chunk_size_feed_forward=0,
    output_attentions=False,
    output_hidden_states=False,
    use_return_dict=True,
):
    """Poseidon-L (Large) ~628.6 M params.

    Scalable Operator Transformer (ScOT) with Swin-Transformer backbone and
    U-Net-style skip connections for multi-scale operator learning.

    Reference:
        Herde et al., *Poseidon: Efficient Foundation Models for PDEs*
        (2024). https://arxiv.org/abs/2405.19101

    Example::

        model = foundax.poseidon.L(num_channels=1, num_out_channels=1)

    Shape:
        - Input: ``(B, H, W, num_channels)``; optional time ``(B,)``
        - Output: ``(B, H, W, num_out_channels)``

    See Also:
        :func:`T`, :func:`B`

    Args:
        name: Model identifier.
        image_size: Spatial resolution of the input grid.
        patch_size: Patch size for input tokenisation.
        num_channels: Number of input physical channels.
        num_out_channels: Number of output physical channels.
        embed_dim: Base embedding dimension (doubled per stage).
        depths: Number of Swin-Transformer blocks per encoder stage.
        num_heads: Number of attention heads per stage.
        skip_connections: Number of skip-connection blocks between
            encoder and decoder at each stage (0 = none).
        window_size: Swin-Transformer local window size.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        qkv_bias: Add learnable bias to query/key/value projections.
        hidden_dropout_prob: Dropout probability on hidden states.
        attention_probs_dropout_prob: Dropout on attention weights.
        drop_path_rate: Stochastic depth rate (linearly increases).
        hidden_act: Activation callable (e.g. ``jax.nn.gelu``,
            ``jax.nn.relu``).
        use_absolute_embeddings: Use absolute instead of relative
            positional embeddings.
        initializer_range: Std-dev for weight initialisation.
        layer_norm_eps: Epsilon for layer normalisation.
        p: Conditioning parameter ``p`` passed to the residual block.
        channel_slice_list_normalized_loss: Channel indices used for
            the normalised loss term (``None`` → ``[0, 1, 3, 4]``).
        residual_model: Residual block type (``"convnext"``).
        use_conditioning: Enable input conditioning via the residual
            pathway.
        learn_residual: Learn an additive residual correction.
        pretrained_window_sizes: Per-stage window sizes from a
            pre-trained checkpoint (``0`` = not pre-trained).
        chunk_size_feed_forward: Chunk the FFN for lower peak memory
            (``0`` = no chunking).
        output_attentions: Return attention weight tensors.
        output_hidden_states: Return hidden states from every layer.
        use_return_dict: Return a dataclass instead of a plain tuple.

    Returns:
        An ``equinox.Module`` (ScOT).
    """
    if channel_slice_list_normalized_loss is None:
        channel_slice_list_normalized_loss = [0, 1, 3, 4]
    return _build(**{k: v for k, v in locals().items()})


t = T
b = B
l = L

__all__ = ["T", "B", "L", "t", "b", "l"]

_callable_module.install(__name__, _build)
