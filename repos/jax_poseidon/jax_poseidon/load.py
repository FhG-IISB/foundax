"""
Poseidon Model Loading Utilities.

This module provides functions to instantiate Poseidon foundation models
for PDE solving, including pre-configured variants (T, B, L) and custom
ScOT architectures.

Reference:
    Herde et al., "Poseidon: Efficient Foundation Models for PDEs" (2024)
    https://arxiv.org/abs/2405.19101
"""

from .scot import ScOT, ScOTConfig
import jax
from typing import Optional, Tuple, List, Union
import jax.numpy as jnp
from flax.serialization import from_bytes


def poseidonT(
    rng: Optional[jax.random.PRNGKey] = None,
    weight_path: Optional[str] = None,
    verbose: bool = True,
) -> Union[ScOT, Tuple[ScOT, dict]]:
    """
    Initialize a Poseidon-T (Tiny) foundation model for PDE solving.

    Poseidon-T is the smallest variant with ~20.8M parameters, suitable for
    quick experimentation and resource-constrained environments.

    Model specifications:
        - Parameters: ~20.8M
        - Embedding dimension: 48
        - Depths: (4, 4, 4, 4)
        - Input/Output size: 128×128×4

    Args:
        rng: JAX random key for parameter initialization. Required if
            `weight_path` is provided to initialize the model structure
            before loading weights.
        weight_path: Path to pretrained weights file (.msgpack format).
            If provided along with `rng`, weights will be loaded and merged.
        verbose: If True, print information about weight loading and any
            shape mismatches. Default: True.

    Returns:
        If `weight_path` and `rng` are both provided:
            Tuple of (model, params) where params contains the loaded weights.
        Otherwise:
            Uninitialized ScOT model instance.

    Example:
        >>> # Create uninitialized model
        >>> model = poseidonT()
        >>>
        >>> # Load pretrained weights
        >>> rng = jax.random.PRNGKey(0)
        >>> model, params = poseidonT(rng=rng, weight_path="poseidon_t.msgpack")
        >>>
        >>> # Run inference
        >>> output = model.apply(params, pixel_values=x, time=t, deterministic=True)

    See Also:
        poseidonB: Base variant (~157.7M parameters)
        poseidonL: Large variant (~628.6M parameters)
    """
    config = ScOTConfig(
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
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,
        channel_slice_list_normalized_loss=[0, 1, 3, 4],
        residual_model="convnext",
        use_conditioning=True,
        learn_residual=False,
        pretrained_window_sizes=(0, 0, 0, 0),
    )
    model = ScOT(config=config, use_conditioning=True)

    if weight_path is not None and rng is not None:
        return init_poseidon_with_weights(model, rng, weight_path, verbose)

    return model


def poseidonB(
    rng: Optional[jax.random.PRNGKey] = None,
    weight_path: Optional[str] = None,
    verbose: bool = True,
) -> Union[ScOT, Tuple[ScOT, dict]]:
    """
    Initialize a Poseidon-B (Base) foundation model for PDE solving.

    Poseidon-B is the medium-sized variant with ~157.7M parameters,
    offering a balance between performance and computational cost.

    Model specifications:
        - Parameters: ~157.7M
        - Embedding dimension: 96
        - Depths: (8, 8, 8, 8)
        - Input/Output size: 128×128×4

    Args:
        rng: JAX random key for parameter initialization. Required if
            `weight_path` is provided to initialize the model structure
            before loading weights.
        weight_path: Path to pretrained weights file (.msgpack format).
            If provided along with `rng`, weights will be loaded and merged.
        verbose: If True, print information about weight loading and any
            shape mismatches. Default: True.

    Returns:
        If `weight_path` and `rng` are both provided:
            Tuple of (model, params) where params contains the loaded weights.
        Otherwise:
            Uninitialized ScOT model instance.

    Example:
        >>> # Create uninitialized model
        >>> model = poseidonB()
        >>>
        >>> # Load pretrained weights
        >>> rng = jax.random.PRNGKey(0)
        >>> model, params = poseidonB(rng=rng, weight_path="poseidon_b.msgpack")
        >>>
        >>> # Run inference
        >>> output = model.apply(params, pixel_values=x, time=t, deterministic=True)

    See Also:
        poseidonT: Tiny variant (~20.8M parameters)
        poseidonL: Large variant (~628.6M parameters)
    """
    config = ScOTConfig(
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
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,
        channel_slice_list_normalized_loss=[0, 1, 3, 4],
        residual_model="convnext",
        use_conditioning=True,
        learn_residual=False,
        pretrained_window_sizes=(0, 0, 0, 0),
        chunk_size_feed_forward=0,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
    )
    model = ScOT(config=config, use_conditioning=True)

    if weight_path is not None and rng is not None:
        return init_poseidon_with_weights(model, rng, weight_path, verbose)

    return model


def poseidonL(
    rng: Optional[jax.random.PRNGKey] = None,
    weight_path: Optional[str] = None,
    verbose: bool = True,
) -> Union[ScOT, Tuple[ScOT, dict]]:
    """
    Initialize a Poseidon-L (Large) foundation model for PDE solving.

    Poseidon-L is the largest variant with ~628.6M parameters, providing
    the highest capacity for complex PDE systems at the cost of increased
    computational requirements.

    Model specifications:
        - Parameters: ~628.6M
        - Embedding dimension: 192
        - Depths: (8, 8, 8, 8)
        - Input/Output size: 128×128×4

    Args:
        rng: JAX random key for parameter initialization. Required if
            `weight_path` is provided to initialize the model structure
            before loading weights.
        weight_path: Path to pretrained weights file (.msgpack format).
            If provided along with `rng`, weights will be loaded and merged.
        verbose: If True, print information about weight loading and any
            shape mismatches. Default: True.

    Returns:
        If `weight_path` and `rng` are both provided:
            Tuple of (model, params) where params contains the loaded weights.
        Otherwise:
            Uninitialized ScOT model instance.

    Example:
        >>> # Create uninitialized model
        >>> model = poseidonL()
        >>>
        >>> # Load pretrained weights
        >>> rng = jax.random.PRNGKey(0)
        >>> model, params = poseidonL(rng=rng, weight_path="poseidon_l.msgpack")
        >>>
        >>> # Run inference
        >>> output = model.apply(params, pixel_values=x, time=t, deterministic=True)

    See Also:
        poseidonT: Tiny variant (~20.8M parameters)
        poseidonB: Base variant (~157.7M parameters)
    """
    config = ScOTConfig(
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
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,
        channel_slice_list_normalized_loss=[0, 1, 3, 4],
        residual_model="convnext",
        use_conditioning=True,
        learn_residual=False,
        pretrained_window_sizes=(0, 0, 0, 0),
        chunk_size_feed_forward=0,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
    )

    model = ScOT(config=config, use_conditioning=True)

    if weight_path is not None and rng is not None:
        return init_poseidon_with_weights(model, rng, weight_path, verbose)

    return model


def init_poseidon_with_weights(
    model: ScOT, rng: jax.random.PRNGKey, weight_path: str, verbose: bool = True
) -> Tuple[ScOT, dict]:
    """
    Initialize a ScOT model with pretrained weights from a file.

    This function performs the following steps:
        1. Creates fresh parameters by initializing the model with dummy inputs
        2. Loads pretrained weights from the specified msgpack file
        3. Merges pretrained weights with fresh parameters, handling shape

           mismatches (e.g., different channel dimensions)

    Args:
        model: An uninitialized ScOT model instance.
        rng: JAX random key used for initializing fresh parameters.
        weight_path: Path to the pretrained weights file in msgpack format.
        verbose: If True, print detailed information about the weight loading
            process, including any shape mismatches. Default: True.

    Returns:
        A tuple containing:
            - model: The same ScOT model instance (unchanged)
            - params: Dictionary of merged parameters ready for model.apply()

    Raises:
        FileNotFoundError: If the weight_path does not exist.
        ValueError: If the weights file is corrupted or incompatible.

    Example:
        >>> model = ScOT(config=my_config, use_conditioning=True)
        >>> rng = jax.random.PRNGKey(42)
        >>> model, params = init_poseidon_with_weights(
        ...     model, rng, "weights.msgpack", verbose=True
        ... )
        >>> # Use the model
        >>> output = model.apply(params, pixel_values=x, time=t, deterministic=True)

    Note:
        The function uses dummy inputs of shape (1, 128, 128, 4) for pixel_values
        and (1,) for time during initialization. The actual input shapes during
        inference can differ as long as they're compatible with the model config.
    """
    dummy_pixel_values = jnp.ones((1, 128, 128, 4))
    dummy_time = jnp.zeros((1,))

    fresh_params = model.init(
        {"params": rng, "dropout": rng},
        pixel_values=dummy_pixel_values,
        time=dummy_time,
        deterministic=False,
    )

    with open(weight_path, "rb") as f:
        pretrained_bytes = f.read()

    layer_params = from_bytes(fresh_params, pretrained_bytes)

    params = merge_pretrained_params(layer_params, fresh_params, verbose)

    return model, params


def merge_pretrained_params(
    pretrained_params: dict, new_params: dict, verbose: bool = True
) -> dict:
    """
    Merge pretrained weights with freshly initialized parameters.

    This function recursively traverses the parameter trees and:
        - Copies pretrained weights when shapes match
        - Falls back to fresh initialization when shapes differ
        - Handles missing keys in either parameter dict

    This is particularly useful when fine-tuning with different input/output
    channel dimensions than the pretrained model.

    Args:
        pretrained_params: Parameter dictionary loaded from pretrained weights.
        new_params: Freshly initialized parameter dictionary from model.init().
        verbose: If True, print information about shape mismatches and
            statistics about matched vs. reinitialized parameters. Default: True.

    Returns:
        Merged parameter dictionary with the same structure as new_params,
        containing pretrained weights where shapes match and fresh weights
        where they don't.

    Example:
        >>> # Load pretrained weights
        >>> with open("pretrained.msgpack", "rb") as f:
        ...     pretrained = from_bytes(template, f.read())
        >>>
        >>> # Initialize fresh params (possibly with different channels)
        >>> fresh = model.init(rng, dummy_input)
        >>>
        >>> # Merge, keeping pretrained where possible
        >>> merged = merge_pretrained_params(pretrained, fresh, verbose=True)
        Shape mismatch at params/embeddings/proj/kernel: (4, 4, 3, 48) -> (4, 4, 5, 48), reinitializing
        Pretrained weights: 20,750,000/20,774,444 params matched (99.88%), 24,444 reinitialized

    Note:
        The function counts parameters and reports statistics when verbose=True,
        helping diagnose how much of the pretrained model is being utilized.
    """
    stats = {"matched": 0, "replaced": 0}

    def count_params(arr):
        return arr.size if hasattr(arr, "size") else 0

    def merge(pretrained, new, path=""):
        if isinstance(pretrained, dict) and isinstance(new, dict):
            result = {}
            all_keys = set(list(pretrained.keys()) + list(new.keys()))

            for key in all_keys:
                current_path = f"{path}/{key}" if path else key

                if key in pretrained and key in new:
                    if isinstance(pretrained[key], dict):
                        result[key] = merge(pretrained[key], new[key], current_path)
                    else:
                        if pretrained[key].shape == new[key].shape:
                            result[key] = pretrained[key]
                            stats["matched"] += count_params(pretrained[key])
                        else:
                            if verbose:
                                print(
                                    f"Shape mismatch at {current_path}: "
                                    f"{pretrained[key].shape} -> {new[key].shape}, reinitializing"
                                )
                            result[key] = new[key]
                            stats["replaced"] += count_params(new[key])
                elif key in pretrained:
                    result[key] = pretrained[key]
                    if not isinstance(pretrained[key], dict):
                        stats["matched"] += count_params(pretrained[key])
                else:
                    result[key] = new[key]
                    if not isinstance(new[key], dict):
                        stats["replaced"] += count_params(new[key])

            return result
        else:
            if hasattr(pretrained, "shape") and hasattr(new, "shape"):
                if pretrained.shape == new.shape:
                    stats["matched"] += count_params(pretrained)
                    return pretrained
                else:
                    stats["replaced"] += count_params(new)
                    return new
            return new if new is not None else pretrained

    merged = merge(pretrained_params, new_params)

    total = stats["matched"] + stats["replaced"]
    pct = 100 * stats["matched"] / total if total > 0 else 0
    if verbose:
        print(
            f"Pretrained weights: {stats['matched']:,}/{total:,} params matched ({pct:.2f}%), "
            f"{stats['replaced']:,} reinitialized"
        )

    return merged


def scot(
    name: str,
    image_size: int,
    patch_size: int = 4,
    num_channels: int = 4,
    num_out_channels: int = 4,
    embed_dim: int = 48,
    depths: Tuple[int, int, int, int] = (4, 4, 4, 4),
    num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
    skip_connections: Tuple[int, int, int, int] = (2, 2, 2, 0),
    window_size: int = 16,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    hidden_dropout_prob: float = 0.0,
    attention_probs_dropout_prob: float = 0.0,
    drop_path_rate: float = 0.0,
    hidden_act: str = "gelu",
    use_absolute_embeddings: bool = False,
    initializer_range: float = 0.02,
    layer_norm_eps: float = 1e-5,
    p: int = 1,
    channel_slice_list_normalized_loss: Optional[List[int]] = None,
    residual_model: str = "convnext",
    use_conditioning: bool = True,
    learn_residual: bool = False,
    pretrained_window_sizes: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> ScOT:
    """
    Create a custom Scalable Operator Transformer (ScOT) model.

    ScOT combines Swin Transformer's efficient windowed attention with
    U-Net-style skip connections for multi-scale operator learning.
    It forms the backbone architecture of the Poseidon foundation models.

    Architecture overview::

        Input → PatchEmbed → [SwinBlock × d₁] → Downsample → [SwinBlock × d₂] → ...
                                    ↓ (skip)                        ↓ (skip)
        Output ← PatchExpand ← [SwinBlock × d₁] ← Upsample ← [SwinBlock × d₂] ← ...

    Key features:
        - Windowed self-attention for O(n) complexity instead of O(n²)
        - Shifted windows for cross-window information exchange
        - Skip connections for preserving multi-scale spatial information
        - Optional time/parameter conditioning for dynamic problems

    Args:
        name: Model identifier string for logging and checkpointing.
        image_size: Input image size in pixels (assumes square images).
        patch_size: Size of each patch for tokenization. Smaller patches
            give finer granularity but increase sequence length. Default: 4.
        num_channels: Number of input channels. Default: 4.
        num_out_channels: Number of output channels. Default: 4.
        embed_dim: Base embedding dimension. Doubled at each encoder stage.
            Default: 48.
        depths: Number of Swin Transformer blocks at each stage.
            Length determines the number of stages. Default: (4, 4, 4, 4).
        num_heads: Number of attention heads at each stage.
            Should divide embed_dim * 2^stage evenly. Default: (3, 6, 12, 24).
        skip_connections: Frequency of skip connections at each stage.
            Value of 2 means every other block has a skip. 0 means no skips.
            Default: (2, 2, 2, 0).
        window_size: Size of attention windows. Should divide
            image_size / patch_size evenly. Default: 16.
        mlp_ratio: Expansion ratio for the MLP (feed-forward) layers.
            Default: 4.0.
        qkv_bias: Whether to include bias terms in query/key/value
            projections. Default: True.
        hidden_dropout_prob: Dropout probability for hidden layers.
            Default: 0.0.
        attention_probs_dropout_prob: Dropout probability for attention
            weights. Default: 0.0.
        drop_path_rate: Stochastic depth rate for regularization.
            Default: 0.0.
        hidden_act: Activation function name ('gelu', 'relu', etc.).
            Default: 'gelu'.
        use_absolute_embeddings: Whether to add absolute position embeddings
            to patch embeddings. Default: False.
        initializer_range: Standard deviation for weight initialization.
            Default: 0.02.
        layer_norm_eps: Epsilon for layer normalization numerical stability.
            Default: 1e-5.
        p: Patch merging factor for downsampling. Default: 1.
        channel_slice_list_normalized_loss: List of channel indices for
            computing normalized loss. Default: [0, 1, 3, 4].
        residual_model: Type of residual block ('convnext' or others).
            Default: 'convnext'.
        use_conditioning: Enable time/parameter conditioning via FiLM layers.
            Required for time-dependent PDEs. Default: True.
        learn_residual: If True, model learns the residual (output - input)
            instead of the full output. Default: False.
        pretrained_window_sizes: Window sizes from pretrained model for
            relative position bias interpolation. Default: (0, 0, 0, 0).

    Returns:
        An uninitialized ScOT model instance ready for model.init().

    Example:
        >>> # Custom ScOT for 64x64 RGB images with 1 output channel
        >>> model = scot(
        ...     name="custom_scot",
        ...     image_size=64,
        ...     num_channels=3,
        ...     num_out_channels=1,
        ...     embed_dim=96,
        ...     depths=(2, 2, 6, 2),
        ...     window_size=8,
        ... )
        >>>
        >>> # Initialize parameters
        >>> rng = jax.random.PRNGKey(0)
        >>> params = model.init(
        ...     {"params": rng, "dropout": rng},
        ...     pixel_values=jnp.ones((1, 64, 64, 3)),
        ...     time=jnp.zeros((1,)),
        ...     deterministic=True,
        ... )

    See Also:
        poseidonT: Pre-configured tiny model (~20.8M params)
        poseidonB: Pre-configured base model (~157.7M params)
        poseidonL: Pre-configured large model (~628.6M params)

    Reference:
        Herde et al., "Poseidon: Efficient Foundation Models for PDEs" (2024)
        https://arxiv.org/abs/2405.19101
    """
    if channel_slice_list_normalized_loss is None:
        channel_slice_list_normalized_loss = [0, 1, 3, 4]

    config = ScOTConfig(
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
    )

    model = ScOT(config=config, use_conditioning=use_conditioning)
    return model