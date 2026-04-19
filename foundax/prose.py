"""PROSE -- JAX translation of the PROSE model family.

**Papers:**

- Sun et al., *"PROSE-FD"* (NeurIPS 2024 Workshop)
  https://arxiv.org/abs/2409.09811
- Sun et al., *"Towards a Foundation Model for PDEs"* (2024)
  https://arxiv.org/abs/2404.12355
- Lample & Charton, *"PROSE"* (Neural Networks 2024)
  https://doi.org/10.1016/j.neunet.2024.106707

Architecture: Transformer-based sequence-to-sequence models for
operator learning from finite-difference, ODE, and PDE data.

All factories return an ``equinox.Module`` directly.

Usage::

    model = foundax.prose.fd_1to1(x_num=64)
    model = foundax.prose.pde_2to1(n_words=100, pad_index=0)
"""

import importlib

from ._vendors import ensure_repo_on_path


def fd_1to1(
    x_num=128,
    max_output_dim=4,
    output_len=10,
    *,
    key=None,
):
    """PROSE finite-difference 1-to-1 model.

    Transformer-based sequence-to-sequence model that maps a single
    finite-difference input trajectory to an output trajectory.

    Reference:
        Sun et al., *PROSE-FD: A Multimodal PDE Foundation Model for
        Learning Multiple Operators for Forecasting Fluid Dynamics*
        (NeurIPS 2024 Workshop). https://arxiv.org/abs/2409.09811

    Example::

        model = foundax.prose.fd_1to1(x_num=64, max_output_dim=2)

    Shape:
        - Input: trajectory on an ``x_num × x_num`` grid, up to
          ``max_output_dim`` channels, ``output_len`` output steps.
        - Output: predicted trajectory of same spatial resolution.

    See Also:
        :func:`fd_2to1`, :func:`ode_2to1`, :func:`pde_2to1`

    Args:
        x_num: Spatial grid resolution (number of grid points).
        max_output_dim: Maximum number of output physical channels.
        output_len: Length of the output sequence.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (PROSE1to1).
    """
    import jax

    ensure_repo_on_path("jax_prose")
    mod = importlib.import_module("jax_prose.model_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    return mod.PROSE1to1(
        x_num=x_num, max_output_dim=max_output_dim, output_len=output_len, key=key
    )


def fd_2to1(
    n_words,
    x_num=128,
    max_output_dim=4,
    *,
    key=None,
):
    """PROSE finite-difference 2-to-1 model.

    Transformer that fuses a symbolic equation description **and** a
    finite-difference data trajectory into a single output trajectory.

    Reference:
        Sun et al., *PROSE-FD: A Multimodal PDE Foundation Model for
        Learning Multiple Operators for Forecasting Fluid Dynamics*
        (NeurIPS 2024 Workshop). https://arxiv.org/abs/2409.09811

    Example::

        model = foundax.prose.fd_2to1(n_words=100, x_num=64)

    Shape:
        - Input: symbol sequence + data trajectory on an
          ``x_num × x_num`` grid, up to ``max_output_dim`` channels.
        - Output: predicted trajectory of same spatial resolution.

    See Also:
        :func:`fd_1to1`, :func:`ode_2to1`, :func:`pde_2to1`

    Args:
        n_words: Size of the symbol vocabulary.
        x_num: Spatial grid resolution (number of grid points).
        max_output_dim: Maximum number of output physical channels.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (PROSE2to1).
    """
    import jax

    ensure_repo_on_path("jax_prose")
    mod = importlib.import_module("jax_prose.model_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    return mod.PROSE2to1(
        n_words=n_words, x_num=x_num, max_output_dim=max_output_dim, key=key
    )


def ode_2to1(
    n_words,
    pad_index,
    max_output_dimension=3,
    *,
    key=None,
):
    """PROSE ODE 2-to-1 model.

    Transformer that fuses a symbolic ODE description and observed data
    into a predicted trajectory.

    Reference:
        Lample & Charton, *PROSE: Predicting Multiple Operators and
        Symbolic Expressions using Multimodal Transformers*
        (Neural Networks 2024).
        https://doi.org/10.1016/j.neunet.2024.106707

    Example::

        model = foundax.prose.ode_2to1(n_words=100, pad_index=0)

    Shape:
        - Input: symbol sequence + ODE data, up to
          ``max_output_dimension`` state dimensions.
        - Output: predicted trajectory.

    See Also:
        :func:`fd_1to1`, :func:`fd_2to1`, :func:`pde_2to1`

    Args:
        n_words: Size of the symbol vocabulary.
        pad_index: Index used for padding tokens in the vocabulary.
        max_output_dimension: Maximum number of output state dimensions.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (PROSEODE2to1).
    """
    import jax

    ensure_repo_on_path("jax_prose")
    mod = importlib.import_module("jax_prose.model_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    return mod.PROSEODE2to1(
        n_words=n_words,
        pad_index=pad_index,
        max_output_dimension=max_output_dimension,
        key=key,
    )


def pde_2to1(
    n_words,
    pad_index,
    max_output_dimension=1,
    x_patch_size=1,
    x_grid_size=128,
    *,
    key=None,
):
    """PROSE PDE 2-to-1 model.

    Transformer that fuses a symbolic PDE description and observed data
    into a predicted spatio-temporal field.

    Reference:
        Sun et al., *Towards a Foundation Model for Partial Differential
        Equations: Multi-Operator Learning and Extrapolation* (2024).
        https://arxiv.org/abs/2404.12355

    Example::

        model = foundax.prose.pde_2to1(n_words=100, pad_index=0)

    Shape:
        - Input: symbol sequence + PDE data on an
          ``x_grid_size × x_grid_size`` grid, patched with
          ``x_patch_size``.
        - Output: predicted spatio-temporal field.

    See Also:
        :func:`fd_1to1`, :func:`fd_2to1`, :func:`ode_2to1`

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
    import jax

    ensure_repo_on_path("jax_prose")
    mod = importlib.import_module("jax_prose.model_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    return mod.PROSEPDE2to1(
        n_words=n_words,
        pad_index=pad_index,
        max_output_dimension=max_output_dimension,
        x_patch_size=x_patch_size,
        x_grid_size=x_grid_size,
        key=key,
    )


__all__ = ["fd_1to1", "fd_2to1", "ode_2to1", "pde_2to1"]
