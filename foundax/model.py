"""Lightweight container for initialized Flax models.

``FlaxModel`` holds everything needed to run (and later train) a Flax
``nn.Module``:

* ``apply_fn`` – the ``model.apply`` callable
* ``params``   – the initialized parameter pytree
* ``post_fn``  – optional output post-processing (e.g. ``lambda x: x.output``)
* ``default_kwargs`` – keyword arguments forwarded on every call
  (e.g. ``deterministic=True``)

foundax factories always return a ``FlaxModel``.  The object is
framework-agnostic (no Equinox dependency) so that consumers like jNO
can wrap it into their own ``FlaxModelWrapper`` / ``Model`` pipeline.
"""

from typing import Any, Callable, Optional


class FlaxModel:
    """Initialized Flax model container.

    Parameters
    ----------
    apply_fn : callable
        Typically ``flax_module.apply``.
    params : Any
        Flax parameter tree (output of ``model.init(...)``).
    post_fn : callable, optional
        Applied to the raw output of ``apply_fn`` before returning.
    **default_kwargs
        Extra keyword arguments forwarded to ``apply_fn`` on every call
        (e.g. ``deterministic=True``).
    """

    __slots__ = ("apply_fn", "params", "post_fn", "default_kwargs")

    def __init__(
        self,
        apply_fn: Callable,
        params: Any,
        post_fn: Optional[Callable] = None,
        **default_kwargs: Any,
    ):
        self.apply_fn = apply_fn
        self.params = params
        self.post_fn = post_fn
        self.default_kwargs = default_kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        merged = {**self.default_kwargs, **kwargs}
        result = self.apply_fn(self.params, *args, **merged)
        if self.post_fn is not None:
            result = self.post_fn(result)
        return result

    def __repr__(self) -> str:
        fn_name = getattr(self.apply_fn, "__name__", repr(self.apply_fn))
        return f"FlaxModel(apply_fn={fn_name}, post_fn={self.post_fn is not None})"
