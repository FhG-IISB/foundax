"""Utility to make a module callable — ``foundax.<model>(**kw)``."""

import sys
import types


def install(module_name, call_fn):
    """Make *module_name* callable — mutates the existing module in-place.

    Changing ``__class__`` preserves the module's ``__dict__`` object so that
    function ``__globals__`` references (and ``unittest.mock.patch.object``
    patches) continue to work correctly.
    """
    mod = sys.modules[module_name]

    class _Mod(types.ModuleType):
        def __call__(self, **kwargs):
            return call_fn(**kwargs)

    mod.__class__ = _Mod
