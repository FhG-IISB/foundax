"""Utility to make a module callable — ``foundax.<model>(**kw)``."""

import sys
import types


def install(module_name, call_fn):
    """Replace *module_name* in ``sys.modules`` with a callable version."""
    old = sys.modules[module_name]

    class _Mod(types.ModuleType):
        def __call__(self, **kwargs):
            return call_fn(**kwargs)

    new = _Mod(module_name, old.__doc__)
    new.__dict__.update(old.__dict__)
    sys.modules[module_name] = new
