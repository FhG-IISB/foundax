import importlib as _importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import foundax.bcat as bcat
import foundax.dpot as dpot
import foundax.morph as morph
import foundax.mpp as mpp
import foundax.pdeformer2 as pdeformer2
import foundax.poseidon as poseidon
import foundax.prose as prose
import foundax.walrus as walrus


class Recorder:
    def __init__(self):
        self.calls = []

    def record(self, name):
        def _fn(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return {"name": name, "args": args, "kwargs": kwargs}

        return _fn


def _patch_module(module, fake_vendor_module):
    seen = {}
    _real_import = _importlib.import_module

    def _ensure(name):
        seen["repo"] = name

    def _import(name):
        # Intercept only vendor repos (jax_*) and fall through for all other
        # imports (JAX internals, jaxlib, etc.) to avoid breaking XLA init.
        if name.startswith("jax_"):
            seen["import"] = name
            return fake_vendor_module
        return _real_import(name)

    return seen, _ensure, _import


class DispatchTests(unittest.TestCase):
    def test_bcat_dispatch(self):
        rec = Recorder()
        fake = SimpleNamespace(BCAT=rec.record("BCAT"))
        seen, ensure, importer = _patch_module(bcat, fake)

        with (
            patch.object(bcat, "ensure_repo_on_path", ensure),
            patch.object(bcat.importlib, "import_module", importer),
        ):
            out = bcat.base()

        self.assertEqual(seen["repo"], "jax_bcat")
        self.assertEqual(seen["import"], "jax_bcat.model_eqx")
        self.assertEqual(out["name"], "BCAT")

    def test_walrus_dispatch(self):
        rec = Recorder()
        fake = SimpleNamespace(IsotropicModel=rec.record("IsotropicModel"))
        seen, ensure, importer = _patch_module(walrus, fake)

        with (
            patch.object(walrus, "ensure_repo_on_path", ensure),
            patch.object(walrus.importlib, "import_module", importer),
        ):
            out = walrus.base(processor_blocks=40)

        self.assertEqual(seen["repo"], "jax_walrus")
        self.assertEqual(seen["import"], "jax_walrus.model_eqx")
        self.assertEqual(out["name"], "IsotropicModel")

    def test_dpot_dispatch(self):
        # dpot uses _adapted_cls() which subclasses DPOTNet; provide a real base
        _FakeDPOTNet = type("DPOTNet", (), {"__init__": lambda self, **kw: setattr(self, "_kw", kw)})
        fake = SimpleNamespace(DPOTNet=_FakeDPOTNet)
        seen, ensure, importer = _patch_module(dpot, fake)

        with (
            patch.object(dpot, "ensure_repo_on_path", ensure),
            patch.object(dpot.importlib, "import_module", importer),
        ):
            dpot.H()

        self.assertEqual(seen["repo"], "jax_dpot")
        self.assertEqual(seen["import"], "jax_dpot.model_eqx")

    def test_morph_dispatch(self):
        # morph uses _adapted_cls() which subclasses ViT3DRegression
        _FakeViT = type("ViT3DRegression", (), {"__init__": lambda self, **kw: setattr(self, "_kw", kw)})
        fake = SimpleNamespace(ViT3DRegression=_FakeViT)
        seen, ensure, importer = _patch_module(morph, fake)

        with (
            patch.object(morph, "ensure_repo_on_path", ensure),
            patch.object(morph.importlib, "import_module", importer),
        ):
            morph.Ti()

        self.assertEqual(seen["repo"], "jax_morph")
        self.assertEqual(seen["import"], "jax_morph.model_eqx")

    def test_mpp_dispatch(self):
        # mpp uses _adapted_cls() which subclasses AViT
        _FakeAViT = type("AViT", (), {"__init__": lambda self, **kw: setattr(self, "_kw", kw)})
        fake = SimpleNamespace(AViT=_FakeAViT)
        seen, ensure, importer = _patch_module(mpp, fake)

        with (
            patch.object(mpp, "ensure_repo_on_path", ensure),
            patch.object(mpp.importlib, "import_module", importer),
        ):
            mpp.B(n_states=12)

        self.assertEqual(seen["repo"], "jax_mpp")
        self.assertEqual(seen["import"], "jax_mpp.avit_eqx")

    def test_poseidon_dispatch(self):
        rec = Recorder()
        # _build imports "jax_poseidon" (for ScOTConfig) then "jax_poseidon.scot_eqx"
        # (for ScOT); both intercepts return the same fake
        fake = SimpleNamespace(
            ScOTConfig=lambda **kw: kw,
            ScOT=rec.record("ScOT"),
        )
        seen, ensure, importer = _patch_module(poseidon, fake)

        with (
            patch.object(poseidon, "ensure_repo_on_path", ensure),
            patch.object(poseidon.importlib, "import_module", importer),
        ):
            out = poseidon.L()

        self.assertEqual(seen["repo"], "jax_poseidon")
        self.assertEqual(seen["import"], "jax_poseidon.scot_eqx")
        self.assertEqual(out["name"], "ScOT")

    def test_prose_dispatch(self):
        rec = Recorder()
        fake = SimpleNamespace(
            PROSE1to1=rec.record("PROSE1to1"),
            PROSE2to1=rec.record("PROSE2to1"),
            PROSEODE2to1=rec.record("PROSEODE2to1"),
            PROSEPDE2to1=rec.record("PROSEPDE2to1"),
        )
        seen, ensure, importer = _patch_module(prose, fake)

        with (
            patch.object(prose, "ensure_repo_on_path", ensure),
            patch.object(prose.importlib, "import_module", importer),
        ):
            out = prose.pde_2to1(n_words=100, pad_index=0, x_grid_size=128)

        self.assertEqual(seen["repo"], "jax_prose")
        self.assertEqual(seen["import"], "jax_prose.model_eqx")
        self.assertEqual(out["name"], "PROSEPDE2to1")

    def test_pdeformer2_dispatch(self):
        rec = Recorder()
        fake = SimpleNamespace(PDEformer=rec.record("PDEformer"))
        seen, ensure, importer = _patch_module(pdeformer2, fake)

        with (
            patch.object(pdeformer2, "ensure_repo_on_path", ensure),
            patch.object(pdeformer2.importlib, "import_module", importer),
        ):
            out = pdeformer2.fast()

        self.assertEqual(seen["repo"], "jax_pdeformer2")
        self.assertEqual(seen["import"], "jax_pdeformer2.model_eqx")
        self.assertEqual(out["name"], "PDEformer")


if __name__ == "__main__":
    unittest.main()
