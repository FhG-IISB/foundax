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

    def _ensure(name):
        seen["repo"] = name

    def _import(name):
        seen["import"] = name
        return fake_vendor_module

    return seen, _ensure, _import


class DispatchTests(unittest.TestCase):
    def test_morph_dispatch(self):
        rec = Recorder()
        fake = SimpleNamespace(
            morph_Ti=rec.record("morph_Ti"),
            morph_S=rec.record("morph_S"),
            morph_M=rec.record("morph_M"),
            morph_L=rec.record("morph_L"),
        )
        seen, ensure, importer = _patch_module(morph, fake)

        with (
            patch.object(morph, "ensure_repo_on_path", ensure),
            patch.object(morph.importlib, "import_module", importer),
        ):
            out = morph.Ti(1, foo=2)

        self.assertEqual(seen["repo"], "jax_morph")
        self.assertEqual(seen["import"], "jax_morph")
        self.assertEqual(out["name"], "morph_Ti")
        self.assertEqual(out["args"], (1,))
        self.assertEqual(out["kwargs"], {"foo": 2})

    def test_mpp_dispatch(self):
        rec = Recorder()
        fake = SimpleNamespace(
            avit_Ti=rec.record("avit_Ti"),
            avit_S=rec.record("avit_S"),
            avit_B=rec.record("avit_B"),
            avit_L=rec.record("avit_L"),
        )
        seen, ensure, importer = _patch_module(mpp, fake)

        with (
            patch.object(mpp, "ensure_repo_on_path", ensure),
            patch.object(mpp.importlib, "import_module", importer),
        ):
            out = mpp.B(n_states=12)

        self.assertEqual(seen["repo"], "jax_mpp")
        self.assertEqual(seen["import"], "jax_mpp")
        self.assertEqual(out["name"], "avit_B")
        self.assertEqual(out["kwargs"], {"n_states": 12})

    def test_poseidon_dispatch(self):
        rec = Recorder()
        fake = SimpleNamespace(
            poseidonT=rec.record("poseidonT"),
            poseidonB=rec.record("poseidonB"),
            poseidonL=rec.record("poseidonL"),
        )
        seen, ensure, importer = _patch_module(poseidon, fake)

        with (
            patch.object(poseidon, "ensure_repo_on_path", ensure),
            patch.object(poseidon.importlib, "import_module", importer),
        ):
            out = poseidon.L(num_in_channels=1)

        self.assertEqual(seen["repo"], "jax_poseidon")
        self.assertEqual(seen["import"], "jax_poseidon")
        self.assertEqual(out["name"], "poseidonL")

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
        self.assertEqual(seen["import"], "jax_walrus")
        self.assertEqual(out["name"], "IsotropicModel")

    def test_bcat_dispatch(self):
        rec = Recorder()
        fake = SimpleNamespace(bcat_default=rec.record("bcat_default"))
        seen, ensure, importer = _patch_module(bcat, fake)

        with (
            patch.object(bcat, "ensure_repo_on_path", ensure),
            patch.object(bcat.importlib, "import_module", importer),
        ):
            out = bcat.base()

        self.assertEqual(seen["repo"], "jax_bcat")
        self.assertEqual(seen["import"], "jax_bcat")
        self.assertEqual(out["name"], "bcat_default")

    def test_dpot_dispatch(self):
        rec = Recorder()
        fake = SimpleNamespace(
            dpot_ti=rec.record("dpot_ti"),
            dpot_s=rec.record("dpot_s"),
            dpot_m=rec.record("dpot_m"),
            dpot_l=rec.record("dpot_l"),
            dpot_h=rec.record("dpot_h"),
        )
        seen, ensure, importer = _patch_module(dpot, fake)

        with (
            patch.object(dpot, "ensure_repo_on_path", ensure),
            patch.object(dpot.importlib, "import_module", importer),
        ):
            out = dpot.H()

        self.assertEqual(seen["repo"], "jax_dpot")
        self.assertEqual(seen["import"], "jax_dpot")
        self.assertEqual(out["name"], "dpot_h")

    def test_prose_dispatch(self):
        rec = Recorder()
        fake = SimpleNamespace(
            prose_fd_1to1=rec.record("prose_fd_1to1"),
            prose_fd_2to1=rec.record("prose_fd_2to1"),
            prose_ode_2to1=rec.record("prose_ode_2to1"),
            prose_pde_2to1=rec.record("prose_pde_2to1"),
        )
        seen, ensure, importer = _patch_module(prose, fake)

        with (
            patch.object(prose, "ensure_repo_on_path", ensure),
            patch.object(prose.importlib, "import_module", importer),
        ):
            out = prose.pde_2to1(x_num=128)

        self.assertEqual(seen["repo"], "jax_prose")
        self.assertEqual(seen["import"], "jax_prose")
        self.assertEqual(out["name"], "prose_pde_2to1")
        self.assertEqual(out["kwargs"], {"x_num": 128})

    def test_pdeformer2_dispatch(self):
        rec = Recorder()
        fake = SimpleNamespace(
            PDEFORMER_SMALL_CONFIG={"name": "small"},
            PDEFORMER_BASE_CONFIG={"name": "base"},
            PDEFORMER_FAST_CONFIG={"name": "fast"},
            create_pdeformer_from_config=rec.record("create_pdeformer_from_config"),
        )
        seen, ensure, importer = _patch_module(pdeformer2, fake)

        with (
            patch.object(pdeformer2, "ensure_repo_on_path", ensure),
            patch.object(pdeformer2.importlib, "import_module", importer),
        ):
            out = pdeformer2.fast(alpha=1)

        self.assertEqual(seen["repo"], "jax_pdeformer2")
        self.assertEqual(seen["import"], "jax_pdeformer2")
        self.assertEqual(out["name"], "create_pdeformer_from_config")
        self.assertEqual(out["args"][0], {"model": {"name": "fast"}})
        self.assertEqual(out["kwargs"], {"alpha": 1})


if __name__ == "__main__":
    unittest.main()
