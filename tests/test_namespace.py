import unittest
from pathlib import Path

import foundax


class NamespaceTests(unittest.TestCase):
    def test_top_level_namespaces_present(self):
        for name in [
            "poseidon",
            "walrus",
            "morph",
            "mpp",
            "bcat",
            "pdeformer2",
            "dpot",
            "prose",
        ]:
            self.assertTrue(hasattr(foundax, name))

    def test_version_aliases_match_jno_surface(self):
        self.assertIs(foundax.morph.ti, foundax.morph.Ti)
        self.assertIs(foundax.morph.s, foundax.morph.S)
        self.assertIs(foundax.morph.m, foundax.morph.M)
        self.assertIs(foundax.morph.l, foundax.morph.L)

        self.assertIs(foundax.mpp.ti, foundax.mpp.Ti)
        self.assertIs(foundax.mpp.s, foundax.mpp.S)
        self.assertIs(foundax.mpp.b, foundax.mpp.B)
        self.assertIs(foundax.mpp.l, foundax.mpp.L)

        self.assertIs(foundax.poseidon.t, foundax.poseidon.T)
        self.assertIs(foundax.poseidon.b, foundax.poseidon.B)
        self.assertIs(foundax.poseidon.l, foundax.poseidon.L)

        self.assertIs(foundax.dpot.ti, foundax.dpot.Ti)
        self.assertIs(foundax.dpot.s, foundax.dpot.S)
        self.assertIs(foundax.dpot.m, foundax.dpot.M)
        self.assertIs(foundax.dpot.l, foundax.dpot.L)
        self.assertIs(foundax.dpot.h, foundax.dpot.H)

    def test_single_variant_aliases(self):
        self.assertIs(foundax.walrus.default, foundax.walrus.base)
        self.assertTrue(callable(foundax.walrus.v1))

        self.assertIs(foundax.bcat.default, foundax.bcat.base)
        self.assertTrue(callable(foundax.bcat.v1))

    def test_vendored_repo_roots_exist(self):
        repo_root = Path(__file__).resolve().parents[1] / "repos"
        expected = [
            "jax_morph",
            "jax_mpp",
            "jax_poseidon",
            "jax_walrus",
            "jax_bcat",
            "jax_pdeformer2",
            "jax_dpot",
            "jax_prose",
        ]
        for repo in expected:
            self.assertTrue(
                (repo_root / repo).exists(), "missing vendored repo: {}".format(repo)
            )


if __name__ == "__main__":
    unittest.main()
