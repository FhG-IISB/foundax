import sys
import unittest

from foundax import _vendors


class VendorPathTests(unittest.TestCase):
    def test_ensure_repo_on_path_inserts_once(self):
        repo_name = "jax_mpp"
        repo_path = _vendors._repos_root() / repo_name
        self.assertTrue(repo_path.exists())

        repo_path_str = str(repo_path)
        old_sys_path = list(sys.path)
        try:
            sys.path = [p for p in sys.path if p != repo_path_str]

            returned = _vendors.ensure_repo_on_path(repo_name)
            self.assertEqual(returned, repo_path)
            self.assertEqual(sys.path[0], repo_path_str)

            _vendors.ensure_repo_on_path(repo_name)
            self.assertEqual(sys.path.count(repo_path_str), 1)
        finally:
            sys.path = old_sys_path

    def test_ensure_repo_on_path_missing_repo_raises(self):
        with self.assertRaises(ImportError):
            _vendors.ensure_repo_on_path("does_not_exist_repo")


if __name__ == "__main__":
    unittest.main()
