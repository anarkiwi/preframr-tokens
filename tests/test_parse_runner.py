"""Smoke tests for ``preframr_tokens.parse_runner``. Comprehensive end-to-end coverage of the parser stage lives in the main `preframr` repo where the corpus + dump fixtures are available; here we only verify the module imports cleanly and the public API is exposed."""

import unittest


class TestModuleImport(unittest.TestCase):
    def test_module_imports(self):
        from preframr_tokens import parse_runner

        self.assertTrue(hasattr(parse_runner, "write_df"))
        self.assertTrue(hasattr(parse_runner, "parse_corpus"))


if __name__ == "__main__":
    unittest.main()
