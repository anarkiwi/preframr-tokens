"""Pin the public surface: every name in every module's ``__all__`` must resolve at import time."""

import importlib
import unittest
from pathlib import Path

PACKAGE = "preframr_tokens"
REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / PACKAGE


def _module_names():
    """Yield dotted module names under ``preframr_tokens`` (skipping __pycache__)."""
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if "__pycache__" in rel.parts:
            continue
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        yield ".".join(parts)


class TestEveryModuleAllResolves(unittest.TestCase):
    def test_every_all_symbol_resolves(self):
        offenders = []
        modules = list(_module_names())
        self.assertGreater(len(modules), 0, "no modules discovered")
        for mod_name in modules:
            try:
                mod = importlib.import_module(mod_name)
            except Exception as exc:
                offenders.append(f"{mod_name}: import failed: {exc!r}")
                continue
            names = getattr(mod, "__all__", None)
            if not names:
                continue
            for name in names:
                if not hasattr(mod, name):
                    offenders.append(
                        f"{mod_name}.__all__ lists {name!r} but it is not defined"
                    )
        if offenders:
            self.fail(
                "Public-surface contract violations:\n  " + "\n  ".join(offenders)
            )


class TestAllNamesAreStrings(unittest.TestCase):
    def test_all_entries_are_strings(self):
        offenders = []
        for mod_name in _module_names():
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            names = getattr(mod, "__all__", None)
            if names is None:
                continue
            for i, name in enumerate(names):
                if not isinstance(name, str):
                    offenders.append(
                        f"{mod_name}.__all__[{i}] is {type(name).__name__}, not str"
                    )
        if offenders:
            self.fail("Bad __all__ entries:\n  " + "\n  ".join(offenders))


if __name__ == "__main__":
    unittest.main()
