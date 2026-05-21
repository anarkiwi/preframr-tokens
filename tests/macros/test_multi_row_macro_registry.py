"""Property tests for the multi-row macro registry."""

import inspect
import unittest

from preframr_tokens.macros import loops as loops_mod
from preframr_tokens.macros.loops import (
    MULTI_ROW_MACRO_EMITTERS,
    MULTI_ROW_MACRO_HEAD_OPS,
)


def _call_with_int_sentinels(fn):
    """Call ``fn`` with each required positional/keyword arg set to 1."""
    sig = inspect.signature(fn)
    kwargs = {}
    for pname, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            continue
        kwargs[pname] = 1
    return fn(**kwargs)


class TestMultiRowMacroRegistry(unittest.TestCase):
    def test_every_registered_emitter_has_matching_head(self):
        """Each entry's ``(head_op, head_subreg)`` must match what the
        emitter actually produces.
        """
        for emitter_fn, expected_op, expected_subreg in MULTI_ROW_MACRO_EMITTERS:
            rows = _call_with_int_sentinels(emitter_fn)
            self.assertGreaterEqual(
                len(rows),
                2,
                f"{emitter_fn.__name__} returned {len(rows)} rows; not multi-row",
            )
            head = rows[0]
            head_op = int(head["op"])
            head_subreg = int(head["subreg"])
            self.assertEqual(
                head_op,
                int(expected_op),
                f"{emitter_fn.__name__} head op={head_op}; "
                f"registry says {expected_op}",
            )
            self.assertEqual(
                head_subreg,
                int(expected_subreg),
                f"{emitter_fn.__name__} head subreg={head_subreg}; "
                f"registry says {expected_subreg}",
            )

    def test_head_ops_constant_matches_emitters(self):
        """``MULTI_ROW_MACRO_HEAD_OPS`` is the (op, subreg) projection of
        ``MULTI_ROW_MACRO_EMITTERS``. Drift between the two means the
        tokenizer's isolation set may pick up entries the emitter side
        no longer produces, or vice versa.
        """
        derived = tuple((op, sr) for _fn, op, sr in MULTI_ROW_MACRO_EMITTERS)
        self.assertEqual(MULTI_ROW_MACRO_HEAD_OPS, derived)

    def test_no_undeclared_multi_row_emitter_in_loops(self):
        """Every ``*_rows`` function in ``preframr.macros.loops`` that
        emits 2+ rows MUST be in ``MULTI_ROW_MACRO_EMITTERS``. The
        check is intentionally introspective so a future macro added
        as ``_my_macro_rows`` without registry update fails the build
        gate, not at sample time.
        """
        declared_fns = {fn for fn, _, _ in MULTI_ROW_MACRO_EMITTERS}
        for name, fn in inspect.getmembers(loops_mod, inspect.isfunction):
            if not name.endswith("_rows"):
                continue
            if fn in declared_fns:
                continue
            try:
                rows = _call_with_int_sentinels(fn)
            except (TypeError, ValueError, AssertionError) as e:
                self.fail(
                    f"{name} matches the *_rows naming convention but "
                    f"isn't in MULTI_ROW_MACRO_EMITTERS and the property "
                    f"probe couldn't introspect it ({e}). Either declare "
                    f"it in the registry (preferred) or rename so the "
                    f"property test can ignore it."
                )
            if not isinstance(rows, list):
                continue
            if len(rows) >= 2:
                self.fail(
                    f"{name} emits {len(rows)} rows but is not in "
                    f"MULTI_ROW_MACRO_EMITTERS. Multi-row macros require "
                    f"head-row isolation in the tokenizer; declare it so "
                    f"the isolation set picks up the head op."
                )


if __name__ == "__main__":
    unittest.main()
