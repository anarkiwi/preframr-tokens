"""Direct unit tests for ``transform_registry._normalize_spec`` and ``register``."""

import json
import unittest

from preframr_tokens.macros.transform_registry import (
    PipelineEntry,
    _REGISTRY,
    _normalize_spec,
    register,
)


class TestNormalizeSpec(unittest.TestCase):
    def test_json_string_input(self):
        out = _normalize_spec(json.dumps({"transforms": ["slope", "preset"]}))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].name, "slope")
        self.assertEqual(out[1].name, "preset")

    def test_dict_with_transforms_key(self):
        out = _normalize_spec({"transforms": ["slope"]})
        self.assertEqual([e.name for e in out], ["slope"])

    def test_dict_missing_transforms_key_raises(self):
        with self.assertRaises(ValueError):
            _normalize_spec({"other_key": []})

    def test_bare_list_accepted(self):
        out = _normalize_spec(["slope", "preset"])
        self.assertEqual(len(out), 2)

    def test_unsupported_spec_type_raises(self):
        with self.assertRaises(TypeError):
            _normalize_spec(42)

    def test_str_entries_become_pipeline_entries(self):
        out = _normalize_spec(["slope"])
        self.assertIsInstance(out[0], PipelineEntry)
        self.assertEqual(out[0].name, "slope")
        self.assertEqual(out[0].params, {})

    def test_dict_entries_carry_params(self):
        out = _normalize_spec([{"name": "loop", "params": {"lookahead": 5}}])
        self.assertEqual(out[0].name, "loop")
        self.assertEqual(out[0].params, {"lookahead": 5})

    def test_dict_entry_missing_name_raises(self):
        with self.assertRaises(ValueError):
            _normalize_spec([{"params": {}}])

    def test_pipeline_entry_passthrough(self):
        entry = PipelineEntry(name="slope", params={"foo": 1})
        out = _normalize_spec([entry])
        self.assertEqual(out, [entry])

    def test_unsupported_entry_type_raises(self):
        with self.assertRaises(TypeError):
            _normalize_spec([42])


class TestRegisterCollision(unittest.TestCase):
    def test_duplicate_registration_raises(self):
        @register("__lint_test_dummy_xyz__")
        class _Dummy:
            pass

        with self.assertRaises(ValueError):

            @register("__lint_test_dummy_xyz__")
            class _Dummy2:
                pass

        _REGISTRY.pop("__lint_test_dummy_xyz__", None)


if __name__ == "__main__":
    unittest.main()
