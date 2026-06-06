"""Tests for the pass-by-pass pipeline tracer: spec/flag resolution, the diff
helpers, the apply-wrapping instrumentation (restored cleanly on exit), and an
end-to-end run + flag isolation over a hermetic synthetic dump."""

import json
import tempfile
import unittest

import pandas as pd

from preframr_tokens import pipeline_trace as pt
from preframr_tokens.macros.preset_pass import PresetPass

_IRQ_PERIOD = 19656


def _write_synthetic_dump(path):
    """A minimal register dump that survives the real parse: the ``irq`` column
    steps one frame-period per frame (so ``_add_frame_reg`` finds frames) and the
    per-frame values change (so squeeze keeps them and the macros have material)."""
    rows = []
    for f in range(40):
        irq = f * _IRQ_PERIOD
        writes = [
            (0, (f * 7) % 256),
            (1, (f // 4) % 8),
            (2, (f * 5) % 256),
            (3, (f // 3) % 16),
            (4, 0x11 if f % 2 else 0x10),
            (5, (0xA0 + f) % 256),
            (6, 0xF0),
            (21, (f * 9) % 256),
            (22, (f // 2) % 8),
            (24, 15),
        ]
        for i, (reg, val) in enumerate(writes):
            rows.append(
                {"clock": irq + i * 10, "irq": irq, "reg": reg, "val": val, "chipno": 0}
            )
    pd.DataFrame(rows).to_parquet(path)
    return path


_FULL_MACROS_SPEC = {
    "transforms": [
        {"name": "freq_trajectory"},
        {"name": "preset"},
        {"name": "hard_restart"},
        {"name": "legato_per_cluster", "params": {"clusters": [2, 4]}},
        {"name": "voice_block_order"},
        {"name": "loop"},
    ]
}
_ABSORBERS = [
    "--preset-pass",
    "--hard-restart-pass",
]


class TestBuildArgs(unittest.TestCase):
    def test_spec_names_map_to_flags(self):
        args, resolution, unknown_names, unknown_flags = pt.build_args(
            _FULL_MACROS_SPEC, _ABSORBERS, {}
        )
        self.assertTrue(args.freq_trajectory_pass)
        self.assertTrue(args.preset_pass)
        self.assertTrue(args.voice_canonical_block_order)
        self.assertEqual(unknown_names, [])
        self.assertEqual(unknown_flags, [])
        self.assertIn(("freq_trajectory", "freq_trajectory_pass", True), resolution)

    def test_legato_clusters_expand(self):
        args, _, _, _ = pt.build_args(_FULL_MACROS_SPEC, [], {})
        self.assertTrue(args.legato_pass_c2)
        self.assertTrue(args.legato_pass_c4)
        self.assertFalse(args.legato_pass_c7)

    def test_unknown_spec_name_is_flagged_not_silent(self):
        _, _, unknown_names, _ = pt.build_args(
            {"transforms": [{"name": "slope"}]}, [], {}
        )
        self.assertEqual(unknown_names, ["slope"])

    def test_unknown_flag_is_flagged(self):
        _, _, _, unknown_flags = pt.build_args(None, ["--not-a-real-pass"], {})
        self.assertIn("not_a_real_pass", unknown_flags)

    def test_no_macro_flag_defaults_true(self):
        args, _, _, _ = pt.build_args(None, [], {})
        self.assertFalse(any(getattr(args, f) for f in pt.macro_flag_names()))

    def test_cargs_negation(self):
        args, _, _, _ = pt.build_args(None, ["--no-preset-pass"], {})
        self.assertFalse(args.preset_pass)

    def test_overrides_applied(self):
        args, _, _, _ = pt.build_args(None, [], {"min_song_tokens": 7})
        self.assertEqual(args.min_song_tokens, 7)


class TestHelpers(unittest.TestCase):
    def test_op_delta(self):
        self.assertEqual(
            pt._op_delta({"SET": 10, "DIFF": 2}, {"SET": 4, "FREQ_TRAJ": 3}),
            {"SET": -6, "DIFF": -2, "FREQ_TRAJ": 3},
        )

    def test_opname_and_reglabel(self):
        self.assertEqual(pt._opname(45), "FREQ_TRAJ")
        self.assertEqual(pt._opname(0), "SET")
        self.assertEqual(pt._reglabel(2), "v0.PW")
        self.assertEqual(pt._reglabel(21), "FC")
        self.assertEqual(pt._reglabel(24), "MODE_VOL")

    def test_ophist(self):
        df = pd.DataFrame([{"op": 0}, {"op": 0}, {"op": 45}])
        self.assertEqual(pt._ophist(df), {"SET": 2, "FREQ_TRAJ": 1})

    def test_mark_branches_flags_discarded_output(self):
        records = [
            {"rows_before": 10, "rows_after": 10, "branch": False},
            {"rows_before": 10, "rows_after": 20, "branch": False},
            {"rows_before": 10, "rows_after": 12, "branch": False},
        ]
        pt._mark_branches(records)
        self.assertFalse(records[0]["branch"])
        self.assertTrue(records[1]["branch"])
        self.assertFalse(records[2]["branch"])

    def test_flag_report(self):
        args, _, _, _ = pt.build_args(None, ["--preset-pass"], {})
        records = [
            {
                "stage": "PresetPass",
                "gate_flags": {"preset_pass": True},
                "status": "FIRED",
            },
            {"stage": "Other", "gate_flags": {}, "status": "FIRED"},
        ]
        rep = pt.flag_report(records, args)
        self.assertTrue(rep["preset_pass"]["effective"])
        self.assertEqual(rep["preset_pass"]["fired_in"], ["PresetPass"])

    def test_decode_rows(self):
        df = pd.DataFrame([{"reg": 21, "op": 36, "val": 256, "subreg": -1}])
        rows = pt._decode_rows(df, 10)
        self.assertEqual(rows[0]["reg_label"], "FC")
        self.assertEqual(rows[0]["op"], "FC_PRESET")

    def test_slice_dump_head(self):
        df = pd.DataFrame(
            {"clock": range(10), "irq": 1, "reg": 0, "val": 0, "chipno": 0}
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet") as src:
            df.to_parquet(src.name)
            out = pt._slice_dump(src.name, 3, 0, "")
            self.assertEqual(len(pd.read_parquet(out)), 3)


class TestTracerInstrumentation(unittest.TestCase):
    def test_wraps_and_restores(self):
        args, _, _, _ = pt.build_args({"transforms": [{"name": "preset"}]}, [], {})
        pw_set = pd.DataFrame(
            [
                {
                    "reg": 2,
                    "val": 130,
                    "op": 0,
                    "subreg": -1,
                    "diff": 0,
                    "irq": 20000,
                    "description": 0,
                }
            ]
        )
        before = PresetPass.__dict__["apply"]
        with pt.Tracer(args) as tracer:
            self.assertIsNot(PresetPass.__dict__["apply"], before)
            PresetPass().apply(pw_set, args=args)
        self.assertIs(PresetPass.__dict__["apply"], before)
        rec = tracer.records[0]
        self.assertEqual(rec["stage"], "PresetPass")
        self.assertEqual(rec["status"], "FIRED")
        self.assertEqual(rec["op_delta"], {"SET": -1, "PWM_PRESET": 1})

    def test_gate_off_pass_records_skip(self):
        args, _, _, _ = pt.build_args(None, [], {})
        fc_set = pd.DataFrame(
            [
                {
                    "reg": 21,
                    "val": 100,
                    "op": 0,
                    "subreg": -1,
                    "diff": 0,
                    "irq": 20000,
                    "description": 0,
                }
            ]
        )
        with pt.Tracer(args) as tracer:
            PresetPass().apply(fc_set, args=args)
        self.assertEqual(tracer.records[0]["status"], "skip(off)")


class TestRenderText(unittest.TestCase):
    def test_render_smoke(self):
        report = {
            "dump": "x.parquet",
            "max_perm": 1,
            "resolution": [("preset", "preset_pass", True)],
            "unknown_names": ["slope"],
            "unknown_flags": [],
            "active_flags": {"preset_pass": True},
            "records": [
                {
                    "idx": 0,
                    "stage": "PresetPass",
                    "gate_flags": {"preset_pass": True},
                    "rows_after": 5,
                    "delta": -1,
                    "op_delta": {"PWM_PRESET": 1, "SET": -1},
                    "status": "FIRED",
                    "branch": False,
                }
            ],
            "flag_report": {
                "preset_pass": {
                    "read_by": ["PresetPass"],
                    "fired_in": ["PresetPass"],
                    "effective": True,
                }
            },
            "final_rows": 5,
            "final_op_hist": {"PWM_PRESET": 1},
            "final_decoded": [],
            "isolation": None,
            "isolate_flag": "",
        }
        text = pt.render_text(report, full=False, show_rows=80)
        self.assertIn("UNRECOGNIZED spec name 'slope'", text)
        self.assertIn("PresetPass", text)
        self.assertIn("EFFECTIVE", text)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.NamedTemporaryFile(suffix=".dump.parquet", delete=False)
        cls.dump = _write_synthetic_dump(cls._tmp.name)
        cls.overrides = {"min_irq": 0, "max_irq": 10**9, "min_song_tokens": 1}

    def test_trace_fires_expected_passes(self):
        args, _, _, _ = pt.build_args(_FULL_MACROS_SPEC, _ABSORBERS, self.overrides)
        records, final = pt.run_trace(self.dump, args, 1)
        self.assertIsNotNone(final)
        by_stage = {r["stage"]: r for r in records}
        self.assertEqual(by_stage["FreqTrajectoryPass"]["status"], "FIRED")
        self.assertIn("FREQ_TRAJ", by_stage["FreqTrajectoryPass"]["op_delta"])
        self.assertEqual(by_stage["combine_regs"]["kind"], "parser")

    def test_isolate_localizes_freq_trajectory(self):
        args, _, _, _ = pt.build_args(_FULL_MACROS_SPEC, _ABSORBERS, self.overrides)
        iso = pt.isolate(self.dump, args, "freq_trajectory_pass", 1)
        sites = {d["stage"] for d in iso["sites"]}
        self.assertIn("FreqTrajectoryPass", sites)
        self.assertIn("FREQ_TRAJ", iso["net"]["op"])

    def test_main_json_runs(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as spec_f:
            json.dump(_FULL_MACROS_SPEC, spec_f)
            spec_f.flush()
            rc = pt.main(
                [
                    self.dump,
                    "--pipeline-spec",
                    f"@{spec_f.name}",
                    "--cargs",
                    " ".join(_ABSORBERS),
                    "--min-song-tokens",
                    "1",
                    "--min-irq",
                    "0",
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(rc, 0)

    def test_main_text_isolate_full(self):
        rc = pt.main(
            [
                self.dump,
                "--pipeline-spec",
                json.dumps(_FULL_MACROS_SPEC),
                "--cargs",
                " ".join(_ABSORBERS),
                "--min-song-tokens",
                "1",
                "--min-irq",
                "0",
                "--isolate",
                "freq_trajectory_pass",
                "--full",
                "--head",
                "300",
            ]
        )
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
