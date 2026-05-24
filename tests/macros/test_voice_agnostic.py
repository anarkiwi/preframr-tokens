"""Enforce that FREQ macros carry no out-of-band voice information: the same
content on a different voice must yield byte-identical atom payloads (op, subreg,
val), differing only in the register (by VOICE_REG_SIZE). A macro that baked a
voice index into a value/op would fail here, catching the class of bug where an
atom's reconstruction stops matching the voice its register implies."""

import unittest

import pandas as pd

from preframr_tokens.macros.freq_trajectory_pass import FreqTrajectoryPass
from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import FRAME_REG, SET_OP, VOICE_REG_SIZE


class FakeArgs:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


def _frame():
    return {
        "reg": FRAME_REG,
        "val": 0,
        "diff": 19000,
        "op": SET_OP,
        "subreg": -1,
        "description": 0,
    }


def _row(reg, val):
    return {
        "reg": reg,
        "val": val,
        "diff": 32,
        "op": SET_OP,
        "subreg": -1,
        "description": 0,
    }


def _emit(reg, values):
    rows = [_frame(), _row(reg, values[0])]
    for v in values[1:]:
        rows += [_frame(), _row(reg, v)]
    rows += [_frame()]
    out = FreqTrajectoryPass().apply(
        pd.DataFrame(rows), args=FakeArgs(freq_trajectory_pass=True)
    )
    return out[out["op"] != SET_OP].reset_index(drop=True)


_CASES = (
    ("oscillate", [120, 122, 120, 122, 120, 122]),
    ("run", [100, 101, 103, 107, 111]),
)


class TestVoiceAgnostic(unittest.TestCase):
    def test_atom_payload_is_voice_invariant(self):
        for label, vals in _CASES:
            with self.subTest(label):
                m0 = _emit(FREQ_REGS_BY_VOICE[0], vals)
                m1 = _emit(FREQ_REGS_BY_VOICE[1], vals)
                self.assertGreater(len(m0), 0, "macro did not fire")
                self.assertEqual(len(m0), len(m1))
                p0 = m0[["op", "subreg", "val"]]
                p1 = m1[["op", "subreg", "val"]]
                self.assertTrue(
                    p0.equals(p1),
                    f"{label} payload differs across voices (leaks voice)",
                )
                offsets = m1["reg"].to_numpy() - m0["reg"].to_numpy()
                self.assertTrue(
                    (offsets == VOICE_REG_SIZE).all(),
                    f"{label} register offset is not the voice stride",
                )


if __name__ == "__main__":
    unittest.main()
