"""FreqOnsetPass: re-tag residual op0 SET on TRAJ_REGS to FREQ_ONSET (op48), byte-exact
decode (SET-equivalent), invariant (no op0 SET remains on TRAJ_REGS), default off = noop.
"""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.freq_onset_pass import FreqOnsetPass
from preframr_tokens.stfconstants import (
    FRAME_REG,
    FREQ_ONSET_OP,
    SET_OP,
    TRAJ_REGS,
)


class FakeArgs:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


def _r(reg, val, op=SET_OP, subreg=-1):
    return {
        "reg": reg,
        "val": val,
        "op": op,
        "subreg": subreg,
        "diff": 32,
        "description": 0,
    }


def _stream(rows):
    return pd.DataFrame(rows)


def _per_frame(df, reg):
    dec = expand_ops(df, strict=False).reset_index(drop=True)
    cur = 0
    out = []
    for _, row in dec.iterrows():
        if int(row["reg"]) == FRAME_REG:
            out.append(cur)
        elif int(row["reg"]) == reg:
            cur = int(row["val"])
    return out


class TestFreqOnsetPass(unittest.TestCase):
    def _stream_with_isolated_traj(self):
        return _stream(
            [
                _r(FRAME_REG, 0),
                _r(0, 100),
                _r(FRAME_REG, 0),
                _r(2, 50),
                _r(FRAME_REG, 0),
                _r(21, 200),
                _r(FRAME_REG, 0),
                _r(5, 0xAA),
                _r(FRAME_REG, 0),
                _r(23, 0x0F),
                _r(FRAME_REG, 0),
            ]
        )

    def test_default_off_is_noop(self):
        df = self._stream_with_isolated_traj()
        out = FreqOnsetPass().apply(df.copy(), args=FakeArgs())
        self.assertTrue(out.equals(df))

    def test_retags_only_traj_reg_set(self):
        df = self._stream_with_isolated_traj()
        out = FreqOnsetPass().apply(df.copy(), args=FakeArgs(freq_onset_pass=True))
        traj_set_mask = (
            (out["op"] == SET_OP) & (out["reg"].isin(TRAJ_REGS)) & (out["subreg"] == -1)
        )
        self.assertEqual(int(traj_set_mask.sum()), 0)
        retagged = out[out["op"] == FREQ_ONSET_OP]
        self.assertEqual(sorted(retagged["reg"].tolist()), [0, 2, 21])
        adsr_set = out[(out["op"] == SET_OP) & (out["reg"] == 5)]
        ctrl_set = out[(out["op"] == SET_OP) & (out["reg"] == 23)]
        self.assertEqual(len(adsr_set), 1)
        self.assertEqual(len(ctrl_set), 1)

    def test_roundtrip_byte_exact(self):
        df = self._stream_with_isolated_traj()
        out = FreqOnsetPass().apply(df.copy(), args=FakeArgs(freq_onset_pass=True))
        for reg in (0, 2, 21, 5, 23):
            self.assertEqual(_per_frame(df, reg), _per_frame(out, reg), f"reg {reg}")


if __name__ == "__main__":
    unittest.main()
