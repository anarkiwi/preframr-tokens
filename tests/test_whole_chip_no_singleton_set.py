"""Whole-chip no-singleton gate: the deployed default (``full_macros``) must leave NO raw ``SET`` on
any NON-FREQ register through the full ``RegLogParser.parse``. A SID driver is a deterministic program,
so every Mode/Vol or Res/Filt write it makes -- even one that occurs only ONCE in a tune -- is a driver
mechanism, not content; a surviving raw ``SET`` is an un-modelled straggler ("singleton"), which this
gate forbids. FREQ (regs 0/1/7/8/14/15) is the intentional pitch slate and is excluded.
"""

import os
import tempfile
from collections import Counter

from tests.parse_probes import DumpBuilder, write_dump
from preframr_tokens.macros.skeleton_pass import LUT
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import SET_OP
from preframr_tokens.tokenizer_config import named_config

_FREQ_REGS = frozenset((0, 1, 7, 8, 14, 15))


def _build_dump(path):
    """A voice-0 tune carrying genuine ONE-OFF global writes: a master-volume set (reg 24) held for the
    whole song, a single mid-song filter-mode change, and a one-off resonance/routing set (reg 23) -- the
    step_hold/irregular singletons that survive as raw SET today."""
    b = DumpBuilder().adsr().pw(0x800)
    b.modevol(0x0F).resfilt(0x00)
    b.note([LUT[60]] * 5)
    b.note([LUT[55]] * 4)
    b.modevol(0x1F)
    b.note([LUT[57]] * 4)
    b.resfilt(0xF1)
    b.note([LUT[52]] * 4)
    b.note([LUT[48]] * 5)
    return write_dump(b, path)


def _parse(path):
    return next(
        RegLogParser(args=named_config("full_macros", seq_len=4096)).parse(
            path, max_perm=1, require_pq=False, reparse=True
        ),
        None,
    )


def _nonfreq_set_by_reg(df):
    op = df["op"].to_numpy()
    reg = df["reg"].to_numpy()
    out = Counter()
    for i in range(len(df)):
        if int(op[i]) == SET_OP:
            r = int(reg[i])
            if 0 <= r < 25 and r not in _FREQ_REGS:
                out[r] += 1
    return out


def test_no_nonfreq_raw_set_singletons():
    """Every non-FREQ chip write is modelled: the deployed token stream has zero raw SET on regs 2-6,
    9-13, 16-22 and on Res/Filt(23)+Mode/Vol(24); any survivor names the register it leaked on.
    """
    with tempfile.TemporaryDirectory() as tmp:
        df = _parse(_build_dump(os.path.join(tmp, "whole_chip.dump.parquet")))
    assert df is not None
    offenders = _nonfreq_set_by_reg(df)
    assert (
        not offenders
    ), f"un-modelled non-FREQ raw SET (reg->count): {dict(sorted(offenders.items()))}"
