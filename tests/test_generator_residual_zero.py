"""GeneratorPass residual-zero self-audit (work order §3/§7): the unified generator model must leave
ZERO raw ``SET`` on any generator channel (freq x3 + pw/cut/res/modevol) through the full parse, every
length-1 atom must start at the final frame F-1, the parse must be byte-exact (the ``validate=True``
arbiter guard, hardened here with ``PREFRAMR_ARBITER_STRICT``), and permuting a freq-only frame's
waveform nibble must not change the emitted tokens (the Facemorph waveform-agnostic guardrail).
"""

import os
import tempfile
from collections import Counter

import pytest

from tests.parse_probes import DumpBuilder, write_dump, cents_to_fn
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    GEN_FREQ_REGS,
    GEN_SCALAR_REGS,
    GEN_TRI_OP,
    GEN_TRI_SUBREG_LEN,
    SET_OP,
    SWEEP_OP,
    SWEEP_SUBREG_LEN,
)
from preframr_tokens.tokenizer_config import default_tokenizer_args

_GEN_REGS = frozenset(int(r) for r in GEN_FREQ_REGS + GEN_SCALAR_REGS)


def _gen_args(**over):
    return default_tokenizer_args(generator_pass=True, instrument_program=True, **over)


def _parse(path, **over):
    return next(
        RegLogParser(args=_gen_args(**over)).parse(
            path, max_perm=1, require_pq=False, reparse=True
        ),
        None,
    )


def _multi_feature_dump(path):
    """A voice-0 tune exercising every generator: a period-3 arp (TABLE), a constant-delta downward
    slide (ACCUM), a bounded vibrato (TRIANGLE), held PW/modevol (HOLD), plus one-off global writes.
    """
    b = DumpBuilder().adsr().pw(0x800).modevol(0x1F).resfilt(0x00)
    b.note([cents_to_fn([60, 64, 67][i % 3], 0) for i in range(12)])
    b.note([cents_to_fn(60, 0) - 200 * i for i in range(8)])
    base = cents_to_fn(60, 0)
    b.resfilt(0xF1)
    b.note(
        ([base, base + 40, base + 80, base + 40, base, base - 40, base - 80, base - 40])
        * 2
    )
    b.note([cents_to_fn(48, 0)] * 5)
    return write_dump(b, path)


def _raw_gen_sets(df):
    op = df["op"].to_numpy()
    reg = df["reg"].to_numpy()
    return Counter(
        int(reg[i])
        for i in range(len(df))
        if int(op[i]) == SET_OP and int(reg[i]) in _GEN_REGS
    )


def _real_frame_of_rows(df):
    """Decoded real-frame index per row (FRAME=+1, DELAY=+val), and the total frame count."""
    reg = df["reg"].to_numpy()
    val = df["val"].to_numpy()
    out = []
    f = -1
    for i in range(len(df)):
        r = int(reg[i])
        if r == FRAME_REG:
            f += 1
        elif r == DELAY_REG:
            f += int(val[i])
        out.append(f)
    return out, f + 1


def test_generator_residual_zero_multi_feature():
    """ZERO raw SET on any generator channel through the full deployed parse."""
    with tempfile.TemporaryDirectory() as tmp:
        df = _parse(_multi_feature_dump(os.path.join(tmp, "mf.dump.parquet")))
    assert df is not None
    offenders = _raw_gen_sets(df)
    assert (
        not offenders
    ), f"un-modelled generator-channel raw SET (reg->count): {dict(offenders)}"


def test_generator_strict_byte_exact():
    """Under PREFRAMR_ARBITER_STRICT every generator claim must be byte-exact, else the arbiter RAISES."""
    prev = os.environ.get("PREFRAMR_ARBITER_STRICT")
    os.environ["PREFRAMR_ARBITER_STRICT"] = "1"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            df = _parse(_multi_feature_dump(os.path.join(tmp, "mf.dump.parquet")))
        assert df is not None
        assert not _raw_gen_sets(df)
    finally:
        if prev is None:
            del os.environ["PREFRAMR_ARBITER_STRICT"]
        else:
            os.environ["PREFRAMR_ARBITER_STRICT"] = prev


def test_generator_table_resid_split_byte_exact():
    """``table_resid_split`` keys a freq GEN_TABLE on the OFFSET cycle alone and carries the per-frame
    residual on the per-instance REF (the §3 de-fragmentation): same note-shape, different residual,
    collapses to one DEF. The parse stays byte-exact (no raw gen SET under the strict arbiter) and the
    residual STEP atoms move off the DEF (op GEN_TABLE_STEP) onto the REF (op GEN_TABLE_REF).
    """
    from preframr_tokens.stfconstants import (
        GEN_TABLE_REF_OP as _REF,
        GEN_TABLE_REF_SUBREG_RESID_HI as _RREF_HI,
        GEN_TABLE_REF_SUBREG_RESID_LO as _RREF_LO,
        GEN_TABLE_STEP_OP as _STEP,
        GEN_TABLE_SUBREG_RESID_HI as _RDEF_HI,
        GEN_TABLE_SUBREG_RESID_LO as _RDEF_LO,
    )

    prev = os.environ.get("PREFRAMR_ARBITER_STRICT")
    os.environ["PREFRAMR_ARBITER_STRICT"] = "1"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = _multi_feature_dump(os.path.join(tmp, "mf.dump.parquet"))
            base = _parse(path)
            split = _parse(path, table_resid_split=True)
    finally:
        if prev is None:
            del os.environ["PREFRAMR_ARBITER_STRICT"]
        else:
            os.environ["PREFRAMR_ARBITER_STRICT"] = prev
    assert base is not None and split is not None
    assert not _raw_gen_sets(
        split
    ), "table_resid_split left un-modelled generator-channel raw SET"

    def _resid(df, op, los, his):
        return sum(
            1
            for o, s in zip(df["op"], df["subreg"])
            if int(o) == op and int(s) in (los, his)
        )

    assert (
        _resid(base, _STEP, _RDEF_LO, _RDEF_HI) > 0
    ), "expected DEF-keyed residuals in the baseline"
    assert (
        _resid(split, _STEP, _RDEF_LO, _RDEF_HI) == 0
    ), "split must drop residuals off the DEF key"
    assert (
        _resid(split, _REF, _RREF_LO, _RREF_HI) > 0
    ), "split must carry residuals on the per-instance REF"


def test_generator_table_resid_split_universal_byte_exact():
    """``table_resid_split`` + ``universal_pitch`` (design item #4): a freq GEN_TABLE keys its OFFSET
    cycle off the PER-VOICE ``note_index`` (DEF mode NOTE_UNIV) and carries the residual on the REF, so
    static-note residuals go ~0. Byte-exact under the strict arbiter and a NOTE_UNIV DEF is emitted.
    """
    from preframr_tokens.stfconstants import (
        GEN_TABLE_MODE_NOTE_UNIV as _UNIV,
        GEN_TABLE_STEP_OP as _STEP,
        GEN_TABLE_SUBREG_MODE as _MODE,
    )

    prev = os.environ.get("PREFRAMR_ARBITER_STRICT")
    os.environ["PREFRAMR_ARBITER_STRICT"] = "1"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = _multi_feature_dump(os.path.join(tmp, "mf.dump.parquet"))
            item4 = _parse(
                path, melody_skeleton=True, universal_pitch=True, table_resid_split=True
            )
    finally:
        if prev is None:
            del os.environ["PREFRAMR_ARBITER_STRICT"]
        else:
            os.environ["PREFRAMR_ARBITER_STRICT"] = prev
    assert item4 is not None
    assert not _raw_gen_sets(
        item4
    ), "item #4 left un-modelled generator-channel raw SET"
    n_univ = sum(
        1
        for o, s, v in zip(item4["op"], item4["subreg"], item4["val"])
        if int(o) == _STEP and int(s) == _MODE and int(v) == _UNIV
    )
    assert n_univ > 0, "expected a per-voice NOTE_UNIV GEN_TABLE DEF"


def test_generator_length1_only_at_final_frame():
    """A length-1 generator atom (a degenerate SWEEP/TRI LEN=1) may occur only on the final frame F-1 --
    no interior length-1, the work order's entire 'tail' story (no GEN_END op)."""
    with tempfile.TemporaryDirectory() as tmp:
        df = _parse(_multi_feature_dump(os.path.join(tmp, "mf.dump.parquet")))
    assert df is not None
    op = df["op"].to_numpy()
    sub = df["subreg"].to_numpy()
    val = df["val"].to_numpy()
    rows_frame, n_frames = _real_frame_of_rows(df)
    for i in range(len(df)):
        is_len1 = (
            int(op[i]) == SWEEP_OP
            and int(sub[i]) == SWEEP_SUBREG_LEN
            and int(val[i]) == 1
        ) or (
            int(op[i]) == GEN_TRI_OP
            and int(sub[i]) == GEN_TRI_SUBREG_LEN
            and int(val[i]) == 1
        )
        if is_len1:
            assert (
                rows_frame[i] >= n_frames - 1
            ), f"interior length-1 atom at frame {rows_frame[i]} of {n_frames}"


def test_generator_waveform_agnostic_property():
    """Facemorph guardrail (work order §7.2 / decision #2): permuting the waveform nibble of a freq-only
    frame must NOT change the emitted generator tokens -- nothing reads the waveform bit to route pitch.
    """
    base = cents_to_fn(60, 0)

    def build(waveform):
        b = DumpBuilder().adsr().pw(0x800).modevol(0x1F)
        b.note([cents_to_fn([60, 64, 67][i % 3], 0) for i in range(9)])
        b.frame().ctrl(waveform).freq(base)
        b.note([base] * 5)
        return b

    gen_ops = {SWEEP_OP, GEN_TRI_OP, 84, 85, 86, 87, 88}

    def toks(waveform):
        with tempfile.TemporaryDirectory() as tmp:
            df = _parse(
                write_dump(build(waveform), os.path.join(tmp, "wf.dump.parquet"))
            )
        gen = df[df["op"].isin(list(gen_ops))]
        return [
            (int(r), int(o), int(s), int(v))
            for r, o, s, v in zip(gen["reg"], gen["op"], gen["subreg"], gen["val"])
        ]

    pulse = toks(0x41)
    noise = toks(0x81)
    assert (
        pulse == noise
    ), "freq generator tokens changed when only the waveform nibble was permuted"


@pytest.mark.parametrize(
    "builder",
    [
        lambda b: b.note([cents_to_fn([60, 63, 67][i % 3], 0) for i in range(15)]),
        lambda b: b.note([cents_to_fn(72, 0) - 137 * i for i in range(10)]),
        lambda b: b.note([cents_to_fn(48, 0)] * 20),
    ],
)
def test_generator_residual_zero_singletons(builder):
    """Each isolated generator gesture (arp / slide / long hold) reaches generator-channel raw-SET zero."""
    b = DumpBuilder().adsr().pw(0x444).modevol(0x0F)
    builder(b)
    with tempfile.TemporaryDirectory() as tmp:
        df = _parse(write_dump(b, os.path.join(tmp, "s.dump.parquet")))
    assert df is not None
    assert not _raw_gen_sets(df)


def test_generator_scalar_channel_oscillation_abs_table():
    """A per-frame filter-cutoff oscillation (period-2 cycle on a scalar channel) is interned as an
    absolute-keyed GEN_TABLE -- the non-freq codebook path, byte-exact under strict arbiter.
    """
    prev = os.environ.get("PREFRAMR_ARBITER_STRICT")
    os.environ["PREFRAMR_ARBITER_STRICT"] = "1"
    try:
        b = DumpBuilder().adsr().pw(0x300).modevol(0x1F)
        cyc = [0x200, 0x600, 0x400]
        for i in range(18):
            b.frame().fc(cyc[i % 3])
            if i == 0:
                b.gate_on()
            b.freq(cents_to_fn(57, 0))
        df = None
        with tempfile.TemporaryDirectory() as tmp:
            df = _parse(write_dump(b, os.path.join(tmp, "osc.dump.parquet")))
        assert df is not None
        assert not _raw_gen_sets(df)
        ops = set(int(o) for o in df["op"].to_numpy())
        assert (
            85 in ops and 88 in ops
        ), f"expected a GEN_TABLE DEF+REF, got ops {sorted(ops)}"
    finally:
        if prev is None:
            del os.environ["PREFRAMR_ARBITER_STRICT"]
        else:
            os.environ["PREFRAMR_ARBITER_STRICT"] = prev


def test_generator_zeroes_whole_chip_singleton_regs():
    """The irregular Res/Filt(23) + Mode/Vol(24) singletons the deployed default leaves as raw SET
    (test_whole_chip_no_singleton_set, the deferred whole-chip tail) ARE modelled to zero by the
    generator -- a one-off global write is an ordinary HOLD generator, not an un-modelled straggler.
    Runs the whole-chip gate's own tune through the generator config (real parse, not a post-split df).
    """
    from preframr_tokens.macros.freq_lut import LUT

    prev = os.environ.get("PREFRAMR_ARBITER_STRICT")
    os.environ["PREFRAMR_ARBITER_STRICT"] = "1"
    try:
        b = DumpBuilder().adsr().pw(0x800).modevol(0x0F).resfilt(0x00)
        b.note([LUT[60]] * 5)
        b.note([LUT[55]] * 4)
        b.modevol(0x1F)
        b.note([LUT[57]] * 4)
        b.resfilt(0xF1)
        b.note([LUT[52]] * 4)
        b.note([LUT[48]] * 5)
        with tempfile.TemporaryDirectory() as tmp:
            df = _parse(write_dump(b, os.path.join(tmp, "wc.dump.parquet")))
        assert df is not None
        offenders = _raw_gen_sets(df)
        assert (
            not offenders
        ), f"generator left raw SET on (reg->count): {dict(offenders)}"
    finally:
        if prev is None:
            del os.environ["PREFRAMR_ARBITER_STRICT"]
        else:
            os.environ["PREFRAMR_ARBITER_STRICT"] = prev


def test_generator_long_hold_survives_consolidation():
    """A long HOLD atom (a channel held constant while another changes every frame) stays byte-exact
    through frame consolidation + the now-lossless _cap_delay -- no MAX_SPAN chunking needed (cap_delay
    is chain_delay, frame-preserving, not the old truncate-to-255 that dropped long atoms).
    """
    prev = os.environ.get("PREFRAMR_ARBITER_STRICT")
    os.environ["PREFRAMR_ARBITER_STRICT"] = "1"
    try:
        b = DumpBuilder().adsr().pw(0x800).modevol(0x1F)
        b.note([cents_to_fn(40, 0) + 7 * i for i in range(200)])
        with tempfile.TemporaryDirectory() as tmp:
            df = _parse(write_dump(b, os.path.join(tmp, "long.dump.parquet")))
        assert df is not None
        assert not _raw_gen_sets(df)
        lens = [
            int(v)
            for o, s, v in zip(df["op"], df["subreg"], df["val"])
            if int(o) == SWEEP_OP and int(s) == SWEEP_SUBREG_LEN
        ]
        assert (
            max(lens) > 100
        ), f"expected a long (>100) HOLD atom, got max LEN {max(lens or [0])}"
    finally:
        if prev is None:
            del os.environ["PREFRAMR_ARBITER_STRICT"]
        else:
            os.environ["PREFRAMR_ARBITER_STRICT"] = prev
