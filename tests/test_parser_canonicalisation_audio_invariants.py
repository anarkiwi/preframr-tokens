"""Pin parser canonicalisations that the 2026-05-16 investigation
found to be audio-neutral. Regression-guards against any future
parser change that re-introduces audible loss in these axes.
"""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

pyresidfp = pytest.importorskip("pyresidfp")
SoundInterfaceDevice = pyresidfp.SoundInterfaceDevice
ChipModel = pyresidfp.sound_interface_device.ChipModel


SYNTHETIC_AUDIO_NEUTRAL_BOUND = 200
REAL_FIXTURE_PARSE_BOUND = 0.02


def _make_parser(cents: int = 50):
    """Build a RegLogParser with the args fields it actually reads. Detached from preframr.args.add_args so this test lives torch-free in preframr-tokens. Defaults mirror preframr.args (min_irq=15000, max_irq=25000)."""
    from preframr_tokens.reglogparser import RegLogParser

    args = argparse.Namespace(
        cents=cents,
        min_irq=15000,
        max_irq=25000,
        min_song_tokens=64,
    )
    return RegLogParser(args=args, logger=logging.getLogger("test_parser_audio"))


def _fresh_sid():
    sid = SoundInterfaceDevice(model=ChipModel.MOS8580)
    sid.reset()
    for r in range(25):
        sid.write_register(r, 0)
    _ = sid.clock(timedelta(seconds=0.1))
    return sid


def _render_dump(sid, df):
    """Render a parser-input-shape df (columns: clock, reg, val) on the SID.
    Returns int16 mono samples."""
    clock_hz = int(sid.clock_frequency)
    out = []
    last_clock = 0
    clocks = df["clock"].astype("int64").to_numpy()
    regs = df["reg"].astype("int64").to_numpy()
    vals = df["val"].astype("int64").to_numpy()
    for i in range(len(df)):
        c = int(clocks[i])
        d = c - last_clock
        if d > 0:
            chunk = sid.clock(timedelta(seconds=d / clock_hz))
            if chunk:
                out.append(np.asarray(chunk, dtype=np.int16))
        sid.write_register(int(regs[i]), int(vals[i]))
        last_clock = c
    if not out:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(out)


def _wave_max_diff(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0
    aa = a[:n].astype(np.int32)
    bb = b[:n].astype(np.int32)
    return int(np.abs(aa - bb).max())


def _build_input_df(rows):
    """Build a parser-input df from (clock, reg, val) tuples."""
    from preframr_tokens.stfconstants import VAL_PDTYPE

    df = pd.DataFrame(rows, columns=["clock", "reg", "val"])
    df["irq"] = 19656
    df = df[["clock", "irq", "reg", "val"]]
    df["clock"] = df["clock"].astype("UInt32")
    df["irq"] = df["irq"].astype("UInt32")
    df["reg"] = df["reg"].astype("UInt8")
    df["val"] = df["val"].astype(VAL_PDTYPE)
    return df


def _make_noise_with_pw_changes_df():
    """Voice 0 plays noise; several PW_LO writes happen during the
    noise phase (CTRL bit 6 == 0, no pulse). PW writes are silent on
    real SID per the chip spec; ``_simplify_pcm`` canonicalises that
    by zeroing them.
    """
    return _build_input_df(
        [
            (0, 5, 0x09),
            (10, 6, 0xF0),
            (20, 24, 0x0F),
            (30, 1, 0x08),
            (40, 0, 0x80),
            (50, 4, 0x81),
            (1000, 2, 0x55),
            (5000, 2, 0xAA),
            (10000, 2, 0xFF),
        ]
    )


def test_simplify_pcm_structurally_zeroes_pw_during_pulse_off():
    """Pins the canonicalisation that ``_simplify_pcm`` claims to do
    (reglogparser.py:548 docstring transformation 1): PW writes are
    overwritten to val=0 in frames where CTRL bit 6 is clear (pulse
    waveform not selected)."""
    parser = _make_parser()
    df = _make_noise_with_pw_changes_df()
    out = parser._simplify_pcm(df.copy())
    pw_rows_after_ctrl = out[(out["clock"] > 50) & (out["reg"] == 2)]
    nonzero = pw_rows_after_ctrl[pw_rows_after_ctrl["val"] != 0]
    assert len(nonzero) == 0, (
        f"_simplify_pcm should zero PW writes during pulse-off, but "
        f"these survived non-zero:\n{nonzero.to_string()}"
    )


def test_simplify_pcm_audio_neutral_on_noise_phase_pw_writes():
    """PW writes during noise should be silent on real SID hardware
    AND on pyresidfp. Therefore the parser zeroing them at parse time
    must not change the rendered audio. If this test fails, either
    pyresidfp's noise+PW interaction differs from chip spec, or
    ``_simplify_pcm`` is doing more than what its docstring claims."""
    parser = _make_parser()
    df_pre = _make_noise_with_pw_changes_df()
    df_post = parser._simplify_pcm(df_pre.copy())

    sid_a = _fresh_sid()
    wav_pre = _render_dump(sid_a, df_pre)
    sid_b = _fresh_sid()
    wav_post = _render_dump(sid_b, df_post)

    mx = _wave_max_diff(wav_pre, wav_post)
    print(
        f"  _simplify_pcm audio diff: max|Δ|={mx} (bound {SYNTHETIC_AUDIO_NEUTRAL_BOUND})"
    )
    assert mx <= SYNTHETIC_AUDIO_NEUTRAL_BOUND, (
        f"_simplify_pcm changed audio by max|Δ|={mx} > "
        f"{SYNTHETIC_AUDIO_NEUTRAL_BOUND} int16. The transformation "
        f"should be silent on noise-phase PW writes; investigate "
        f"pyresidfp's noise+PW emulation or recent _simplify_pcm changes."
    )


def test_simplify_pcm_pre_pulse_on_synthesis():
    """Pins transformation 2 from the docstring: immediately before
    each CTRL write that turns the pulse waveform ON, ``_simplify_pcm``
    inserts a fresh PW row carrying the running PW value, so the SID
    gets the intended PW at the moment pulse becomes active."""
    parser = _make_parser()
    df = _build_input_df(
        [
            (0, 5, 0x09),
            (10, 6, 0xF0),
            (20, 24, 0x0F),
            (30, 1, 0x08),
            (40, 0, 0x80),
            (50, 4, 0x81),
            (1000, 2, 0xAB),
            (1010, 3, 0x07),
            (5000, 4, 0x41),
        ]
    )
    out = parser._simplify_pcm(df.copy())
    ctrl_pulse_on = out[(out["reg"] == 4) & (out["val"] == 0x41)]
    assert len(ctrl_pulse_on) > 0
    pulse_on_idx = ctrl_pulse_on.index[0]
    rows_before = out.loc[: pulse_on_idx - 1]
    pw_writes_just_before = rows_before[rows_before["reg"].isin([2, 3])].tail(2)
    assert len(pw_writes_just_before) > 0, (
        f"_simplify_pcm should synthesise a PW write before pulse-on "
        f"CTRL transitions; none found before idx {pulse_on_idx}"
    )


def test_simplify_ctrl_audio_neutral_when_zeroing_inactive_bits():
    """``_simplify_ctrl`` zeroes:
    - ring-mod (bit 2) when triangle waveform isn't selected
    - sync (bit 1) when no waveform is selected
    """
    parser = _make_parser()
    df = _build_input_df(
        [
            (0, 5, 0x09),
            (10, 6, 0xF0),
            (20, 24, 0x0F),
            (30, 1, 0x08),
            (40, 0, 0x80),
            (50, 4, 0x45),
            (5000, 4, 0x41),
        ]
    )
    df_pre = df.copy()
    df_post = parser._simplify_ctrl(df_pre.copy())

    sid_a = _fresh_sid()
    wav_pre = _render_dump(sid_a, df_pre)
    sid_b = _fresh_sid()
    wav_post = _render_dump(sid_b, df_post)

    mx = _wave_max_diff(wav_pre, wav_post)
    print(
        f"  _simplify_ctrl audio diff: max|Δ|={mx} (bound {SYNTHETIC_AUDIO_NEUTRAL_BOUND})"
    )
    assert mx <= SYNTHETIC_AUDIO_NEUTRAL_BOUND, (
        f"_simplify_ctrl changed audio by max|Δ|={mx}; should be "
        f"silent when zeroing inactive bits (ring-mod without "
        f"triangle, sync without waveform)"
    )


def test_squeeze_changes_audio_neutral_on_redundant_writes():
    """``_squeeze_changes`` drops rows where (reg, val) repeats the
    prev write to that reg. The same-value re-writes are audio-neutral
    per ``test_sid_same_value_writes.py``. Verify the squeeze
    transformation preserves audio."""
    parser = _make_parser()
    rows = [
        (0, 5, 0x09),
        (10, 6, 0xF0),
        (20, 1, 0x08),
        (30, 0, 0x80),
        (40, 4, 0x41),
    ]
    for i in range(20):
        rows.append((100 + i * 100, 24, 0x0F))
    df_pre = _build_input_df(rows)
    df_post = parser._squeeze_changes(df_pre.copy())

    mode_vol_count = (df_post["reg"] == 24).sum()
    assert mode_vol_count == 1, (
        f"_squeeze_changes should keep only the first MODE_VOL=15, "
        f"got {mode_vol_count}"
    )

    sid_a = _fresh_sid()
    wav_pre = _render_dump(sid_a, df_pre)
    sid_b = _fresh_sid()
    wav_post = _render_dump(sid_b, df_post)

    mx = _wave_max_diff(wav_pre, wav_post)
    print(
        f"  _squeeze_changes audio diff: max|Δ|={mx} (bound {SYNTHETIC_AUDIO_NEUTRAL_BOUND})"
    )
    assert mx <= SYNTHETIC_AUDIO_NEUTRAL_BOUND, (
        f"_squeeze_changes changed audio by max|Δ|={mx}; dropping "
        f"same-value writes should be safe per the same-value-writes "
        f"unit test"
    )


def test_cents_finer_grain_increases_distinct_fi_indices():
    """Structural invariant of FreqMapper: cents=1 should preserve
    more granularity than cents=50 on FREQ values spaced finer than
    a 50-cent bucket. Pin so a future FreqMapper change doesn't
    regress to a degenerate quantisation grid.
    """
    parser_50 = _make_parser(cents=50)
    parser_1 = _make_parser(cents=1)

    rows = [
        (0, 5, 0x09),
        (10, 6, 0xF0),
        (20, 24, 0x0F),
        (30, 2, 0x00),
        (40, 3, 0x08),
        (50, 4, 0x41),
    ]
    base_clock = 200000
    for i, freq_word in enumerate([4440, 4460, 4480, 4500, 4520]):
        rows.append((base_clock + i * 50000, 0, freq_word & 0xFF))
        rows.append((base_clock + i * 50000 + 10, 1, (freq_word >> 8) & 0xFF))
    df = _build_input_df(rows)

    df_50 = parser_50._quantize_freq_to_cents(parser_50._combine_regs(df.copy()))
    df_1 = parser_1._quantize_freq_to_cents(parser_1._combine_regs(df.copy()))

    freq_50 = df_50[df_50["reg"] == 0]["val"].nunique()
    freq_1 = df_1[df_1["reg"] == 0]["val"].nunique()
    print(f"  cents=50 distinct FREQ fi-indices: {freq_50}")
    print(f"  cents=1  distinct FREQ fi-indices: {freq_1}")
    assert freq_1 > freq_50, (
        f"cents=1 should produce > cents=50 distinct fi-indices on "
        f"FREQ values within a 50-cent bucket; got {freq_1} > {freq_50} "
        f"= {freq_1 > freq_50}"
    )
