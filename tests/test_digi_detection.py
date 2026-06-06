"""Digi-detection validation (work order PART C / design/digi_detection_reference.md): each real digi
playback method's per-frame write-density signature must flip ``dump_meta.is_digi``, a melodic tune and
the Baggis non-digi must NOT, a PW *sweep* (one write/frame) must NOT be mistaken for a PWM digi, and a
flagged digi must be excluded by ``filter_dump_paths`` (the corpus/parse residual-zero gate keeps digis
-- a sub-frame PCM modality below register_state resolution -- out of the generator model).
"""

import os
import tempfile

import pandas as pd

from preframr_tokens.dump_meta import (
    _build_meta_from_raw,
    filter_dump_paths,
    write_meta,
)

_DUMP_COLS = ("clock", "irq", "chipno", "reg", "val")
_PHI = 985248
_FRAME = _PHI // 50


def _raw(rows):
    return pd.DataFrame(rows, columns=list(_DUMP_COLS))


def _frame_rows(frame, writes):
    """``writes`` = list of (reg, val) all stamped with the same per-frame ``irq`` value (``frame+1``),
    so ``_build_meta_from_raw`` (which groups by the ``irq`` column) counts them as one frame's worth.
    """
    irq = frame + 1
    clock = frame * _FRAME
    return [(clock, irq, 0, int(reg), int(val)) for reg, val in writes]


def _meta(rows):
    return _build_meta_from_raw(None, _raw(rows))


def _melodic_preamble(frame):
    """A normal per-frame note: one freq lo/hi, one gate-on ctrl, held adsr -- ctrl/vol/pw density ~1."""
    return _frame_rows(frame, [(0, 0x80), (1, 0x11), (4, 0x41), (5, 0x00), (6, 0xF0)])


def test_d418_volume_digi_flagged():
    """$D418 volume digi (the common method): many MODE_VOL(24) writes in one frame -> vol_max high."""
    rows = []
    for f in range(4):
        rows += _melodic_preamble(f)
    rows += _frame_rows(4, [(24, v & 0x0F) for v in range(60)])
    m = _meta(rows)
    assert m["vol_changes_per_frame_max"] >= 40
    assert m["is_digi"] is True


def test_soundemon_ctrl_digi_flagged():
    """SounDemoN test-bit sample-and-hold: dense per-sample CONTROL(reg 4) toggles -> ctrl_max high."""
    rows = []
    for f in range(4):
        rows += _melodic_preamble(f)
    rows += _frame_rows(4, [(4, 0x09 if i % 2 else 0x01) for i in range(40)])
    m = _meta(rows)
    assert m["ctrl_changes_per_frame_max"] >= 20
    assert m["is_digi"] is True


def test_pwm_digi_flagged():
    """PWM digi: a pulse+test+freq0 voice modulating PULSE WIDTH ($D402/$D403) per sample -> pw_max high.
    The signature the old heuristic MISSED; pw_max>=40 now catches it."""
    rows = []
    for f in range(4):
        rows += _melodic_preamble(f)
    setup = [(0, 0x00), (1, 0x00), (4, 0x49), (6, 0xFF)]
    rows += _frame_rows(4, setup + [(3, 0x00 if i % 2 else 0xFF) for i in range(50)])
    m = _meta(rows)
    assert m["pw_changes_per_frame_max"] >= 40
    assert m["is_digi"] is True


def test_melodic_tune_not_flagged():
    """A plain melodic tune (ctrl/vol/pw density ~1, peak ctrl_max ~3) is NOT a digi."""
    rows = []
    for f in range(40):
        rows += _melodic_preamble(f)
        if f % 8 == 0:
            rows += _frame_rows(f, [(4, 0x40), (4, 0x41)])
    m = _meta(rows)
    assert m["ctrl_changes_per_frame_max"] <= 3
    assert m["vol_changes_per_frame_max"] <= 1
    assert m["pw_changes_per_frame_max"] <= 1
    assert m["is_digi"] is False


def test_baggis_like_ctrl_active_non_digi_not_flagged():
    """The Baggis correction: a long ctrl-ACTIVE non-digi (vol_max=1, ctrl_max~3, pw_max~6) is NOT a digi
    -- row count / ctrl activity below the per-frame density thresholds must not flip is_digi.
    """
    rows = []
    for f in range(200):
        rows += _melodic_preamble(f)
        rows += _frame_rows(f, [(4, 0x40), (4, 0x41), (2, f & 0xFF), (3, 0)])
    m = _meta(rows)
    assert m["ctrl_changes_per_frame_max"] <= 4
    assert m["pw_changes_per_frame_max"] <= 6
    assert m["is_digi"] is False


def test_pw_sweep_not_mistaken_for_pwm_digi():
    """A PW *sweep* writes PW about once per frame (pw_max ~1), far below the digi threshold -- the
    work order's explicit guardrail that the PWM clause does not false-positive on sweeps.
    """
    rows = []
    for f in range(60):
        rows += _frame_rows(
            f, [(0, 0x80), (1, 0x11), (2, f & 0xFF), (3, (f >> 8) & 0x0F)]
        )
    m = _meta(rows)
    assert m["pw_changes_per_frame_max"] <= 2
    assert m["is_digi"] is False


def test_digi_excluded_by_filter_dump_paths():
    """A flagged digi is dropped by ``filter_dump_paths(exclude_digi=True)`` -- the corpus/parse path that
    keeps is_digi out of the generator residual-zero gate; a melodic tune is admitted.
    """
    with tempfile.TemporaryDirectory() as tmp:
        digi_rows = []
        for f in range(4):
            digi_rows += _melodic_preamble(f)
        digi_rows += _frame_rows(4, [(24, v & 0x0F) for v in range(60)])
        digi_path = os.path.join(tmp, "digi.dump.parquet")
        _raw(digi_rows).to_parquet(digi_path)
        write_meta(digi_path, _raw(digi_rows))

        mel_rows = []
        for f in range(40):
            mel_rows += _melodic_preamble(f)
        mel_path = os.path.join(tmp, "melodic.dump.parquet")
        _raw(mel_rows).to_parquet(mel_path)
        write_meta(mel_path, _raw(mel_rows))

        admitted, dropped = filter_dump_paths([digi_path, mel_path], exclude_digi=True)
        assert dropped.get(digi_path) == "digi"
        assert mel_path in admitted
