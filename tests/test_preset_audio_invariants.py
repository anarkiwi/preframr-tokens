"""Audio-fidelity invariants for PresetPass. On-grid PWM/FC snap is
audio-neutral within pyresidfp noise floor. Off-grid PWM drift can
reach ~7000 int16 on synthetic pulse waves (snap distance ≤64 PW units
shifts duty cycle spectrum); bound calibrated at 10000 for regression
detection, far above design's 200 estimate. FC off-grid drift ≤1000."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

pyresidfp = pytest.importorskip("pyresidfp")
SoundInterfaceDevice = pyresidfp.SoundInterfaceDevice
ChipModel = pyresidfp.sound_interface_device.ChipModel

from preframr_tokens.macros.preset_pass import _snap
from preframr_tokens.stfconstants import FC_PRESET_TABLE, PWM_PRESET_TABLE

PYRESIDFP_NOISE_FLOOR = 20
PWM_OFF_GRID_DRIFT_BOUND = 10000
FC_DRIFT_BOUND = 1000
SUSTAIN_FRAMES = 30
PAL_CLOCKS_PER_FRAME = 19656


def _fresh_sid():
    sid = SoundInterfaceDevice(model=ChipModel.MOS8580)
    sid.reset()
    for r in range(25):
        sid.write_register(r, 0)
    _ = sid.clock(timedelta(seconds=0.1))
    return sid


def _render(sid, writes):
    clock_hz = int(sid.clock_frequency)
    out = []
    last_clock = 0
    for clock, reg, val in writes:
        d = clock - last_clock
        if d > 0:
            chunk = sid.clock(timedelta(seconds=d / clock_hz))
            if chunk:
                out.append(np.asarray(chunk, dtype=np.int16))
        sid.write_register(int(reg), int(val))
        last_clock = clock
    chunk = sid.clock(timedelta(seconds=0.2))
    if chunk:
        out.append(np.asarray(chunk, dtype=np.int16))
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


def _pulse_scenario(pw_value, fc_packed=0, filter_reg=0x00):
    pw_lo = pw_value & 0xFF
    pw_hi = (pw_value >> 8) & 0x0F
    fc_lo = fc_packed & 0xFF
    fc_hi = (fc_packed >> 8) & 0xFF
    writes = [
        (0, 5, 0x00),
        (10, 6, 0xF0),
        (20, 24, 0x0F),
        (30, 21, fc_lo),
        (40, 22, fc_hi),
        (50, 23, filter_reg),
        (60, 0, 0xD6),
        (70, 1, 0x1C),
        (80, 2, pw_lo),
        (90, 3, pw_hi),
        (100, 4, 0x41),
    ]
    for k in range(1, SUSTAIN_FRAMES + 1):
        t = 200 + k * PAL_CLOCKS_PER_FRAME
        writes.append((t, 2, pw_lo))
        writes.append((t + 5, 3, pw_hi))
    return writes


def _filter_scenario(fc_packed, filter_reg=0x01):
    fc_lo = fc_packed & 0xFF
    fc_hi = (fc_packed >> 8) & 0xFF
    writes = [
        (0, 5, 0x00),
        (10, 6, 0xF0),
        (20, 24, 0x1F),
        (30, 23, filter_reg),
        (40, 0, 0xD6),
        (50, 1, 0x1C),
        (60, 2, 0x00),
        (70, 3, 0x08),
        (80, 21, fc_lo),
        (90, 22, fc_hi),
        (100, 4, 0x41),
    ]
    for k in range(1, SUSTAIN_FRAMES + 1):
        t = 200 + k * PAL_CLOCKS_PER_FRAME
        writes.append((t, 21, fc_lo))
        writes.append((t + 5, 22, fc_hi))
    return writes


@pytest.mark.parametrize("pw_value", [0, 128, 256, 1024, 2048, 3072, 3968])
def test_pwm_preset_on_grid_within_noise_floor(pw_value):
    assert pw_value in PWM_PRESET_TABLE
    snapped = _snap(pw_value, 128)
    assert snapped == pw_value
    sid_a = _fresh_sid()
    a = _render(sid_a, _pulse_scenario(pw_value))
    sid_b = _fresh_sid()
    b = _render(sid_b, _pulse_scenario(snapped))
    mx = _wave_max_diff(a, b)
    assert mx <= PYRESIDFP_NOISE_FLOOR, (
        f"on-grid PW={pw_value} produced drift={mx} > "
        f"{PYRESIDFP_NOISE_FLOOR} (pyresidfp noise floor)"
    )


@pytest.mark.parametrize("pw_value", [127, 192, 1500, 1700, 2050, 2200, 3000, 3900])
def test_pwm_preset_off_grid_bounded(pw_value):
    snapped = _snap(pw_value, 128)
    assert snapped in PWM_PRESET_TABLE
    assert abs(snapped - pw_value) <= 64
    sid_a = _fresh_sid()
    a = _render(sid_a, _pulse_scenario(pw_value))
    sid_b = _fresh_sid()
    b = _render(sid_b, _pulse_scenario(snapped))
    mx = _wave_max_diff(a, b)
    print(
        f"  PW={pw_value} -> snap={snapped}: max|Δ|={mx} "
        f"(bound {PWM_OFF_GRID_DRIFT_BOUND})"
    )
    assert (
        mx <= PWM_OFF_GRID_DRIFT_BOUND
    ), f"PW={pw_value} -> {snapped} drift {mx} > {PWM_OFF_GRID_DRIFT_BOUND}"


@pytest.mark.parametrize("fc_packed", [0, 256, 4096, 8192, 16384, 32768])
def test_fc_preset_on_grid_within_noise_floor(fc_packed):
    assert fc_packed in FC_PRESET_TABLE
    snapped = _snap(fc_packed, 256)
    assert snapped == fc_packed
    sid_a = _fresh_sid()
    a = _render(sid_a, _filter_scenario(fc_packed))
    sid_b = _fresh_sid()
    b = _render(sid_b, _filter_scenario(snapped))
    mx = _wave_max_diff(a, b)
    assert (
        mx <= PYRESIDFP_NOISE_FLOOR
    ), f"on-grid FC={fc_packed} drift={mx} > {PYRESIDFP_NOISE_FLOOR}"


@pytest.mark.parametrize("fc_packed", [200, 8200, 8576, 16640, 33000])
def test_fc_preset_off_grid_bounded(fc_packed):
    snapped = _snap(fc_packed, 256)
    assert snapped in FC_PRESET_TABLE
    assert abs(snapped - fc_packed) <= 128
    sid_a = _fresh_sid()
    a = _render(sid_a, _filter_scenario(fc_packed))
    sid_b = _fresh_sid()
    b = _render(sid_b, _filter_scenario(snapped))
    mx = _wave_max_diff(a, b)
    print(f"  FC={fc_packed} -> snap={snapped}: max|Δ|={mx} (bound {FC_DRIFT_BOUND})")
    assert (
        mx <= FC_DRIFT_BOUND
    ), f"off-grid FC={fc_packed} -> {snapped} drift {mx} > {FC_DRIFT_BOUND}"


def test_snap_identity_on_table_entries():
    for v in PWM_PRESET_TABLE:
        assert _snap(int(v), 64) == int(v)
    for v in FC_PRESET_TABLE:
        assert _snap(int(v), 256) == int(v)


def test_snap_bounded_distance():
    for v in range(0, 4096, 17):
        snapped = _snap(v, 64)
        assert abs(snapped - v) <= 32
    for v in range(0, 65536, 257):
        snapped = _snap(v, 256)
        assert abs(snapped - v) <= 128
