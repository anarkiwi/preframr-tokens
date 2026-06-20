"""Gate: the framing substrate is lossless on a multispeed tune.

A multispeed tune calls its play routine N>1 times per PAL raster frame. Framing
at the single raster frame (CPF=19656) keeps only the LAST of the N play-calls'
writes per register, silently dropping every intermediate change -- measured
>50% of register-value changes dropped on Galway's Times_of_Lore. Auto-detecting
the tune's true play period (``detect_play_period``) and framing at that sub-
frame period puts every play-call boundary on its own frame, so no change is
dropped (lossless).

These assert the substrate property only (capture the true bus); they do NOT
require a Galway backend.
"""

import numpy as np
import pandas as pd

from preframr_tokens.codec import lsp_validate as LV


def _dump_arrays(dump_path):
    df = pd.read_parquet(dump_path, columns=["clock", "reg", "val", "chipno"])
    df = df[df["chipno"] == 0].sort_values("clock")
    return (
        df["clock"].to_numpy(np.int64),
        df["reg"].to_numpy(int),
        df["val"].to_numpy(int),
    )


def test_galway_is_multispeed_and_subframe_framing_is_lossless(galway_paths):
    cyc, reg, val = _dump_arrays(galway_paths[1])

    # (a) the detector classifies the tune multispeed and finds a sub-frame N.
    period = LV.detect_play_period(cyc)
    assert period < 0.9 * LV.CPF, "Galway tune must be detected multispeed"
    n_calls = LV.CPF / period
    assert n_calls > 1.5, f"expected several play-calls/frame, got {n_calls:.2f}"

    # (b) single-CPF framing drops a large fraction of true-bus changes ...
    loss_single = LV.framing_change_loss(cyc, reg, val, LV.CPF)
    assert loss_single > 0.3, f"single-CPF should drop a lot, got {loss_single:.3f}"

    # ... while framing at the detected play period is lossless.
    loss_detected = LV.framing_change_loss(cyc, reg, val, period)
    assert loss_detected < 0.01, f"detected-period framing lossy: {loss_detected:.4f}"

    # The improvement is large and unambiguous.
    assert loss_single - loss_detected > 0.3


def test_galway_default_per_frame_state_autodetects(galway_paths):
    # per_frame_state with cpf unset (auto-detect) frames at the sub-frame period:
    # far more frames than the raster-frame count, capturing every play-call.
    from preframr_tokens.codec.lane_grammar import per_frame_state

    auto = per_frame_state(galway_paths[1], maxframes=10**9)
    raster = per_frame_state(galway_paths[1], LV.CPF, maxframes=10**9)
    assert auto is not None and raster is not None
    assert len(auto) > 1.5 * len(raster)
