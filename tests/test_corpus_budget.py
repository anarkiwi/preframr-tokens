"""Parametrized corpus budget gate (HARD RULE #0, scaled to driver diversity).

The A Mind Is Born gate proves a single adversarial tune is not an irreducible
wall.  This gate proves the SAME for a *spread* of the corpus: 20 HVSC tunes from
20 DIFFERENT drivers (GoatTracker, DMC, Music_Assembler, JCH, Soundmonitor,
Master_Composer, ... -- the major families), each an average-complexity tune
(moderate length and register activity, no near-silent stub, no digi/extreme
outlier).  Every tune MUST encode byte-exact via the generic driver, and at
< 1 token/frame.

Each fixture is a committed per-frame register ``state`` (uint8, compressed) --
the real ``preframr-sidtrace`` output of the tune, reproducible from the ``.sid``
with the binary (see ``tests/test_fixtures/budget/MANIFEST.tsv``).  So this runs
in CI with no binary, identically to the A_Mind gate, and the heavy generator
cover is what each case exercises.

Parametrized one case per (driver, fixture) -- each is its own test id, so
pytest-xdist distributes them across workers (``-n auto``).  Byte-exactness is
MANDATORY for every tune.  The token budget is the gate, co-equal with
residual-0: byte-exact-but-dense is a failure.

The STRUCTURE path (#12) has now landed.  A structured tune (JCH, DMC,
Music_Assembler, GoatTracker, ...) recovers its real tracker source -- a deduped
instrument pool + factored patterns/orderlist + the porta/vibrato accumulator
generators -- DIRECTLY from the distill artifact (``structure_recover`` +
``structure_ir``), byte-exact, and serializes < 1 token/frame where the output-fit
cover floored >= 1.  The recovered structure (the compact token stream) is committed
as a tiny ``<fixture>.sir.npz`` companion next to each ``state`` fixture, so the
structure gate runs in CI with no binary: the committed ids deserialize to the exact
IR, the FREQ lanes render from that IR alone byte-exact (the §state-machine accumulator
identity), and the full 25-register render is byte-exact against the committed state.
Those drivers are now a HARD PASS (no xfail).

The drivers still marked ``xfail(strict=False)`` are the ones the structure path does
NOT yet recover under budget -- each with a precise, falsifiable reason
(:data:`_XFAIL`): a pure-code / table-less tune whose structure discovery finds nothing
(it correctly FALLS BACK to the generator cover, which for that tune still floors >= 1),
a driver whose pattern grammar the byte-exact round-trip falsifies (a structure-detection
heuristic gap, not a wall), or a recovered structure whose raw instrument/program tables
still cost >= 1 token/frame pending generator-FITTING of those tables (the next
increment).  Byte-exactness is asserted for every tune regardless.
"""

import os

import numpy as np
import pytest

from preframr_tokens.bacc.generic.structure_ir import (
    render_freq_from_ir,
    render_structure,
    structure_ir_from_ids,
    structure_ir_to_ids,
)
from preframr_tokens.bacc.generic.tracker import render_from_fits
from preframr_tokens.bacc.tracker_ir import lift, unlift
from preframr_tokens.bacc.tracker_serialize import _ir_to_ids

_FIXDIR = os.path.join(os.path.dirname(__file__), "test_fixtures", "budget")

# One average-complexity tune per driver (20 distinct drivers).  Each entry is
# ``(driver, fixture_filename)``; the fixture is the committed per-frame state
# rendered from ``hvsc_path`` (recorded in MANIFEST.tsv).  Driver is the test id.
_CORPUS = [
    ("GoatTracker_V2.x", "GoatTracker_V2x__Regurgitated_Meatloaf.npz"),
    ("DMC", "DMC__Liver.npz"),
    ("Music_Assembler", "Music_Assembler__House.npz"),
    ("MoN/FutureComposer", "MoN-FutureComposer__Action_News_5_tune_R3.npz"),
    ("JCH_NewPlayer", "JCH_NewPlayer__Get_Funky.npz"),
    ("Soundmonitor", "Soundmonitor__Dr_Psycolog.npz"),
    ("Master_Composer", "Master_Composer__Believe_in_Music.npz"),
    ("Geir_Tjelta/SIDDuzz'It", "Geir_Tjelta-SIDDuzzIt__Reflections.npz"),
    ("SoedeSoft", "SoedeSoft__Sad_Day.npz"),
    ("Digitalizer_V2.x", "Digitalizer_V2x__What_I_Can_Is_This.npz"),
    ("RoMuzak_V6.x", "RoMuzak_V6x__Flux_Collection_tune_12.npz"),
    ("Electrosound", "Electrosound__Jolly_Rodger.npz"),
    ("DefMon", "DefMon__Teckomatorp.npz"),
    ("Ubik's_Musik", "Ubiks_Musik__Insanity_Insane_Mix.npz"),
    ("TFX", "TFX__Eternal_Skies.npz"),
    ("AMP", "AMP__Raster_Runner_V2.npz"),
    ("20CC", "20CC__What_Have_I_Done_To.npz"),
    ("Cyberlogic_SoundStudio", "Cyberlogic_SoundStudio__Nucular_Beatz.npz"),
    ("EMS/Odie", "EMS-Odie__Reaxion.npz"),
    ("HardTrack_Composer", "HardTrack_Composer__Acid_Runner_Remix.npz"),
]

# HONEST-RED (the C3 loophole-close gate PR): the structure-path budget now requires the
# shipped stream to be LZ-FREE (``c3_no_lz_in_measured_stream`` over the WHOLE stream,
# pattern-bank sections included).  Every committed ``.sir.npz`` still rides on the codec's
# backward-LZ (``_struct_lz``, the ``_REPEAT`` sentinel) -- so the certified sub-1 tok/frame
# was riding on LZ, not on recovered structure (HARD RULE #0).  Closing that loophole flips
# all 18 STRUCTURE-PATH drivers to honest xfail; they are byte-exact (that still holds) but
# NOT under-budget without LZ.  The fix is the instrument-program execution recovery (the
# next PR) so the shipped stream needs no ``_struct_lz`` -- NOT relaxing the gate.  The two
# COVER-PATH drivers (Soundmonitor, Master_Composer; no ``.sir.npz``) are not affected by
# this structure-path check and stay HARD PASS.
_LZ_XFAIL_REASON = (
    "structure shipped stream rides _struct_lz (_REPEAT) -- certified <1 tok/frame was "
    "LZ-dependent, not recovered structure (HARD RULE #0); pending instrument-program "
    "execution recovery so the shipped stream needs no LZ"
)
_COVER_DRIVERS = {"Soundmonitor", "Master_Composer"}
_XFAIL = {
    driver: _LZ_XFAIL_REASON
    for driver, _fixture in _CORPUS
    if driver not in _COVER_DRIVERS
}


def _fixture_base(fixture):
    """The committed structure-serialization companion for a state ``fixture`` (the tiny
    ``<base>.sir.npz`` carrying the recovered structure token ids), or None if absent.
    """
    base = fixture[:-4] if fixture.endswith(".npz") else fixture
    sir = os.path.join(_FIXDIR, base + ".sir.npz")
    return sir if os.path.exists(sir) else None


def _params():
    """Build the parametrized cases: id = driver, xfail the not-yet-under-budget ones."""
    cases = []
    for driver, fixture in _CORPUS:
        marks = ()
        if driver in _XFAIL:
            marks = pytest.mark.xfail(reason=_XFAIL[driver], strict=False)
        cases.append(pytest.param(driver, fixture, marks=marks, id=driver))
    return cases


def _cover_under_budget(driver, state, nframes, boot):
    """The output-fit generator-cover path (the structure-less fallback): cover every
    lane, lift to the Tracker IR, assert byte-exact render and < 1 token/frame.  This is
    the path a tune with no recoverable structure (pure-code / grammar-mismatch) takes.
    """
    ir = lift(state, None, nframes, boot)
    genfits, eventfits = unlift(ir)
    rendered = render_from_fits(genfits, eventfits, ir.note_table, nframes)
    assert np.array_equal(
        rendered, state
    ), f"{driver}: generic (cover) render is not byte-exact"
    tok_per_frame = len(_ir_to_ids(ir)) / nframes
    assert tok_per_frame < 1.0, (
        f"{driver}: generic (cover) encoding is {tok_per_frame:.3f} token/frame "
        f"(>= 1.0). This is NOT an irreducible wall -- recover the structure."
    )


def _structure_under_budget(driver, fixture, state, nframes):
    """The STRUCTURE path: the committed structure token ids deserialize to the exact IR,
    the FREQ lanes render from that IR alone byte-exact (the accumulator identity), the full
    25-register render is byte-exact against ``state``, and the stream is < 1 token/frame.
    """
    ids = list(np.load(_fixture_base(fixture))["ids"].astype(np.int64))

    # The committed ids deserialize to the structure IR; re-serializing is identical (the
    # codec is exact) -- the structure round-trips with no escape.
    ir = structure_ir_from_ids(ids)
    assert (
        structure_ir_to_ids(ir) == ids
    ), f"{driver}: structure ids re-serialize mismatch"

    # FREQ renders from the DESERIALIZED IR alone, byte-exact full-length (the displaced
    # "note table" collapses to grid pitches + the recovered porta/vibrato accumulators).
    freq = render_freq_from_ir(ir, state)
    for vi, (rlo, rhi) in enumerate(((0, 1), (7, 8), (14, 15))):
        ref = state[:, rlo] | (state[:, rhi] << 8)
        assert np.array_equal(
            freq[vi], ref
        ), f"{driver}: structure freq lane {vi} render is not byte-exact"

    # Full 25-register render byte-exact against the committed state (freq from the IR; the
    # instrument-driven lanes from the byte-exact anchor pending their replay -- M0).
    ir._state = state
    rendered = render_structure(ir)
    assert np.array_equal(
        rendered, state
    ), f"{driver}: structure render is not byte-exact"

    # Token budget: < 1 token/frame AND the shipped stream must be LZ-FREE.  HARD RULE #0:
    # the certified tok/frame must come from RECOVERED STRUCTURE, not from compressing the
    # serialized stream.  The committed structure ids still ride on the codec's backward-LZ
    # (`_struct_lz`, the `_REPEAT` sentinel) across the pattern-bank sections, so this no-LZ
    # requirement currently FAILS for every structure-path tune (the honest red the gate
    # PR surfaces): a sub-1 that rides on `_struct_lz` is not a recovered floor.  The fix is
    # the instrument-program execution recovery (so the shipped stream needs no `_struct_lz`),
    # NOT relaxing this check.
    from tools.codec_gate import c3_no_lz_in_measured_stream

    c3_no_lz_in_measured_stream(ids)  # raises CheckFailure while the stream rides on LZ
    tok_per_frame = len(ids) / nframes
    assert (
        tok_per_frame < 1.0
    ), f"{driver}: structure encoding is {tok_per_frame:.3f} token/frame (>= 1.0)."


@pytest.mark.parametrize("driver,fixture", _params())
def test_corpus_tune_generic_under_one_token_per_frame(driver, fixture):
    """``driver``'s average tune encodes byte-exact at < 1 token/frame (generic).

    A structured tune takes the STRUCTURE path (its committed ``.sir.npz`` recovered
    tracker source); an unstructured tune takes the generator-cover fallback.  Both MUST
    render byte-exact and serialize < 1 token/frame.
    """
    state = np.load(os.path.join(_FIXDIR, fixture))["state"].astype(np.int64)
    nframes = len(state)
    boot = [int(v) for v in state[0]]

    if _fixture_base(fixture) is not None:
        _structure_under_budget(driver, fixture, state, nframes)
    else:
        _cover_under_budget(driver, state, nframes, boot)
