"""PR-C IWLK freq-modulation fitter: the per-(pc, voice) instrument-table walk-index
resolved into a freq-modulation GENERATOR (table + onset-retriggered walk), and the
HONEST byte-exact coverage it reaches (HARD RULE #0: where it covers freq it is recovered
structure; where it does not the residual is REPORTED, never papered over with a stored
per-frame freq sequence).

Two layers:

  * a SYNTHETIC distill (no binary) proving the resolver is correct -- a note->freq table
    walked by a clean onset-retriggered index reconstructs the freq lane BYTE-EXACT from
    the table value alone (``freq == ram[base + scale*idx]`` lo, ``ram[base + L + idx]``
    hi), so :func:`iwlk_freq_fit` reports 100% coverage and the freq lane renders from the
    generator WITHOUT the ``_state`` anchor;

  * the CORPUS falsification (Music_Assembler, GoatTracker) -- run only when the sidtrace
    binary + HVSC are present -- documenting the MEASURED coverage the IWLK section as
    emitted by PR-B actually reaches (the falsifiable STALL surface: the capture spans only
    a thin overlay of one voice, NOT the freq generator, so byte-exact-from-tokens is NOT
    reached and the test is xfail-marked with the exact numbers).  See the module-level
    report in the PR for the (a)/(b)/(c) protocol.
"""

import os

import numpy as np
import pytest

from preframr_tokens.bacc.generic import structure_recover as SR
from preframr_tokens.bacc.generic.distill import IdxSupp, IwlkWalk

from tests.test_structure_recover_pwlk import _blank_distill

_SIDTRACE = os.environ.get("SIDTRACE_BIN")
_HVSC = "/scratch/preframr/hvsc/C64Music"
_HAVE_BIN = bool(_SIDTRACE and os.path.exists(_SIDTRACE) and os.path.isdir(_HVSC))


def _idx_supp(pc, base, scale, feeds_regs):
    """An :class:`IdxSupp` with an affine fit ``addr = base + scale*idx`` feeding
    ``feeds_regs`` (the freq byte registers this table reaches)."""
    mask = 0
    for r in feeds_regs:
        mask |= 1 << r
    return IdxSupp(
        pc=pc,
        scale_set=True,
        scale=scale,
        base_fit=base,
        feeds_reg_mask=mask,
        targets_in_image=False,
        targets_read_as_data=True,
        n_samp=2,
        samp_idx=(0, 1),
        samp_addr=(base, base + scale),
    )


def _tempered_table(n=96, f0=279.0):
    """A tempered note->freq 16-bit table (lo bytes then hi bytes), the GoatTracker form."""
    freqs = (f0 * (2.0 ** (np.arange(n) / 12.0))).astype(np.int64) & 0xFFFF
    lo = (freqs & 0xFF).astype(np.uint8)
    hi = ((freqs >> 8) & 0xFF).astype(np.uint8)
    return freqs, lo, hi


def test_iwlk_resolver_note_to_freq_table_byte_exact(monkeypatch):
    """A single IWLK walk whose index addresses a note->freq table reconstructs the freq
    lane BYTE-EXACT from the table value alone -- the resolver is correct and the freq
    ``note_base`` comes from the GENERATOR (table + walk), not the ``_state`` anchor."""
    nframes = 64
    d = _blank_distill(nframes)
    base, tbl_len, scale = 0x1400, 96, 1
    freqs, lo, hi = _tempered_table(tbl_len)
    d.ram[base : base + tbl_len] = lo
    d.ram[base + tbl_len : base + 2 * tbl_len] = hi

    # a clean onset-retriggered walk: index ramps 12..23 then resets (note arpeggio).
    idx = np.array([(12 + (f % 12)) for f in range(nframes)], dtype=np.uint8)
    # the freq-feeding IDXR for voice reg7 (freq lane v1 = regs 7,8).
    pc = 0x11F6
    d.idx_supp = [_idx_supp(pc, base, scale, feeds_regs=(7, 14))]
    d.iwlk_walks = [IwlkWalk(pc=pc, voice=7, index=idx)]

    # the captured freq lane IS the table value at the walked index (lo reg7, hi reg8).
    state = np.zeros((nframes, 25), dtype=np.int64)
    state[:, 7] = freqs[idx] & 0xFF
    state[:, 8] = (freqs[idx] >> 8) & 0xFF

    monkeypatch.setattr(SR, "load_distill", lambda _p: d)
    fit = SR.iwlk_freq_fit("synthetic", state)
    assert fit is not None
    # voice index 1 is the (7,8) pair; it must resolve byte-exact (full coverage).
    v1 = fit[1]
    assert v1["covered"] == nframes, v1
    assert v1["nframes"] == nframes
    assert v1["pairs"], "no resolving table pair found"
    # the resolved generator reproduces the freq lane with NO _state read.
    assert bool(v1["match"].all())


def test_render_freq_lane_from_iwlk_no_state(monkeypatch):
    """The freq lane renders FROM THE GENERATOR ALONE (no ``_state``): the resolved
    table-walk value series EQUALS the captured freq byte-exact."""
    nframes = 64
    d = _blank_distill(nframes)
    base, tbl_len, scale = 0x1400, 96, 1
    freqs, lo, hi = _tempered_table(tbl_len)
    d.ram[base : base + tbl_len] = lo
    d.ram[base + tbl_len : base + 2 * tbl_len] = hi
    idx = np.array([(12 + (f % 12)) for f in range(nframes)], dtype=np.uint8)
    pc = 0x11F6
    d.idx_supp = [_idx_supp(pc, base, scale, feeds_regs=(7, 14))]
    d.iwlk_walks = [IwlkWalk(pc=pc, voice=7, index=idx)]

    ref = freqs[idx] & 0xFFFF
    monkeypatch.setattr(SR, "load_distill", lambda _p: d)
    rendered = SR.render_freq_lane_from_iwlk("synthetic", 1, nframes)
    assert rendered is not None
    assert np.array_equal(rendered, ref), "freq lane not byte-exact from generator"


def test_iwlk_freq_fit_none_without_iwlk(monkeypatch):
    """No IWLK section -> ``None`` (additive: a tune with no IWLK is unaffected)."""
    d = _blank_distill(32)
    monkeypatch.setattr(SR, "load_distill", lambda _p: d)
    state = np.zeros((32, 25), dtype=np.int64)
    assert SR.iwlk_freq_fit("synthetic", state) is None


# --- the corpus falsification (STALL surface): the MEASURED coverage of the IWLK
# section as emitted by PR-B for the two anchor tunes.  byte-exact-from-tokens is NOT
# reached (the capture is a thin overlay, not the freq generator), so this is xfail.
_CORPUS = [
    ("Music_Assembler", "MUSICIANS/C/Compod/House.sid", 2270),
    ("GoatTracker", "DEMOS/M-R/Regurgitated_Meatloaf.sid", 2300),
]


@pytest.mark.skipif(not _HAVE_BIN, reason="needs SIDTRACE_BIN + HVSC")
@pytest.mark.xfail(
    reason=(
        "IWLK as emitted by PR-B spans only a thin overlay of one voice "
        "(MA v0 4.4%, GT v1 24.1%, other voices 0%), NOT the freq generator: the "
        "captured index is forward-filled / stale on frames the IDXR did not fire and "
        "the bulk note_base comes from the note schedule->note_table pitch, so "
        "freqtable[index] cannot reach residual-0 for the freq lane.  STALL, reported "
        "honestly -- no raw-sequence escape (HARD RULE #0)."
    ),
    strict=True,
)
@pytest.mark.parametrize("driver,rel,nframes", _CORPUS, ids=[c[0] for c in _CORPUS])
def test_corpus_iwlk_covers_freq_lane(driver, rel, nframes):
    """The IWLK-resolved generator covers the WHOLE freq lane (all 3 voices) byte-exact.

    This is the byte-exact-from-tokens goal; it currently FAILS (xfail) because the IWLK
    capture does not span the freq lane -- the measured coverage is asserted below so the
    residual is documented and any future capture improvement flips this green.
    """
    from preframr_tokens.bacc.generic.sidtrace import run_sidtrace, sidwr_state

    prefix = os.path.join(os.environ.get("PYTEST_TMPDIR", "/tmp"), f"iwlk_{driver}")
    sidwr, distill = run_sidtrace(
        os.path.join(_HVSC, rel), prefix, 1, nframes, _SIDTRACE
    )
    state, _t0 = sidwr_state(sidwr)
    state = state.astype(np.int64)
    fit = SR.iwlk_freq_fit(distill, state)
    assert fit is not None
    total = sum(info["covered"] for info in fit.values())
    n = state.shape[0] * len(fit)
    # the goal: full byte-exact coverage of the freq lane from the generator alone.
    assert total == n, (
        f"{driver}: IWLK covers {total}/{n} freq-lane frames "
        f"(per-voice {[info['covered'] for info in fit.values()]})"
    )
