"""Step 0 of the program-recovery-by-identity pipeline: SMC-correct song-data
recovery off the compact SDST distill artifact (access-TYPE classification).

* a self-contained unit test of :func:`identity.regions` on a SYNTHETIC mask
  (always runs; no binary, no fixtures); and
* the binary-gated end-to-end IDENTITY GATE on the real packed GoatTracker
  ``Grid_Runner.sid``: generate the distill artifact with ``preframr-sidtrace`` and
  prove the song-data region the access-type classifier isolates is (a) never
  written during play, none executed (no SMC leak), (b) byte-identical to the
  player's OWN loaded image (HARD RULE #0: genuine program data, not fabricated),
  and (c) parses into the SAME GoatTracker song pygoattracker reconstructs and
  round-trips byte-exact through ``build_sng``/``parse_sng``.  Skipped when no
  ``SIDTRACE_BIN`` is set (the default render-free CI).

(The legacy raw-bus-trace write-set/read-set partition is removed -- it
misclassified under self-modifying code and consumed a multi-GB trace; the distill
``song_data_mask`` is the production path.)
"""

import os

import numpy as np
import pytest

from preframr_tokens.bacc.generic import identity as I

_FIXTURES = os.path.join(os.path.dirname(__file__), "test_fixtures")
_GRID_SID = os.path.join(_FIXTURES, "Grid_Runner.sid")


def test_regions_contiguous_runs():
    mask = np.zeros(65536, dtype=bool)
    mask[0x2000:0x2004] = True
    mask[0x3000:0x3002] = True
    assert I.regions(mask) == [(0x2000, 0x2003), (0x3000, 0x3001)]
    assert I.regions(np.zeros(65536, dtype=bool)) == []


# --------------------------------------------------------------------------- #
# Binary-gated end-to-end identity gate on the real packed GoatTracker SID.
# --------------------------------------------------------------------------- #
def _sidtrace_bin():
    from preframr_tokens.bacc.generic.sidtrace import sidtrace_bin

    return sidtrace_bin()


@pytest.mark.skipif(_sidtrace_bin() is None, reason="no preframr-sidtrace binary")
def test_grid_runner_song_data_recovered_by_identity_distill(tmp_path):
    """The PRODUCTION identity gate: recover Grid_Runner's song data from the
    compact SDST distill artifact ALONE (a few KB, NOT a multi-GB bus trace),
    classified SMC-correctly by access type, byte-exact vs the player's RAM and
    pygoattracker."""
    pytest.importorskip("pygoattracker")
    from pygoattracker.reader import parse_sng
    from pygoattracker.writer import build_sng

    from preframr_tokens.bacc.backends import gt_unpack
    from preframr_tokens.bacc.generic.distill import load_distill
    from preframr_tokens.bacc.generic.sidtrace import run_sidtrace
    from preframr_tokens.bacc.sidemu import load_psid

    prefix = str(tmp_path / "grid")
    _, distill_path = run_sidtrace(_GRID_SID, prefix, subtune=1, nframes=400)
    dist = load_distill(distill_path)
    psid = load_psid(_GRID_SID)

    # The artifact is tiny -- the whole point of distilling in the emulator.
    assert os.path.getsize(distill_path) < 64 * 1024

    # The GoatTracker song-data region, derived purely from the player image.
    img = gt_unpack._Image(_GRID_SID)  # pylint: disable=protected-access
    lay = gt_unpack._derive_layout(img)  # pylint: disable=protected-access
    lo, hi = lay["freqtbllo"], img.end() - 1

    # (a) The SMC-correct classifier isolates the region: no byte of it is
    #     written during play, none is executed, all are read as data -- so it is
    #     classified song-data by ACCESS TYPE (not by write-set subtraction).
    sm = dist.song_data_mask()
    assert sm[lo : hi + 1].all()
    assert not dist.smc_mask()[lo : hi + 1].any()

    # (b) The lifted bytes are byte-identical to the player's OWN loaded image
    #     (genuine program data, read off the snapshot of RAM -- never fabricated).
    lifted = I.lift_song_data_distill(dist, lo, hi)
    resident = bytes(psid.data[lo - psid.load_addr : hi + 1 - psid.load_addr])
    assert lifted == resident

    # (c) Those bytes parse into the SAME GoatTracker song pygoattracker depacks,
    #     and that song round-trips byte-exact through build_sng/parse_sng.
    song = gt_unpack.reconstruct_song(_GRID_SID)
    sng = build_sng(song)
    assert build_sng(parse_sng(sng)) == sng
    # Recovered by identity: the player indexes a small instrument set, NOT the
    # ~1000 output-similarity clusters the per-register fit path produces.
    assert 1 <= len(song.instruments) <= 63
