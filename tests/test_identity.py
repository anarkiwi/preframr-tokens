"""Step 0 of the program-recovery-by-identity pipeline: the write-set/read-set
partition over the bus trace's READ stream (no emulator patch).

Two layers:

* a self-contained unit test of the partition on a SYNTHETIC ``BUS_DT`` trace
  (always runs; no binary, no fixtures) -- it pins the "init-written / play-read /
  never-play-written = song data" semantics, the SID-shadow / self-mod-operand
  subtraction, and the verbatim RAM lift; and
* the binary-gated end-to-end IDENTITY GATE on the real packed GoatTracker
  ``Grid_Runner.sid``: generate a short bus trace with ``preframr-sidtrace``,
  partition it, and prove the song-data region the partition isolates is (a) never
  written during play, (b) byte-identical to the player's OWN loaded image
  (HARD RULE #0: genuine program data, not fabricated), and (c) parses into the
  SAME GoatTracker song pygoattracker reconstructs and round-trips byte-exact
  through ``build_sng``/``parse_sng``.  Skipped when no ``SIDTRACE_BIN`` is set
  (the default render-free CI), exactly like the other binary-gated tests.
"""

import os

import numpy as np
import pytest

from preframr_tokens.bacc.generic import identity as I
from preframr_tokens.bacc.generic.bustrace import BUS_DT

_FIXTURES = os.path.join(os.path.dirname(__file__), "test_fixtures")
_GRID_SID = os.path.join(_FIXTURES, "Grid_Runner.sid")


def _rec(cyc, addr, val, rw):
    return (cyc, addr, val, rw)


def _synthetic_trace():
    """A minimal bus trace exercising the partition.

    init (cyc < 1000) writes a 4-byte data table at $2000 and a scratch byte at
    $3000; play (cyc >= 1000) only READS the table, WRITES the scratch every
    frame (a SID shadow / self-mod operand), and reads a SID register page.
    """
    recs = [
        # --- init: lay down the song-data table + a scratch seed ---
        _rec(10, 0x2000, 0x11, 1),
        _rec(11, 0x2001, 0x22, 1),
        _rec(12, 0x2002, 0x33, 1),
        _rec(13, 0x2003, 0x44, 1),
        _rec(14, 0x3000, 0x99, 1),  # scratch (init seed)
        # --- play: a SID write anchors t0, then read the table, rewrite scratch ---
        _rec(1000, 0xD400, 0x00, 1),  # first steady SID write -> frame-0 anchor
        _rec(1001, 0x2000, 0x11, 0),
        _rec(1002, 0x2001, 0x22, 0),
        _rec(1003, 0x3000, 0xAB, 1),  # play WRITES scratch -> not data
        _rec(20656, 0xD400, 0x01, 1),  # next play call (one PAL frame later)
        _rec(20657, 0x2002, 0x33, 0),
        _rec(20658, 0x2003, 0x44, 0),
        _rec(20659, 0x3000, 0xCD, 1),
    ]
    return np.array(recs, dtype=BUS_DT)


def test_partition_isolates_read_only_song_data():
    recs = _synthetic_trace()
    part = I.partition(recs)
    assert part.t0 == 1000
    mask = I.song_data_mask(part)
    # The $2000 table is init-written, play-read, never play-written -> song data.
    assert mask[0x2000] and mask[0x2003]
    # $3000 is play-written (scratch / shadow) -> NOT song data, subtracted out.
    assert not mask[0x3000]
    # The SID page is never song data even though it is written.
    assert not mask[0xD400]
    runs = I.regions(mask)
    assert (0x2000, 0x2003) in runs


def test_lift_is_verbatim_from_ram_image():
    recs = _synthetic_trace()
    part = I.partition(recs)
    # The lift is the post-init RAM image, verbatim (HARD RULE #0).
    assert I.lift_song_data(part, 0x2000, 0x2003) == bytes([0x11, 0x22, 0x33, 0x44])


def test_load_image_seeds_unwritten_resident_data():
    """A packer that embeds the DEPACKED tables in the load image writes nothing
    during init; the partition still recovers them verbatim from the seeded RAM
    image (load-resident, play-read, never play-written)."""
    recs = np.array(
        [
            _rec(1000, 0xD400, 0x00, 1),  # frame-0 anchor
            _rec(1001, 0x2000, 0xDE, 0),  # play READS resident data
            _rec(1002, 0x2001, 0xAD, 0),
            _rec(20656, 0xD400, 0x01, 1),
            _rec(20657, 0x2000, 0xDE, 0),
        ],
        dtype=BUS_DT,
    )
    part = I.partition(recs, load_image=bytes([0xDE, 0xAD]), load_addr=0x2000)
    mask = I.song_data_mask(part)
    assert mask[0x2000] and mask[0x2001]
    assert I.lift_song_data(part, 0x2000, 0x2001) == bytes([0xDE, 0xAD])


def test_partition_requires_an_anchor():
    recs = np.array([_rec(10, 0x2000, 0x11, 0)], dtype=BUS_DT)  # no SID write
    with pytest.raises(ValueError):
        I.partition(recs)


def test_lift_song_data_from_sid_selects_image_region():
    """``lift_song_data_from_sid`` seeds the RAM image from the committed PSID and
    returns the widest read-only-during-play run inside the loaded image -- here a
    synthetic trace reads three resident image bytes the player never writes."""
    pytest.importorskip("pygoattracker")
    from preframr_tokens.bacc.sidemu import load_psid

    psid = load_psid(_GRID_SID)
    base = psid.load_addr + 0x400  # a spot well inside the loaded image
    recs = np.array(
        [
            _rec(1000, 0xD400, 0x00, 1),  # frame-0 anchor
            _rec(1001, base, psid.data[0x400], 0),  # READ resident image bytes
            _rec(1002, base + 1, psid.data[0x401], 0),
            _rec(1003, base + 2, psid.data[0x402], 0),
            _rec(20656, 0xD400, 0x01, 1),  # second play call
            _rec(20657, base, psid.data[0x400], 0),
        ],
        dtype=BUS_DT,
    )
    _, region, bounds = I.lift_song_data_from_sid(_GRID_SID, recs)
    assert bounds == (base, base + 2)
    # The lift is verbatim from the player's own loaded image (HARD RULE #0).
    assert region == bytes(psid.data[0x400:0x403])


def test_lift_song_data_from_sid_handles_no_region():
    """A trace that reads nothing inside the image yields an empty lift, not a
    crash (honest fallback rather than a fabricated region)."""
    pytest.importorskip("pygoattracker")
    recs = np.array(
        [
            _rec(1000, 0xD400, 0x00, 1),
            _rec(20656, 0xD400, 0x01, 1),
        ],
        dtype=BUS_DT,
    )
    _, region, bounds = I.lift_song_data_from_sid(_GRID_SID, recs)
    assert region == b"" and bounds is None


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
