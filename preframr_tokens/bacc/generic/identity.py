"""Recover the song-data tables by IDENTITY -- read off the player's own RAM.

This is Step 0 of the white-box "recover the program, not its output" pipeline
(``design/encoding/sidtrace_program_recovery.md``).  The inverse-problem recovery
in :mod:`recover` is handed only the SID-write OUTPUT and must invert it to a
program; under-constrained, it tends toward a clever RLE of the trace (Grid_Runner:
~1000 output-similarity instrument clusters for a song the player indexes with 14
real instruments).

The reads are ALREADY in the ``.bus.bin`` (the ``rw == 0`` records carry the
addresses the player reads INTO its own song-data tables).  By the player contract
(HVSC SID format: init populates the data once; play only reads it), the RAM region
written-once-at-init / load-resident and then ONLY READ during play **is** the song
data, recoverable byte-for-byte from RAM (HARD RULE #0: genuine program data, never
fabricated, never re-fit from output).

:func:`partition` computes, per address, the init/play write-sets and the play
read-set over the trace; :func:`song_data_mask` is the write-set/read-set
partition (init-written-or-load-resident, play-read, never play-written -- the
self-mod / SID-shadow / scratch addresses subtract out); :func:`lift_song_data`
lifts those bytes verbatim from the post-init RAM image (load image for the bytes
no write ever touched, last-init-write for the rest).  No emulator patch -- this
exploits the read stream already on disk (P0 in the design doc).
"""

from dataclasses import dataclass

import numpy as np

from preframr_tokens.bacc.generic.busstate import sid_writes
from preframr_tokens.bacc.generic.bustrace import load_bus
from preframr_tokens.codec.lsp_validate import detect_play_period, first_play_cycle

# IO / SID / CIA pages are never song DATA -- they are memory-mapped hardware the
# player pokes and polls every frame.  Song data lives in low RAM (the loaded
# image is well under $D000); excluding the IO page keeps the partition to RAM.
_IO_LO = 0xD000
_RAM_LO = 0x0002  # skip the zeropage CPU port ($00/$01)
_CHUNK = 20_000_000  # bus records per streaming pass (the traces are multi-GB)


@dataclass
class Partition:
    """Per-address access summary over a bus trace, split at the init->play
    boundary.  Each field is a 65536-long boolean mask indexed by address."""

    init_written: np.ndarray  # written during init (cyc < t0)
    play_written: np.ndarray  # written during play (cyc >= t0)
    play_read: np.ndarray  # read during play (cyc >= t0)
    ram_image: np.ndarray  # uint8[65536] post-init RAM (last init write wins)
    t0: int  # frame-0 anchor cycle (first steady play call)


def play_anchor(records):
    """The frame-0 anchor cycle ``t0`` (first steady play call) derived from the
    bus's OWN SID-write cadence -- the init->play phase boundary without any
    emulator phase tag (Step 0 has no ``BUS_DT2``)."""
    cyc, _, _ = sid_writes(records)
    if len(cyc) == 0:
        return None
    return int(first_play_cycle(cyc, detect_play_period(cyc)))


def partition(records, t0=None, load_image=None, load_addr=None):
    """Compute the init/play write/read partition over a bus trace.

    ``records`` is a native ``.bus.bin`` path or a pre-loaded :data:`BUS_DT`
    record array.  ``t0`` overrides the derived frame-0 anchor (the init->play
    boundary).  ``load_image``/``load_addr`` seed the post-init RAM image with the
    PSID load image so bytes that NO write ever touches (load-resident song data --
    the common case for a packer that embeds the depacked tables) are still
    recovered verbatim; init writes then overwrite them where the player rewrites
    data on init.

    Streams the (multi-GB) trace in chunks so the partition builds in seconds with
    bounded memory.  Returns a :class:`Partition`.
    """
    recs = records if isinstance(records, np.ndarray) else load_bus(records)
    if t0 is None:
        t0 = play_anchor(recs)
    if t0 is None:
        raise ValueError("bus trace has no SID writes; cannot anchor init/play")

    ram = np.zeros(65536, dtype=np.uint8)
    if load_image is not None and load_addr is not None:
        img = np.frombuffer(bytes(load_image), dtype=np.uint8)
        end = min(load_addr + len(img), 65536)
        ram[load_addr:end] = img[: end - load_addr]

    init_w = np.zeros(65536, dtype=bool)
    play_w = np.zeros(65536, dtype=bool)
    play_r = np.zeros(65536, dtype=bool)

    n = len(recs)
    for start in range(0, n, _CHUNK):
        chunk = recs[start : start + _CHUNK]
        cyc, addr, val, rw = chunk["cyc"], chunk["addr"], chunk["val"], chunk["rw"]
        is_init = cyc < t0
        iw = is_init & (rw == 1)
        a_iw = addr[iw]
        if len(a_iw):
            ram[a_iw] = val[iw]  # ordered stream -> last init write wins per addr
            init_w[a_iw] = True
        play = ~is_init
        a_pw = addr[play & (rw == 1)]
        if len(a_pw):
            play_w[a_pw] = True
        a_pr = addr[play & (rw == 0)]
        if len(a_pr):
            play_r[a_pr] = True

    return Partition(init_w, play_w, play_r, ram, int(t0))


def song_data_mask(part):
    """The write-set/read-set partition: addresses that are SONG DATA.

    Song data = RAM (not the IO page), READ during play, and NEVER WRITTEN during
    play.  Subtracting the play-written addresses drops the SID shadow file, the
    per-voice scratch, and self-modified code operands (a ``play`` write to an
    address means it is mutable state, not read-only data -- the §4.1 / SMC
    caveat).  A load-resident table need not be init-written, so init-written is
    NOT required -- only "read during play, never written during play".
    """
    in_ram = np.zeros(65536, dtype=bool)
    in_ram[_RAM_LO:_IO_LO] = True
    return part.play_read & (~part.play_written) & in_ram


def regions(mask):
    """Contiguous ``(lo, hi)`` byte ranges (inclusive) of a boolean address mask."""
    idx = np.nonzero(mask)[0]
    if not len(idx):
        return []
    out = []
    start = prev = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        out.append((start, prev))
        start = prev = value
    out.append((start, prev))
    return out


def lift_song_data(part, lo, hi):
    """Lift the song-data bytes for the (inclusive) address range ``[lo, hi]``
    VERBATIM from the post-init RAM image -- genuine program data, read off the
    player's own RAM, never fabricated (HARD RULE #0).  Returns ``bytes``."""
    return bytes(part.ram_image[lo : hi + 1])


def lift_song_data_from_sid(sid_path, records, t0=None):
    """Convenience: lift the contiguous song-data region spanning the PSID load
    image from a tune's bus trace.

    Seeds the RAM image with the PSID load image (so load-resident tables are
    recovered even when init rewrites nothing), partitions, and returns
    ``(part, region_bytes, (lo, hi))`` for the song-data region that overlaps the
    loaded image -- the largest contiguous read-only-during-play run inside
    ``[load_addr, load_end)``.
    """
    from preframr_tokens.bacc.sidemu import load_psid  # local: optional GT dep

    psid = load_psid(sid_path)
    recs = records if isinstance(records, np.ndarray) else load_bus(records)
    part = partition(recs, t0=t0, load_image=psid.data, load_addr=psid.load_addr)
    mask = song_data_mask(part)
    load_end = psid.load_addr + len(psid.data)
    # Restrict to the loaded image and take the widest contiguous read-only run.
    in_img = np.zeros(65536, dtype=bool)
    in_img[psid.load_addr : load_end] = True
    runs = regions(mask & in_img)
    if not runs:
        return part, b"", None
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    return part, lift_song_data(part, lo, hi), (lo, hi)
