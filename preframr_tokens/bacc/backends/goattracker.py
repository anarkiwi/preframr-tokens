"""GoatTracker 2 driver backend.

The per-tune PROGRAM is the GoatTracker song (orderlists + patterns + instruments
+ the four tables); the VM is pygoattracker's playroutine. recover() reconstructs
the song from a packed GoatTracker SID (inverting the gt2reloc packer, layout
auto-derived -- no per-tune hand disassembly) and carries it as the abstract
``Song`` object (the common rows + instrument-generators + orderlist structure,
NOT raw .SNG bytes) plus the per-SID render parameters baked into the packed
player (adparam, the pulse/realtime skip optimizations). render() runs the song
back through pygoattracker, aligned to the dump's frame grid.

The pulse-high (3, 10, 17) and filter (21, 23) registers are masked to the bits
the dump keeps (vsiddump.reduce_res), so the render compares byte-exact to the
register state the codec reconstructs with per_frame_state.
"""

import numpy as np

from preframr_tokens.bacc.backends.base import DriverBackend
from preframr_tokens.bacc.primitive import BaccProgram

NAME = "goattracker"
# Slack frames rendered ahead of nframes so the dump-alignment search can drop
# the player's leading init/hard-restart frames (the dump's first_play_cycle
# starts a few frames into playback).
_ALIGN_SLACK = 32


def _mask_row(reg):
    """Match vsiddump.reduce_res: PW-high low nibble, filter cutoff-lo 3 bits,
    clear filter external bit3."""
    reg[3] &= 0x0F
    reg[10] &= 0x0F
    reg[17] &= 0x0F
    reg[21] &= 0x07
    reg[23] &= 0xF7
    return reg


def render_song(song_or_bytes, seed, nframes):
    """Render a GoatTracker song to (nframes, 25), aligned 1:1 with the dump via
    the boot frames (set by recover_program). Accepts a pygoattracker ``Song``
    (the abstract recovered program) or, for back-compat, raw .SNG bytes."""
    from pygoattracker import read_sng

    from preframr_tokens.bacc.backends import gt_unpack

    if isinstance(song_or_bytes, (bytes, bytearray, list)):
        song = read_sng(bytes(song_or_bytes))
    else:
        song = song_or_bytes
    rendered = gt_unpack.render_state(
        song,
        nframes + _ALIGN_SLACK,
        adparam=seed["adparam"],
        optimize_pulse=bool(seed["optimize_pulse"]),
        optimize_realtime=bool(seed["optimize_realtime"]),
        subtune=int(seed["subtune"]),
    )
    out = np.array([_mask_row(list(row)) for row in rendered], dtype=np.int64)
    offset = _align_offset(out, seed.get("boot"), seed.get("boot1"))
    return out[offset : offset + nframes]


def _align_offset(rendered, boot, boot1):
    """Leading frames to drop so rendered[offset:] matches the dump grid: the
    first index whose frame (and the next) equals the dump's boot frames."""
    if boot is None:
        return 0
    boot = list(boot)
    for offset in range(min(_ALIGN_SLACK, len(rendered) - 1)):
        if list(rendered[offset]) == boot and (
            boot1 is None or list(rendered[offset + 1]) == list(boot1)
        ):
            return offset
    return 0


def make_program_from_song(song, seed, nframes):
    """Wrap a reconstructed GoatTracker ``Song`` (the abstract program -- rows +
    instrument-generators + orderlist) as a BaccProgram. No .SNG bytes are stored;
    the song object IS the recovered structure, rendered back via pygoattracker."""
    return BaccProgram(
        driver=NAME,
        nframes=nframes,
        boot=[],
        instruments=[],
        score=[],
        seed=dict(seed),
        tables={"song": song},
    )


def make_program(sng_bytes, seed, nframes):
    """Back-compat helper: wrap reconstructed .SNG bytes as a BaccProgram by
    parsing them into the abstract ``Song`` (no bytes are retained)."""
    from pygoattracker import read_sng

    return make_program_from_song(read_sng(bytes(sng_bytes)), seed, nframes)


class GoatTrackerBackend(DriverBackend):
    name = NAME

    def matches(self, psid):
        """A GoatTracker-packed PSID (gt2reloc single-speed): play = init + 3, and
        the relocated player opens with two JMPs (init -> initsong, play ->
        playroutine), so the image begins 4C .. .. 4C .. .."""
        data = getattr(psid, "data", b"")
        return (
            psid.play_addr == psid.init_addr + 3
            and len(data) >= 6
            and data[0] == 0x4C
            and data[3] == 0x4C
        )

    def recover(self, psid, nframes, subtune):
        from preframr_tokens.bacc.backends import gt_unpack

        song = gt_unpack.reconstruct_song(psid.path)
        params = gt_unpack.render_params()
        seed = {"subtune": int(subtune), **params}
        return make_program_from_song(song, seed, nframes)

    def render(self, program):
        seed = dict(program.seed)
        seed["boot"] = program.boot or None
        seed["boot1"] = program.tables.get("boot1")
        return render_song(program.tables["song"], seed, program.nframes)
