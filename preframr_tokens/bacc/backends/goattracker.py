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
# the player's leading init/hard-restart frames. The dump's first_play_cycle can
# start well into playback (the player spends its first frames ramping ADSR/init
# from cold), so the render needs a window wide enough to reach the dump's frame
# 0 -- empirically up to a couple hundred frames on some tunes, not just a few.
_ALIGN_SLACK = 256


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
    try:
        rendered = gt_unpack.render_state(
            song,
            nframes + _ALIGN_SLACK,
            adparam=seed["adparam"],
            optimize_pulse=bool(seed["optimize_pulse"]),
            optimize_realtime=bool(seed["optimize_realtime"]),
            subtune=int(seed["subtune"]),
        )
    except IndexError as exc:
        # A recovered wave/pulse/filter table pointer walked off the end of its
        # (MAX_TABLELEN-padded) table without hitting a TABLEJUMP terminator, so
        # the playroutine wrapped its pointer to 0 and indexed past the table.
        # This is a structural reconstruction edge, not a rendering glitch we can
        # paper over (HARD RULE #0: no fabricated frames); surface it precisely
        # instead of leaking a bare IndexError out of pygoattracker.
        raise RuntimeError(
            "goattracker render: a recovered wave/pulse/filter table pointer "
            "overran its table (unterminated table -- the playroutine wrapped "
            "its pointer past the table end). The reconstructed song is not "
            "byte-exact-renderable; reported precisely rather than rendering "
            "fabricated frames."
        ) from exc
    out = np.array([_mask_row(list(row)) for row in rendered], dtype=np.int64)
    return _align(out, seed.get("boot"), seed.get("boot1"), nframes)


def _align(rendered, boot, boot1, nframes):
    """Align the raw render to the dump's frame grid and return exactly ``nframes``.

    Two boot framings occur in the wild (both lossless, no stored correction):

    * The render contains the dump's frame-0 boot frame after some leading
      init/hard-restart frames -- drop those: ``rendered[off:]`` where
      ``rendered[off]==boot`` (and, when known, ``rendered[off+1]==boot1``).
    * The dump captured a leading all-zero **silence** frame that the player's
      render never emits (the render starts straight at ``boot1``). The render
      then has no frame equal to ``boot``; PREPEND the dump's frame-0 silence to
      the render aligned at ``boot1``. ``boot`` is literally ``state[0]`` from the
      dump, so prepending it reproduces frame 0 byte-for-byte; the residual check
      verifies the remainder.
    """
    off = _align_offset(rendered, boot, boot1)
    if off is not None:
        return rendered[off : off + nframes]
    # Leading-silence dump: render begins at boot1; restore the dropped frame 0.
    off1 = _boot1_offset(rendered, boot1)
    if off1 is not None and boot is not None:
        prefix = np.array([list(boot)], dtype=rendered.dtype)
        return np.concatenate([prefix, rendered[off1:]])[:nframes]
    return rendered[:nframes]


def _align_offset(rendered, boot, boot1):
    """First index whose frame (and the next, when ``boot1`` is known) equals the
    dump's boot frames, else ``None`` (so a leading-silence render can be handled
    by prepending the dump's frame 0)."""
    if boot is None:
        return 0
    boot = list(boot)
    for offset in range(min(_ALIGN_SLACK, len(rendered) - 1)):
        if list(rendered[offset]) == boot and (
            boot1 is None or list(rendered[offset + 1]) == list(boot1)
        ):
            return offset
    return None


def _boot1_offset(rendered, boot1):
    """First index whose frame equals ``boot1`` -- the start of playback when the
    dump's frame 0 is a leading silence frame the render does not reproduce."""
    if boot1 is None:
        return None
    boot1 = list(boot1)
    for offset in range(min(_ALIGN_SLACK, len(rendered))):
        if list(rendered[offset]) == boot1:
            return offset
    return None


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
