"""DMC (Demo Music Creator) v7.62 driver backend.

DMC by Balazs Farkas (Brian) / Graffity is the single most-used editor in the
HVSC (~10,700 tunes). This backend recovers a v7.62 ``$1000`` image into the
common BACC abstraction and renders it back byte-exact.

The recovered PROGRAM is the DMC SONG MODEL (per-voice orderlists + patterns with
notes on the canonical A440 grid + instrument records + the note table) parsed by
``dmc_format`` -- NOT raw image bytes. The VM is the DMC player engine itself: a
fixed engine template (the player code + its effect/wave/filter tables) re-run
under py65 (DMC uses no illegal opcodes, so the plain ``SIDEmu`` path suffices).

``render`` REBUILDS the song-data regions of the engine template from the recovered
model (``dmc_format.emit_song``) and re-runs init+play once per frame, so the render
is driven by the recovered abstraction, not a replay of the untouched image. The
player boots one frame ahead of the dump grid (siddump samples after play N); the
GoatTracker ``_align`` boot-frame logic absorbs that +1 offset verbatim.
"""

import numpy as np

from preframr_tokens.bacc import dmc_format as df
from preframr_tokens.bacc.backends.base import DriverBackend
from preframr_tokens.bacc.backends.goattracker import _align, _mask_row
from preframr_tokens.bacc.primitive import BaccProgram
from preframr_tokens.bacc.sidemu import NREG, PSID, SIDEmu

# Fingerprint: the v7.62 player embeds this author string in its image. The
# leading "762" sits in the cur_note overlay ($1012-$1014) so it is not literal
# in the image; the contiguous embedded text is "-PLAYER (C) BRIAN/GRAFFITY!-".
SIGNATURE = b"-PLAYER (C) BRIAN/GRAFFITY!-"


class DmcBackend(DriverBackend):
    """Run-the-player backend for DMC v7.62 ``$1000`` tunes."""

    name = "dmc"
    mask_state = staticmethod(lambda s: _mask(s))

    def matches(self, psid):
        """v7.62 ``$1000`` build: load/init ``$1000``, play ``$1003`` (= init + 3,
        the same shape as GoatTracker), disambiguated by the embedded
        ``-PLAYER (C) BRIAN/GRAFFITY!-`` signature, which makes it unambiguous
        (so this backend is tried ahead of the broad GoatTracker matcher)."""
        data = getattr(psid, "data", b"")
        return (
            psid.load_addr == 0x1000
            and psid.init_addr == 0x1000
            and psid.play_addr == 0x1003
            and SIGNATURE in bytes(data)
        )

    def recover(self, psid, nframes, subtune):
        """Parse the DMC song into the common abstraction.

        The ``seed`` carries the engine template (the image; render overwrites its
        song-data regions from the model, so the abstraction drives the render) and
        the load/init/play addresses. The musical structure (orderlists, patterns
        with grid notes, instruments, freq table) rides in ``tables['song']``.
        """
        mem = df.Mem(psid.load_addr, psid.data)
        song = df.parse_song(mem, subtune)
        # Blank the song-data regions in the stored template so the recovered model
        # is the SOLE source of the song data -- the template carries only the
        # player engine + its fixed effect/wave/filter tables (the VM). render
        # rebuilds the blanked regions from the model (no raw song bytes retained).
        template = bytearray(psid.data)
        for start, length in df.song_regions(song):
            off = start - psid.load_addr
            if 0 <= off and off + length <= len(template):
                for k in range(length):
                    template[off + k] = 0
        prog = BaccProgram(
            driver=self.name,
            nframes=nframes,
            boot=[],
            instruments=[],
            score=[],
            seed={
                "load_addr": psid.load_addr,
                "init_addr": psid.init_addr,
                "play_addr": psid.play_addr,
                "image": list(template),
            },
            tables={"song": song},
        )
        return prog

    def render(self, program):
        """Rebuild the image from the recovered model and re-run the player.

        The engine template (``seed['image']``) has its song-data regions
        overwritten by ``dmc_format.emit_song`` from the recovered song model, so
        every rendered register derives from the abstraction. init once, play per
        frame, snapshot ``$D400-$D418``; ``_align`` drops the +1 boot frame so the
        render lands 1:1 on the dump grid.
        """
        seed = program.seed
        load_addr = seed["load_addr"]
        image = bytearray(0x10000)
        image[load_addr : load_addr + len(seed["image"])] = bytes(seed["image"])
        df.emit_song(image, program.tables["song"])
        data = bytes(image[load_addr : load_addr + len(seed["image"])])
        psid = PSID(
            path="",
            version=2,
            init_addr=seed["init_addr"],
            play_addr=seed["play_addr"],
            songs=1,
            start_song=1,
            speed=0,
            flags=0,
            load_addr=load_addr,
            data=data,
        )
        emu = SIDEmu(psid)
        emu.init(program.tables["song"]["subtune"])
        slack = program.nframes + 8
        raw = np.zeros((slack, NREG), dtype=np.int64)
        raw[0] = _mask_row(list(emu.state()))
        for f in range(1, slack):
            emu.play_frame()
            raw[f] = _mask_row(list(emu.state()))
        boot = program.boot or None
        boot1 = program.tables.get("boot1")
        return _align(raw, boot, boot1, program.nframes)


def _mask(state):
    """Apply the standard PW-high / filter don't-care masks (``_mask_row``)."""
    s = np.asarray(state).copy()
    return np.array([_mask_row(list(row)) for row in s], dtype=s.dtype)
