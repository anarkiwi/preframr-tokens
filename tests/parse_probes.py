"""Torch-free parse-side probes shared by the deterministic encoding suite: build a
raw register dump, run it through the FULL ``RegLogParser.parse`` + block path (the
deployed token stream, not a hand-built df), and count ops. These replace the
model-side xpt op48 probes for the encoding-completeness checks."""

from collections import Counter
from types import SimpleNamespace

import pandas as pd

from preframr_tokens.macros.blocks import iter_self_contained_row_blocks
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.tokenizer_config import PARSER_DEFAULTS

DUMP_COLUMNS = ("clock", "irq", "chipno", "reg", "val")
PHI = 985248
FRAME_CYCLES = PHI // 50
_VOICE0_FREQ_LO = 0
_VOICE0_FREQ_HI = 1
_VOICE0_PW_LO = 2
_VOICE0_PW_HI = 3
_VOICE0_CTRL = 4
_VOICE0_AD = 5
_VOICE0_SR = 6
_FC_LO = 21
_FC_HI = 22
_RES_FILT = 23
_MODE_VOL = 24


def parse_args(**over):
    """A full parser args namespace (every macro flag present + off) with the freq
    family flags forced off so callers opt in to exactly one encoding config."""
    cfg = dict(PARSER_DEFAULTS)
    cfg.update(
        freq_trajectory_pass=False,
        trajectory_anchor_pass=False,
        loop_pass=False,
        loop_transposed=False,
    )
    cfg.update(over)
    return SimpleNamespace(**cfg)


def cents_to_fn(note, cents):
    """16-bit SID freq word for ``note`` semitones offset by sub-semitone ``cents``."""
    hz = 440.0 * 2 ** ((note + cents / 100.0 - 69) / 12.0)
    return max(0, min(0xFFFF, int(round(hz * 16777216.0 / PHI))))


class DumpBuilder:
    """Build a raw ``clock,irq,chipno,reg,val`` dump for voice 0 with SEPARATE lo+hi
    byte writes and per-frame freq writes, so the full parser runs ``_combine_regs``
    and (when skeleton is off) ``_quantize_freq_to_cents`` -- the stages a hand-built
    ``Pass.apply`` test skips. ``frame()`` advances the PAL clock one tick."""

    def __init__(self):
        self._rows = []
        self._clock = 0

    def _write(self, reg, val):
        self._rows.append((self._clock, self._clock, 0, int(reg), int(val) & 0xFF))

    def frame(self):
        self._clock += FRAME_CYCLES
        return self

    def freq(self, fn):
        self._write(_VOICE0_FREQ_LO, fn & 0xFF)
        self._write(_VOICE0_FREQ_HI, (fn >> 8) & 0xFF)
        return self

    def gate_on(self):
        self._write(_VOICE0_CTRL, 0x40)
        self._write(_VOICE0_CTRL, 0x41)
        return self

    def ctrl(self, val):
        """Write the voice-0 control register (waveform/gate/test bits) directly, so a test can
        place a noise/test frame mid-note."""
        self._write(_VOICE0_CTRL, val & 0xFF)
        return self

    def adsr(self, ad=0x00, sr=0xF0):
        self._write(_VOICE0_AD, ad)
        self._write(_VOICE0_SR, sr)
        return self

    def pw(self, value):
        self._write(_VOICE0_PW_LO, value & 0xFF)
        self._write(_VOICE0_PW_HI, (value >> 8) & 0x0F)
        return self

    def fc(self, value):
        """Write the global filter cutoff lo+hi (reg 21/22), the way a cutoff sweep pokes it."""
        self._write(_FC_LO, value & 0xFF)
        self._write(_FC_HI, (value >> 8) & 0xFF)
        return self

    def resfilt(self, value):
        """Write the global resonance/filter-routing register (reg 23, RES_FILT)."""
        self._write(_RES_FILT, value & 0xFF)
        return self

    def modevol(self, value):
        """Write the global filter-mode/master-volume register (reg 24, MODE_VOL)."""
        self._write(_MODE_VOL, value & 0xFF)
        return self

    def note(self, per_frame_fns, gate=True):
        """One note: gate-on at its onset then ``per_frame_fns`` 16-bit freqs, one per
        PAL frame (the per-frame freq write the held/arp/slide drivers emit)."""
        for i, fn in enumerate(per_frame_fns):
            self.frame()
            if i == 0 and gate:
                self.gate_on()
            self.freq(int(fn))
        return self

    def dataframe(self):
        self.frame()
        return pd.DataFrame(self._rows, columns=list(DUMP_COLUMNS))


def write_dump(builder, path):
    """Persist a builder's dump to ``path`` (a ``.dump.parquet`` the parser reads)."""
    builder.dataframe().to_parquet(path)
    return path


def block_op_counts(path, args):
    """Op histogram of the DEPLOYED token stream: parse ``path`` then run the freq
    block passes the tokenizer sees (op48 only fires here, not in the inline parse
    call, because it needs the op column the block path adds)."""
    parsed = next(
        RegLogParser(args=args).parse(path, max_perm=1, require_pq=False, reparse=True),
        None,
    )
    counts = Counter()
    if parsed is None:
        return counts
    for block in iter_self_contained_row_blocks(parsed, 999999, args=args):
        counts.update(int(o) for o in block["op"].to_numpy())
    return counts
