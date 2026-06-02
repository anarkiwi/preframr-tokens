"""Torch-free parse-side probes shared by the deterministic encoding suite: build a
raw register dump, run it through the FULL ``RegLogParser.parse`` + block path (the
deployed token stream, not a hand-built df), count ops, and classify each ORN RESID
escape note by what its per-frame offsets ARE (fast-melodic-run / glissando / noise).
These replace the model-side xpt op48 probes for the encoding-completeness checks."""

from collections import Counter
from types import SimpleNamespace

import pandas as pd

from preframr_tokens.macros.blocks import iter_self_contained_row_blocks
from preframr_tokens.macros.skeleton_pass import ARP_MAX_PERIOD
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    ORN_OP,
    ORN_SUBREG_P1,
    ORN_SUBREG_P2,
    ORN_SUBREG_TYPE,
    ORN_TYPE_RESID,
    SKEL_OP,
)
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


def parse_args(**over):
    """A full parser args namespace (every macro flag present + off) with the freq
    family flags forced off so callers opt in to exactly one encoding config."""
    cfg = dict(PARSER_DEFAULTS)
    cfg.update(
        skeleton_pass=False,
        freq_trajectory_pass=False,
        freq_onset_pass=False,
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


def inline_orn_notes(path, args):
    """Yield ``(orn_type, offsets)`` per note from the SINGLE-encode parsed df (skeleton
    applied once inline at parse time), NOT the block path -- the block path decodes to
    the content floor then re-encodes, collapsing the lossy VIB primitive to PLAIN, so
    mechanism-classification fixtures must read the once-encoded inline stream."""
    parsed = next(
        RegLogParser(args=args).parse(path, max_perm=1, require_pq=False, reparse=True),
        None,
    )
    if parsed is None or "op" not in parsed.columns:
        return
    yield from _iter_orn_notes(parsed)


def inline_note_signature(path, args):
    """Per-note ``((skel_subreg, skel_val), orn_type, offsets)`` from the single-encode inline
    stream -- the provenance-invariance comparison key (#11.4): the exact SKEL+ORN tokens a note
    encodes to, so two register renderings of one gesture can be asserted identical."""
    parsed = next(
        RegLogParser(args=args).parse(path, max_perm=1, require_pq=False, reparse=True),
        None,
    )
    if parsed is None or "op" not in parsed.columns:
        return []
    ops = parsed["op"].to_numpy()
    subs = parsed["subreg"].to_numpy()
    vals = parsed["val"].to_numpy()
    n = len(parsed)
    out = []
    pending = None
    i = 0
    while i < n:
        if int(ops[i]) == SKEL_OP:
            pending = (int(subs[i]), int(vals[i]))
            i += 1
            continue
        if int(ops[i]) == ORN_OP and int(subs[i]) == ORN_SUBREG_TYPE:
            orn_type = int(vals[i])
            offs = []
            j = i + 1
            while j < n and int(ops[j]) == ORN_OP and int(subs[j]) != ORN_SUBREG_TYPE:
                if int(subs[j]) == ORN_SUBREG_P1:
                    v = int(vals[j])
                    offs.append(v - 256 if v > 127 else v)
                j += 1
            out.append((pending, orn_type, tuple(offs)))
            pending = None
            i = j
            continue
        i += 1
    return out


def _iter_orn_notes(block):
    """Yield ``(orn_type, offsets)`` per note in one block: each ORN descriptor is a
    TYPE atom followed by its P1/P2 parameter atoms (RESID carries one signed P1
    offset per frame)."""
    ops = block["op"].to_numpy()
    subs = block["subreg"].to_numpy()
    vals = block["val"].to_numpy()
    n = len(block)
    i = 0
    while i < n:
        if int(ops[i]) == ORN_OP and int(subs[i]) == ORN_SUBREG_TYPE:
            orn_type = int(vals[i])
            offs = []
            j = i + 1
            while j < n and int(ops[j]) == ORN_OP and int(subs[j]) != ORN_SUBREG_TYPE:
                if int(subs[j]) == ORN_SUBREG_P1:
                    v = int(vals[j])
                    offs.append(v - 256 if v > 127 else v)
                j += 1
            yield orn_type, offs
            i = j
            continue
        i += 1


def skeleton_orn_summary(path, args):
    """Parse ``path`` under a skeleton config and return per-note ORN-channel counts:
    SKEL atoms, total ORN descriptors, and how many escaped to RESID, over the whole
    deployed block stream."""
    parsed = next(
        RegLogParser(args=args).parse(path, max_perm=1, require_pq=False, reparse=True),
        None,
    )
    skel = orn = resid = 0
    if parsed is None:
        return dict(skel=0, orn=0, resid=0)
    for block in iter_self_contained_row_blocks(parsed, 999999, args=args):
        skel += int((block["op"].to_numpy() == SKEL_OP).sum())
        for orn_type, _offs in _iter_orn_notes(block):
            orn += 1
            if orn_type == ORN_TYPE_RESID:
                resid += 1
    return dict(skel=skel, orn=orn, resid=resid)


def classify_resid(offsets):
    """Bucket one RESID note's per-frame note-relative offsets into the cause of the
    leak: a short low-cardinality run is fast-melodic-run (under-segmentation,
    recoverable as notes); a monotone wide ramp is a long-glissando/sweep (legit
    RESID); a long repeating cycle is a periodic-long-arp; else aperiodic-noise."""
    n = len(offsets)
    if n == 0:
        return "empty"
    distinct = len(set(offsets))
    span = max(offsets) - min(offsets)
    diffs = [b - a for a, b in zip(offsets, offsets[1:])]
    monotone = bool(diffs) and (
        all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)
    )
    if monotone and span >= 3 and n >= 4:
        return "long-glissando/sweep"
    for period in range(ARP_MAX_PERIOD + 1, n // 2 + 1):
        if all(offsets[i] == offsets[i % period] for i in range(n)):
            return "periodic-long-arp"
    if distinct >= 6 or span >= 12:
        return "aperiodic-noise/wide"
    return "fast-melodic-run"


def resid_breakdown(path, args):
    """Classify every RESID note in the deployed stream by note count and frame
    share, for the known-real-tune RESID-gap characterization."""
    parsed = next(
        RegLogParser(args=args).parse(path, max_perm=1, require_pq=False, reparse=True),
        None,
    )
    by_note = Counter()
    by_frame = Counter()
    resid_frames = nonresid_frames = 0
    if parsed is None:
        return dict(by_note={}, by_frame={}, resid_frame_share=0.0)
    for block in iter_self_contained_row_blocks(parsed, 999999, args=args):
        for orn_type, offs in _iter_orn_notes(block):
            frames = max(len(offs), 1)
            if orn_type == ORN_TYPE_RESID:
                cat = classify_resid(offs)
                by_note[cat] += 1
                by_frame[cat] += frames
                resid_frames += frames
            else:
                nonresid_frames += frames
    total = resid_frames + nonresid_frames
    return dict(
        by_note=dict(by_note.most_common()),
        by_frame=dict(by_frame.most_common()),
        resid_frame_share=resid_frames / total if total else 0.0,
    )
