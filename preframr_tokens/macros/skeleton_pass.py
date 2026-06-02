"""SkeletonPass + ornament channel (Stage 1+2 unified pitch): segment each freq reg into
NOTES (semitone-run + ``MIN_HOLD`` UNION gate-on), emit one ``SKEL`` atom per note (LUT
index; abs first per reg, signed interval after) plus one driver-native, constant-size-per-note
``ORN`` descriptor (PLAIN / OCTAVE+ARP period-cycle+length / SLIDE target+rate / VIB depth+rate
/ RESID raw-offset escape). Content-tier (semitone floor); opt-in."""

__all__ = [
    "SkeletonPass",
    "LUT",
    "fn_to_note_resid",
    "midi_to_fn",
    "CLOCK_RATE",
    "CENTS_THRESHOLD",
    "MIN_HOLD",
    "is_fast_melodic_run",
    "fit_descriptor",
    "vib_frame_offsets",
    "slide_frame_offsets",
    "slide2_frame_offsets",
    "cycle_frame_offsets",
]

import math
from bisect import bisect_right
from collections import Counter

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    _frame_index,
    make_row,
    MacroPass,
)
from preframr_tokens.macros.rle import run_length_encode
from preframr_tokens.stfconstants import (
    ARP_CYCLE_MAX_STEPS,
    ARP_CYCLE_MIN_REPEAT,
    FREQ_TRAJ_REGS,
    ORN_OP,
    ORN_SUBREG_HOLD,
    ORN_SUBREG_P1,
    ORN_SUBREG_P2,
    ORN_SUBREG_TYPE,
    ORN_TYPE_ARP,
    ORN_TYPE_HELD_ARP,
    ORN_TYPE_OCTAVE,
    ORN_TYPE_PLAIN,
    ORN_TYPE_RESID,
    ORN_TYPE_SLIDE,
    ORN_TYPE_SLIDE2,
    ORN_TYPE_VIB,
    SLIDE2_MAX_DUR,
    SET_OP,
    SKEL_OP,
    SKEL_SUBREG_ABS,
    SKEL_SUBREG_INTERVAL,
    VOICE_CTRL_REG,
    VOICE_REG_SIZE,
)

CLOCK_RATE = 985248
MIDI_LO, MIDI_HI = 16, 112
CENTS_THRESHOLD = 8.0
MIN_HOLD = 3
LEVELCHANGE_CAP = 12
LEVELCHANGE_HOLD = 2
VIB_MIN_CENTS = 8.0
ARP_MAX_DISTINCT = 4
ARP_MAX_PERIOD = 16
_OFFSET_LIMIT = 24
_VIB_DEPTH_HEAVY_CENTS = 30.0
_VIB_RATE_DEFAULT = 6
_SKELETON_PRIORITY = 0
_CTRL_FOR_FREQ = {
    int(reg): int(VOICE_CTRL_REG[reg // VOICE_REG_SIZE]) for reg in FREQ_TRAJ_REGS
}


def midi_to_fn(m):
    """MIDI note -> 16-bit SID freq word, clamped to 0..0xFFFF."""
    return max(
        0,
        min(
            0xFFFF, int(round(440.0 * 2 ** ((m - 69) / 12.0) * 16777216.0 / CLOCK_RATE))
        ),
    )


LUT = [midi_to_fn(m) for m in range(128)]


def fn_to_note_resid(fn):
    """16-bit freq -> (nearest MIDI semitone, residual in cents). None if silent/out of range."""
    if fn < 8:
        return None
    hz = fn * CLOCK_RATE / 16777216.0
    if hz < 16:
        return None
    mf = 69 + 12 * math.log2(hz / 440.0)
    note = int(round(mf))
    if not MIDI_LO <= note <= MIDI_HI:
        return None
    return note, (mf - note) * 100.0


def _vib_depth(resids):
    """Per-note sub-semitone vibrato depth bucket (0 none / 1 light / 2 heavy) from the cents
    amplitude of the note's on-semitone writes (the held-note wobble)."""
    near = [c for c in resids if abs(c) < 50]
    if len(near) < 3:
        return 0
    amp = (max(near) - min(near)) / 2.0
    return 0 if amp < VIB_MIN_CENTS else (1 if amp <= _VIB_DEPTH_HEAVY_CENTS else 2)


def cycle_frame_offsets(period, length):
    """Replay a note-relative offset CYCLE: ``period[k % len(period)]`` for ``length`` frames
    (OCTAVE/ARP). The minimal repeating period makes this constant-size, not per-frame.
    """
    if not period or length <= 0:
        return []
    return [int(period[k % len(period)]) for k in range(length)]


def slide_frame_offsets(target, rate, length):
    """Replay a portamento as a per-frame note-relative offset ramp: one semitone step per
    ``rate`` frames toward ``target``, clamped, for ``length`` frames (target+rate, not per
    frame)."""
    if length <= 0 or rate <= 0:
        return []
    sign = 1 if target >= 0 else -1
    mag = abs(int(target))
    return [sign * min(mag, (k + 1) // rate) for k in range(length)]


def slide2_frame_offsets(target, duration, length):
    """Replay an exact-landing portamento: a linear note-relative ramp that reaches ``target`` after
    ``duration`` frames (rounded) then holds, for ``length`` frames. Unlike the rate-only SLIDE this
    lands on any target in a given duration (a non-unit per-frame delta), not just unit steps.
    """
    if length <= 0 or duration <= 0:
        return []
    return [int(round(target * min(k + 1, duration) / duration)) for k in range(length)]


def vib_frame_offsets(depth, rate, length):
    """Replay vibrato at the semitone floor: a depth/rate oscillator that stays within a
    semitone, so its per-frame note-relative offset is 0 (the content-tier floor drops the
    sub-semitone wobble). ``depth``/``rate`` carry the learnable wobble signal, not bytes.
    """
    del depth, rate
    if length <= 0:
        return []
    return [0] * length


def _minimal_period(offs):
    """Shortest prefix (<= ARP_MAX_PERIOD) that genuinely REPEATS to reproduce ``offs`` (a
    constant offset, or a cycle seen at least twice), or None — so a one-shot non-repeating run
    is residual, not a spurious whole-sequence 'period'."""
    n = len(offs)
    for p in range(1, min(ARP_MAX_PERIOD, n) + 1):
        if (p == 1 or n >= 2 * p) and all(offs[i] == offs[i % p] for i in range(n)):
            return offs[:p]
    return None


def _rle(seq):
    """Run-length encode a sequence into (values, holds) of consecutive duplicates."""
    pairs = run_length_encode(seq)
    return [v for v, _ in pairs], [h for _, h in pairs]


def held_cycle(target):
    """A held-step wavetable arp at the content floor: RLE-collapse the per-frame offset ``target``,
    find the smallest VALUE period (2..ARP_CYCLE_MAX_STEPS, repeated >= ARP_CYCLE_MIN_REPEAT times);
    holds need not be periodic (a note tail extends its last step), so they are carried per-step.
    Returns (period_offsets, holds) -- replaying period[k%p] held holds[k] reproduces ``target``
    exactly -- or None. Lets the cycle survive holds that blow past ARP_MAX_PERIOD in frame space.
    """
    vals, holds = _rle(target)
    if holds and max(holds) > 255:
        return None
    n = len(vals)
    for p in range(2, min(ARP_CYCLE_MAX_STEPS, n // ARP_CYCLE_MIN_REPEAT) + 1):
        if all(vals[k] == vals[k % p] for k in range(n)):
            return tuple(vals[:p]), tuple(holds)
    return None


def held_cycle_offsets(period, holds):
    """Expand a held-ARP (period offsets, per-step holds) back to one offset per frame."""
    out = []
    p = len(period)
    for k, hold in enumerate(holds):
        out.extend([period[k % p]] * int(hold))
    return out


def _slide_rate(offs, target):
    """Frames-per-semitone-step that reproduces a monotone ramp toward ``target``, or None. A SLIDE's
    first nonzero offset is at frame ``rate-1``, so the only candidate rate is fixed by the leading
    run of zeros -- derive it and verify once (the brute-force smallest-matching rate is identical).
    """
    first_nz = next((k for k, o in enumerate(offs) if o != 0), None)
    if first_nz is None:
        return 1 if offs and target == 0 else None
    rate = first_nz + 1
    return rate if slide_frame_offsets(target, rate, len(offs)) == offs else None


def _slide_descriptor(offs):
    """``(target, rate)`` for a monotone ramp an exact rate-only SLIDE reproduces, target within a
    signed byte, else None -- the note-relative SWEEP survivor a wide-ramp SLIDE routes off RESID.
    """
    if len(offs) < 2:
        return None
    diffs = [b - a for a, b in zip(offs, offs[1:])]
    if not (all(x >= 0 for x in diffs) or all(x <= 0 for x in diffs)):
        return None
    if abs(offs[-1] - offs[0]) < 2 or not -128 <= offs[-1] <= 127:
        return None
    rate = _slide_rate(offs, offs[-1])
    return (offs[-1], rate) if rate is not None else None


def _slide2_descriptor(offs):
    """``(target, duration)`` for a monotone ramp the exact-landing SLIDE reproduces (a constant
    per-frame delta the rate-only form can't express, e.g. ``[2,4,6,8]``), target/duration each within
    a signed byte, else None. ``duration`` = the frame the ramp first reaches its final offset.
    """
    if len(offs) < 2:
        return None
    diffs = [b - a for a, b in zip(offs, offs[1:])]
    if not (all(x >= 0 for x in diffs) or all(x <= 0 for x in diffs)):
        return None
    target = offs[-1]
    if abs(target - offs[0]) < 2 or not -128 <= target <= 127:
        return None
    duration = next((k + 1 for k, o in enumerate(offs) if o == target), len(offs))
    if not 1 <= duration <= SLIDE2_MAX_DUR:
        return None
    if slide2_frame_offsets(target, duration, len(offs)) == offs:
        return (target, duration)
    return None


def is_fast_melodic_run(offs):
    """True when a would-be RESID note's note-relative offsets are a short (distinct<6, span<12),
    non-periodic, non-monotone run of distinct semitones -- under-segmented constituent notes
    recoverable by splitting, NOT a wide glissando, long-period arp, or aperiodic noise. The
    action-side mirror of the parse-probe ``classify_resid`` 'fast-melodic-run' bucket, the
    dominant real-tune RESID source (#13)."""
    n = len(offs)
    if n == 0:
        return False
    distinct = len(set(offs))
    if distinct < 2:
        return False
    span = max(offs) - min(offs)
    diffs = [b - a for a, b in zip(offs, offs[1:])]
    monotone = bool(diffs) and (
        all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)
    )
    if monotone and span >= 3 and n >= 4:
        return False
    for period in range(ARP_MAX_PERIOD + 1, n // 2 + 1):
        if all(offs[i] == offs[i % period] for i in range(n)):
            return False
    return distinct < 6 and span < 12


def fit_descriptor(base, seg_fns, slide_wide=False, slide_landing=False):
    """Classify a note's intra-note freq writes (16-bit, frame order) into one driver-native,
    constant-size ornament. ``base`` = the note semitone, ``seg_fns`` = settled freqs after the onset.
    ``slide_wide`` routes a wide monotone ramp to rate-only SLIDE; ``slide_landing`` routes a constant-
    delta ramp the rate-only form misses to the exact-landing SLIDE2 (W4/W5). Returns (orn_type, params).
    """
    if not seg_fns:
        return ORN_TYPE_PLAIN, ()
    resolved = [fn_to_note_resid(fn) for fn in seg_fns]
    if any(r is None for r in resolved):
        return ORN_TYPE_RESID, tuple(0 for _ in seg_fns)
    offs = [r[0] - base for r in resolved]
    resids = [r[1] for r in resolved]
    nonzero = [o for o in offs if o != 0]
    if not nonzero:
        depth = _vib_depth(resids)
        return (
            (ORN_TYPE_VIB, (depth, _VIB_RATE_DEFAULT))
            if depth
            else (ORN_TYPE_PLAIN, ())
        )
    if any(abs(o) > _OFFSET_LIMIT for o in offs):
        if slide_wide:
            slide = _slide_descriptor(offs)
            if slide is not None:
                return ORN_TYPE_SLIDE, slide
        if slide_landing:
            slide2 = _slide2_descriptor(offs)
            if slide2 is not None:
                return ORN_TYPE_SLIDE2, slide2
        return ORN_TYPE_RESID, tuple(offs)
    diffs = [b - a for a, b in zip(offs, offs[1:])]
    monotone = all(x >= 0 for x in diffs) or all(x <= 0 for x in diffs)
    if monotone and abs(offs[-1] - offs[0]) >= 2:
        rate = _slide_rate(offs, offs[-1])
        if rate is not None:
            return ORN_TYPE_SLIDE, (offs[-1], rate)
        if slide_landing:
            slide2 = _slide2_descriptor(offs)
            if slide2 is not None:
                return ORN_TYPE_SLIDE2, slide2
    period = _minimal_period(offs)
    if period is not None:
        is_octave = set(period) <= {0, 12} or set(period) <= {0, -12}
        return (ORN_TYPE_OCTAVE if is_octave else ORN_TYPE_ARP), tuple(period)
    return ORN_TYPE_RESID, tuple(offs)


def _row(reg, op, subreg, val, diff, irq):
    return make_row(reg, val, op=op, subreg=subreg, diff=diff, irq=irq)


class SkeletonPass(MacroPass):
    """Dense skeleton + ornament: segment each freq reg into notes (semitone-run + MIN_HOLD UNION
    gate-on, then held-gate de-merge of giant-RESID phrases into their constituent notes), emit one
    SKEL atom per note and one ORN descriptor collapsing its intra-note arps/vibrato/slide.
    Requires ``freq_trajectory_pass`` / ``freq_onset_pass`` OFF (skeleton owns the freq channel).
    """

    GATE_FLAGS = frozenset(
        {"skeleton_pass", "held_arp", "zero_plain", "slide_wide", "slide_landing"}
    )

    _held_arp = False  # noqa: per-parse args.held_arp gate (set in apply)
    _zero_plain = False  # noqa: per-parse args.zero_plain gate (set in apply)
    _slide_wide = False  # noqa: per-parse args.slide_wide gate (set in apply)
    _slide_landing = False  # noqa: per-parse args.slide_landing gate (set in apply)
    _resid_diag = None  # noqa: inert RESID-trace sink; None=off (prod). See design/resid_archetype_program.md
    _df_sink = (
        None  # noqa: inert raw-df sink for drum-footprint probes; None=off (prod).
    )

    def apply(self, df, args=None):
        if args is None or not getattr(args, "skeleton_pass", False):
            return df
        if df is None or len(df) == 0:
            return df
        SkeletonPass._held_arp = bool(getattr(args, "held_arp", False))
        SkeletonPass._zero_plain = bool(getattr(args, "zero_plain", False))
        SkeletonPass._slide_wide = bool(getattr(args, "slide_wide", False))
        SkeletonPass._slide_landing = bool(getattr(args, "slide_landing", False))
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        if "traj_anchor" in df.columns:
            df = df.drop(columns=["traj_anchor"])
        ctx = self._context(df)
        if SkeletonPass._df_sink is not None:
            SkeletonPass._df_sink.append(df.assign(_fr=_frame_index(df)))
        irq = _first_irq(df)
        drop_idx = []
        new_rows = []
        for reg in FREQ_TRAJ_REGS:
            self._claim_reg(reg, ctx, irq, drop_idx, new_rows)
        if not new_rows:
            return df
        return arbitrate(
            df,
            [
                Claim(
                    writes=tuple(drop_idx),
                    tokens=new_rows,
                    priority=_SKELETON_PRIORITY,
                    label="skeleton",
                )
            ],
        )

    @staticmethod
    def _context(df):
        """Per-row arrays + per-reg ordered (frame, row_index, val, diff) freq SETs, per-voice
        gate-on rising-edge frames (ctrl bit0), and per-voice ordered ctrl writes (frame, ctrl_val)
        so each freq frame's waveform/test/gate state is recoverable -- noise (bit7) and test (bit3)
        mark a frame's freq as timbre/transient, not melodic pitch (control-aware encoding).
        """
        f_idx = _frame_index(df).to_numpy()
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        freq_sets = {int(reg): [] for reg in FREQ_TRAJ_REGS}
        gate_on = {int(reg): set() for reg in FREQ_TRAJ_REGS}
        ctrl_writes = {int(reg): [] for reg in FREQ_TRAJ_REGS}
        gate_state = {int(c): 0 for c in _CTRL_FOR_FREQ.values()}
        ctrl_to_freq = {int(c): int(f) for f, c in _CTRL_FOR_FREQ.items()}
        for i in range(len(df)):
            reg = int(regs[i])
            if reg in freq_sets and int(ops[i]) == SET_OP and int(subregs[i]) == -1:
                freq_sets[reg].append(
                    (
                        int(f_idx[i]),
                        int(i),
                        int(vals[i]),
                        int(diffs[i]) if diffs is not None else 0,
                    )
                )
            elif reg in gate_state and int(ops[i]) == SET_OP and int(subregs[i]) == -1:
                v = int(vals[i])
                ctrl_writes[ctrl_to_freq[reg]].append((int(f_idx[i]), v))
                g = v & 1
                if g and not gate_state[reg]:
                    gate_on[ctrl_to_freq[reg]].add(int(f_idx[i]))
                gate_state[reg] = g
        return freq_sets, gate_on, ctrl_writes

    @staticmethod
    def _ctrl_at(ctrl_writes, frame):
        """Forward-filled ctrl byte in effect at ``frame`` (last write at or before it), or None.
        ``ctrl_writes`` is frame-ascending, so binary-search the split point (the ``inf`` sentinel
        sorts past any same-frame value) instead of scanning -- the hot per-frame pitched-frame test.
        """
        idx = bisect_right(ctrl_writes, (frame, float("inf")))
        return ctrl_writes[idx - 1][1] if idx else None

    @classmethod
    def _is_pitched_frame(cls, ctrl_writes, frame):
        """A frame carries a melodic PITCH only if its ctrl is a pitched waveform (tri/saw/pulse,
        bits 4-6) with the TEST bit (3) clear. Noise (bit7) is timbre; test/HR frames are transient;
        an unknown (pre-first-ctrl) frame is treated as pitched (no evidence against).
        """
        ctrl = cls._ctrl_at(ctrl_writes, frame)
        if ctrl is None:
            return True
        if ctrl & 0x08:
            return False
        return bool(ctrl & 0x70) and not (ctrl & 0x80)

    def _claim_reg(self, reg, ctx, irq, drop_idx, new_rows):
        freq_sets, gate_on, ctrl_writes = ctx
        cwrites = ctrl_writes[int(reg)]
        sets = freq_sets[int(reg)]
        if not sets:
            return
        by_frame = {s[0]: s for s in sets}
        onsets = self._segment_notes(sets, gate_on[int(reg)])
        onsets = self._resegment_holdgate(onsets, sets, by_frame)
        onsets = self._resegment_fast_run(onsets, sets, by_frame)
        onsets = self._resegment_levelchange(onsets, sets, by_frame, cwrites)
        if len(onsets) < 2:
            return
        onset_frames = [fr for fr, _ in onsets]
        end_frame = max(s[0] for s in sets) + 1
        prev_note = None
        for k, (onset_fr, note) in enumerate(onsets):
            nxt_fr = onset_frames[k + 1] if k + 1 < len(onset_frames) else end_frame
            anchor = by_frame.get(onset_fr)
            if anchor is None:
                continue
            seg = [s for s in sets if onset_fr < s[0] < nxt_fr]
            per_frame = self._per_frame_fns(note, anchor, seg, onset_fr, nxt_fr)
            note = self._rebased_note(note, anchor, per_frame, onset_fr, cwrites)
            orn_type, params = fit_descriptor(
                note,
                per_frame,
                SkeletonPass._slide_wide,
                SkeletonPass._slide_landing,
            )
            claimed = self._emit_note(
                reg, note, anchor, per_frame, orn_type, params, prev_note, irq, new_rows
            )
            if not claimed:
                continue
            if SkeletonPass._resid_diag is not None:
                is_resid = orn_type == ORN_TYPE_RESID
                rec = None
                if is_resid:
                    rec = []
                    for k, fn in enumerate(per_frame):
                        res = fn_to_note_resid(int(fn))
                        if res is None:
                            continue
                        fr = onset_fr + 1 + k
                        ctrl = self._ctrl_at(cwrites, fr)
                        rec.append(
                            (
                                res[0] - note,
                                -1 if ctrl is None else int(ctrl),
                                self._is_pitched_frame(cwrites, fr),
                                int(fn),
                            )
                        )
                SkeletonPass._resid_diag.append(
                    (int(reg), is_resid, int(note), int(onset_fr), rec)
                )
            prev_note = note
            drop_idx.append(anchor[1])
            drop_idx.extend(s[1] for s in seg)

    @classmethod
    def _resegment_holdgate(cls, onsets, sets, by_frame):
        """Held-gate re-segmentation: a giant RESID note (a held >= MIN_HOLD plateau followed by a
        melody that was not re-gated) is split at its first post-plateau moving frame into the
        plateau note + the trailing melody as its own note, recursively, so a held-gate phrase
        de-merges into its constituent notes. Onsets list of (frame, note) -> refined list.
        """
        if len(onsets) < 1:
            return onsets
        end_frame = max(s[0] for s in sets) + 1
        result = []
        queue = list(onsets)
        while queue:
            onset_fr, note = queue.pop(0)
            nxt_fr = queue[0][0] if queue else end_frame
            anchor = by_frame.get(onset_fr)
            if anchor is None:
                result.append((onset_fr, note))
                continue
            seg = [s for s in sets if onset_fr < s[0] < nxt_fr]
            per_frame = cls._per_frame_fns(note, anchor, seg, onset_fr, nxt_fr)
            hold = cls._split_holdgate_resid(note, per_frame)
            if hold <= 0:
                result.append((onset_fr, note))
                continue
            split_fr = onset_fr + hold + 1
            split_anchor = by_frame.get(split_fr)
            if split_anchor is None:
                result.append((onset_fr, note))
                continue
            res = fn_to_note_resid(int(split_anchor[2]))
            if res is None:
                result.append((onset_fr, note))
                continue
            result.append((onset_fr, note))
            queue.insert(0, (split_fr, res[0]))
        return sorted(set(result))

    @classmethod
    def _rebased_note(cls, note, anchor, per_frame, onset_fr, ctrl_writes):
        """Base the note on its sustained PITCHED pitch (control-aware): count semitones only over
        pitched frames (noise = timbre, test/HR = transient -- not a melodic pitch, e.g. Facemorph's
        onset noise-tik at freq~note107), and re-base when the onset semitone is a rare outlier (<=2
        pitched frames) yet another is a >=50% majority of the pitched span. Folds base-misassignment
        into the right note; the ``<=2`` guard leaves clean arps (many onset-pitch frames) untouched.
        """
        onset_ctrl = cls._ctrl_at(ctrl_writes, onset_fr)
        if onset_ctrl is not None and (onset_ctrl & 0x80) and not (onset_ctrl & 0x08):
            return note
        counts = Counter()
        for k, fn in enumerate([anchor[2], *per_frame]):
            if not cls._is_pitched_frame(ctrl_writes, onset_fr + k):
                continue
            res = fn_to_note_resid(int(fn))
            if res is not None:
                counts[res[0]] += 1
        if not counts:
            return note
        dom, dom_n = counts.most_common(1)[0]
        if (
            dom != note
            and counts.get(note, 0) <= 2
            and dom_n > counts.get(note, 0)
            and dom_n * 2 >= sum(counts.values())
        ):
            return dom
        return note

    @staticmethod
    def _per_frame_fns(note, anchor, seg, onset_fr, nxt_fr):
        """Forward-filled per-frame settled freq for frames AFTER the onset up to the next note
        (held frames inherit the last write). Position 0 is the onset frame itself (= anchor
        value, owned by the SKEL atom); the ornament replays positions 1..end byte-exactly.
        """
        span = max(0, nxt_fr - onset_fr - 1)
        if span == 0:
            return []
        seg_at = {s[0] - onset_fr: int(s[2]) for s in seg}
        out = []
        cur = int(anchor[2])
        for k in range(1, span + 1):
            cur = seg_at.get(k, cur)
            out.append(cur)
        return out

    @staticmethod
    def _segment_notes(sets, gate_frames):
        """Note onsets = semitone level-change that HOLDS >= MIN_HOLD frames (so fast arp steps
        are not notes) UNION gate-on frames. Hold duration is measured by frame span (robust to
        per-frame duplicate writes from the block-expand path). Returns sorted [(frame, note)].
        """
        resolved = []
        for s in sets:
            r = fn_to_note_resid(s[2])
            if r is not None:
                resolved.append((s[0], r[0]))
        runs = []
        for fr, note in resolved:
            if runs and runs[-1][1] == note:
                runs[-1] = (runs[-1][0], note, fr)
            else:
                runs.append((fr, note, fr))
        onsets = {}
        for i, (start, note, last) in enumerate(runs):
            end = runs[i + 1][0] if i + 1 < len(runs) else last + MIN_HOLD
            if (end - start) >= MIN_HOLD:
                onsets[start] = note
        fn_at = {fr: note for fr, note in resolved}
        for fr in gate_frames:
            if fr in fn_at:
                onsets.setdefault(fr, fn_at[fr])
        return sorted(onsets.items())

    @staticmethod
    def _split_holdgate_resid(note, per_frame):
        """Held-gate de-merge: a RESID note that opens with a stable >= MIN_HOLD plateau on its own
        semitone is a clean held note followed by a SEPARATE note that was not re-gated (Hubbard
        note-flag bit6 "appended, no attack"); return the plateau length so the caller cuts a new
        onset at the first moving frame, else 0 (clean ornament / no leading plateau / no tail).
        """
        hold = 0
        for fn in per_frame:
            res = fn_to_note_resid(int(fn))
            if res is None or res[0] != note:
                break
            hold += 1
        if hold < MIN_HOLD or hold >= len(per_frame):
            return 0
        orn_type, params = fit_descriptor(note, list(per_frame))
        if orn_type != ORN_TYPE_RESID:
            return 0
        del params
        return hold

    @classmethod
    def _resegment_fast_run(cls, onsets, sets, by_frame):
        """Fast-melodic-run de-merge (#13): a note that ``fit_descriptor`` would leak to RESID and
        whose frames are a non-periodic run of distinct clean semitones (each held < MIN_HOLD, so
        ``_segment_notes`` missed it) is split into one note per semitone step. Genuine ornaments
        (ARP/SLIDE/VIB/OCTAVE), glissandi and wide/aperiodic noise are left untouched.
        Onsets [(frame, note)] -> refined list."""
        if len(onsets) < 1:
            return onsets
        end_frame = max(s[0] for s in sets) + 1
        result = []
        queue = list(onsets)
        while queue:
            onset_fr, note = queue.pop(0)
            nxt_fr = queue[0][0] if queue else end_frame
            anchor = by_frame.get(onset_fr)
            if anchor is None:
                result.append((onset_fr, note))
                continue
            seg = [s for s in sets if onset_fr < s[0] < nxt_fr]
            per_frame = cls._per_frame_fns(note, anchor, seg, onset_fr, nxt_fr)
            orn_type, _ = fit_descriptor(note, per_frame)
            if orn_type != ORN_TYPE_RESID:
                result.append((onset_fr, note))
                continue
            resolved = [fn_to_note_resid(int(fn)) for fn in per_frame]
            if not resolved or any(r is None for r in resolved):
                result.append((onset_fr, note))
                continue
            if not is_fast_melodic_run([r[0] - note for r in resolved]):
                result.append((onset_fr, note))
                continue
            splits = cls._run_split_onsets(onset_fr, note, seg)
            if len(splits) < 2:
                result.append((onset_fr, note))
                continue
            result.append(splits[0])
            for extra in reversed(splits[1:]):
                queue.insert(0, extra)
        return sorted(set(result))

    @staticmethod
    def _run_split_onsets(onset_fr, note, seg):
        """Onsets at each real SET frame in ``seg`` where the settled semitone changes, starting
        from the existing ``(onset_fr, note)``. Last write wins per frame; unresolvable writes
        are skipped (they carry no clean semitone to anchor a note on)."""
        by_fr = {}
        for s in sorted(seg):
            by_fr[int(s[0])] = int(s[2])
        onsets = [(onset_fr, note)]
        cur = note
        for fr in sorted(by_fr):
            res = fn_to_note_resid(by_fr[fr])
            if res is None:
                continue
            if res[0] != cur:
                onsets.append((fr, res[0]))
                cur = res[0]
        return onsets

    @classmethod
    def _resegment_levelchange(cls, onsets, sets, by_frame, ctrl_writes):
        """Control-aware held-level-change de-merge: a held-gate RESID note merging several
        SUSTAINED constituent notes (each holds >= LEVELCHANGE_HOLD pitched frames, within
        LEVELCHANGE_CAP of its predecessor) splits at those held pitched changes -- noise/test
        transparent (not snapped), committed only when it cuts RESID. Recovers held-gate
        compound phrases without splitting glissandi/fast-arps or forging giant intervals (#13).
        """
        if len(onsets) < 1:
            return onsets
        end_frame = max(s[0] for s in sets) + 1
        result = []
        queue = list(onsets)
        while queue:
            onset_fr, note = queue.pop(0)
            nxt_fr = queue[0][0] if queue else end_frame
            anchor = by_frame.get(onset_fr)
            if anchor is None:
                result.append((onset_fr, note))
                continue
            seg = [s for s in sets if onset_fr < s[0] < nxt_fr]
            per_frame = cls._per_frame_fns(note, anchor, seg, onset_fr, nxt_fr)
            orn_type, _ = fit_descriptor(note, per_frame)
            if orn_type != ORN_TYPE_RESID:
                result.append((onset_fr, note))
                continue
            splits = cls._levelchange_onsets(onset_fr, note, seg, ctrl_writes)
            if len(splits) < 2 or not cls._split_cuts_resid(
                splits, sets, by_frame, nxt_fr, len(per_frame)
            ):
                result.append((onset_fr, note))
                continue
            result.append(splits[0])
            for extra in reversed(splits[1:]):
                queue.insert(0, extra)
        return sorted(set(result))

    @classmethod
    def _levelchange_onsets(cls, onset_fr, note, seg, ctrl_writes):
        """Onsets at HELD pitched-semitone changes within LEVELCHANGE_CAP of the running base
        (the change frame plus the next LEVELCHANGE_HOLD-1 pitched frames all settle on the new
        level); noise/test frames are transparent. Starts from the existing (onset_fr, note).
        """
        by_fr = {int(s[0]): int(s[2]) for s in sorted(seg)}
        psem = {}
        for fr in sorted(by_fr):
            if cls._is_pitched_frame(ctrl_writes, fr):
                r = fn_to_note_resid(by_fr[fr])
                psem[fr] = r[0] if r is not None else None
            else:
                psem[fr] = None
        pframes = [fr for fr in sorted(by_fr) if psem[fr] is not None]
        onsets = [(onset_fr, note)]
        cur = note
        for idx, fr in enumerate(pframes):
            level = psem[fr]
            if level == cur or abs(level - cur) > LEVELCHANGE_CAP:
                continue
            fut = [psem[f] for f in pframes[idx : idx + LEVELCHANGE_HOLD]]
            if len(fut) >= LEVELCHANGE_HOLD and all(s == level for s in fut):
                onsets.append((fr, level))
                cur = level
        return onsets

    @classmethod
    def _split_cuts_resid(cls, splits, sets, by_frame, final_nxt, orig_frames):
        """Re-fit each candidate piece; commit only when the split recovers >=1 clean note and
        leaves at most ONE RESID piece, so the RESID note-share can only improve (one RESID note
        -> clean notes + <=1 RESID fragment, count never rises while ORN rises) -- never trading
        a giant RESID note for several smaller RESID fragments."""
        del orig_frames
        bounds = [fr for fr, _ in splits] + [final_nxt]
        resid_pieces = clean_pieces = 0
        for i, (ofr, note) in enumerate(splits):
            anchor = by_frame.get(ofr)
            if anchor is None:
                return False
            seg = [s for s in sets if ofr < s[0] < bounds[i + 1]]
            per_frame = cls._per_frame_fns(note, anchor, seg, ofr, bounds[i + 1])
            orn_type, _ = fit_descriptor(note, per_frame)
            if orn_type == ORN_TYPE_RESID:
                resid_pieces += 1
            else:
                clean_pieces += 1
        return clean_pieces >= 1 and resid_pieces <= 1

    def _emit_note(
        self, reg, note, anchor, per_frame, orn_type, params, prev_note, irq, new_rows
    ):
        """Emit the SKEL atom (abs/interval) for one note followed by its ORN descriptor.
        Returns False (claim nothing) if the skeleton interval overflows a signed byte.
        """
        skel = self._skel_row(reg, note, anchor, prev_note, irq)
        if skel is None:
            return False
        orn = self._orn_rows(reg, note, per_frame, orn_type, params, anchor[3], irq)
        skel["__pos"] = anchor[1]
        new_rows.append(skel)
        for row in orn:
            row["__pos"] = anchor[1]
            new_rows.append(row)
        return True

    @staticmethod
    def _skel_row(reg, note, anchor, prev_note, irq):
        if prev_note is None:
            return _row(reg, SKEL_OP, SKEL_SUBREG_ABS, note, anchor[3], irq)
        interval = note - prev_note
        if not -128 <= interval <= 127:
            return None
        return _row(reg, SKEL_OP, SKEL_SUBREG_INTERVAL, interval & 0xFF, anchor[3], irq)

    @staticmethod
    def _is_transient_blip(target):
        """A held note (>=4 frames) at its base with <=2 ISOLATED non-zero outlier frames (each
        flanked by base frames) is a clean note + a brief attack/grace transient (#16). The content
        floor absorbs the blip -> PLAIN, instead of leaking the whole note to RESID. Isolation
        (neighbours == base) keeps a >=2-frame ornament (arp/slide start) from being mistaken for a
        transient."""
        n = len(target)
        if n < 4:
            return False
        nz = [i for i, t in enumerate(target) if t != 0]
        if not nz or len(nz) > 2:
            return False
        return all(
            (i == 0 or target[i - 1] == 0) and (i == n - 1 or target[i + 1] == 0)
            for i in nz
        )

    @staticmethod
    def _snap_offsets(note, per_frame):
        """Per-frame note-relative semitone offset (content-tier floor) for the held frames,
        clamped to a signed byte. Unresolvable (silent/out-of-range) frames hold the note.
        """
        out = []
        for fn in per_frame:
            res = fn_to_note_resid(int(fn))
            off = (res[0] - note) if res is not None else 0
            out.append(max(-128, min(127, int(off))))
        return out

    @classmethod
    def _orn_rows(cls, reg, note, per_frame, orn_type, params, diff, irq):
        """Build the driver-native constant-size ORN descriptor: a TYPE atom, then this type's
        small parameter list as signed P1 atoms (cycle period / slide target+rate / vib
        depth+rate), terminated by a P2 length atom. RESID escapes to one signed P1 offset per
        frame (semitone floor) after a P2 count. Verifies parametric replay matches the floor;
        else RESID."""
        length = min(len(per_frame), 0xFFFF)
        target = cls._snap_offsets(note, per_frame)
        if cls._is_transient_blip(target):
            orn_type, params = ORN_TYPE_PLAIN, ()
        elif (
            orn_type != ORN_TYPE_RESID
            and cls._reconstruct(orn_type, params, length) != target
        ):
            orn_type, params = ORN_TYPE_RESID, tuple(target)
        if (
            orn_type == ORN_TYPE_RESID
            and cls._zero_plain
            and target
            and not any(target)
        ):
            orn_type, params = ORN_TYPE_PLAIN, ()
        if orn_type == ORN_TYPE_RESID and cls._held_arp:
            hc = held_cycle(target)
            if hc is not None and held_cycle_offsets(hc[0], hc[1]) == target:
                orn_type, params = ORN_TYPE_HELD_ARP, hc
        if orn_type == ORN_TYPE_HELD_ARP:
            period, holds = params
            rows = [_row(reg, ORN_OP, ORN_SUBREG_TYPE, ORN_TYPE_HELD_ARP, diff, irq)]
            rows.extend(
                _row(reg, ORN_OP, ORN_SUBREG_P1, int(o) & 0xFF, diff, irq)
                for o in period
            )
            rows.extend(
                _row(reg, ORN_OP, ORN_SUBREG_HOLD, int(h), diff, irq) for h in holds
            )
            rows.append(_row(reg, ORN_OP, ORN_SUBREG_P2, length, diff, irq))
            return rows
        if orn_type == ORN_TYPE_PLAIN:
            return [_row(reg, ORN_OP, ORN_SUBREG_TYPE, ORN_TYPE_PLAIN, diff, irq)]
        if orn_type == ORN_TYPE_RESID:
            rows = [
                _row(reg, ORN_OP, ORN_SUBREG_TYPE, ORN_TYPE_RESID, diff, irq),
                _row(reg, ORN_OP, ORN_SUBREG_P2, length, diff, irq),
            ]
            rows.extend(
                _row(reg, ORN_OP, ORN_SUBREG_P1, int(off) & 0xFF, diff, irq)
                for off in target[:length]
            )
            return rows
        rows = [_row(reg, ORN_OP, ORN_SUBREG_TYPE, orn_type, diff, irq)]
        rows.extend(
            _row(reg, ORN_OP, ORN_SUBREG_P1, int(p) & 0xFF, diff, irq) for p in params
        )
        rows.append(_row(reg, ORN_OP, ORN_SUBREG_P2, length, diff, irq))
        return rows

    @staticmethod
    def _reconstruct(orn_type, params, length):
        """Per-frame note-relative offsets a non-RESID descriptor replays, for verification
        against the semitone floor at encode time (the same math OrnamentDecoder runs).
        """
        if orn_type == ORN_TYPE_PLAIN:
            return [0] * length
        if orn_type in (ORN_TYPE_OCTAVE, ORN_TYPE_ARP):
            return cycle_frame_offsets(params, length)
        if orn_type == ORN_TYPE_HELD_ARP:
            return held_cycle_offsets(params[0], params[1])
        if orn_type == ORN_TYPE_SLIDE:
            return slide_frame_offsets(params[0], params[1], length)
        if orn_type == ORN_TYPE_SLIDE2:
            return slide2_frame_offsets(params[0], params[1], length)
        if orn_type == ORN_TYPE_VIB:
            return vib_frame_offsets(params[0], params[1], length)
        return None
