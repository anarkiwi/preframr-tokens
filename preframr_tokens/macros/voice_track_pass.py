"""Cross-voice FREQ tracking macro, behind the ``voice_track_pass`` flag
(default OFF). REFUTED 2026-05-24: on cent-bin FREQ a musical interval is an
additive offset, not the ``round(lead*ratio)+detune`` multiplicative model here;
a 40-song headroom probe found zero >=10-frame tracking spans under either
model, so this stays off."""

__all__ = ["VoiceTrackPass"]

from preframr_tokens.macros.passes_base import (
    _first_irq,
    MacroPass,
    _ensure_subreg,
    _frame_index,
    _splice_rows,
)
from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    SET_OP,
    TRACK_INTERVAL_RATIOS,
    TRACK_MAX_DURATION,
    TRACK_MIN_DURATION,
    TRACK_REF_OP,
    TRACK_REF_SUBREG_DETUNE,
    TRACK_REF_SUBREG_DURATION,
    TRACK_REF_SUBREG_INTERVAL,
    TRACK_REF_SUBREG_LEAD,
    VOICES,
)

_FREQ_REGS = tuple(FREQ_REGS_BY_VOICE)


def _track_rows(tracker_reg, lead_voice, interval_id, detune, duration, diff, irq):
    fields = [
        (TRACK_REF_SUBREG_LEAD, int(lead_voice)),
        (TRACK_REF_SUBREG_INTERVAL, int(interval_id)),
        (TRACK_REF_SUBREG_DETUNE, int(detune) & 0xFF),
        (TRACK_REF_SUBREG_DURATION, min(int(duration), TRACK_MAX_DURATION)),
    ]
    return [
        {
            "reg": int(tracker_reg),
            "val": int(val),
            "diff": int(diff),
            "op": int(TRACK_REF_OP),
            "subreg": int(subreg),
            "irq": int(irq),
            "description": 0,
        }
        for subreg, val in fields
    ]


def _match_interval(lead_vals, tracker_vals):
    """Return ``(interval_id, detune)`` holding for all frames, else None."""
    for interval_id, ratio in enumerate(TRACK_INTERVAL_RATIOS):
        detune = tracker_vals[0] - round(lead_vals[0] * ratio)
        if not (-128 <= detune <= 127):
            continue
        if all(
            tv - round(lv * ratio) == detune for lv, tv in zip(lead_vals, tracker_vals)
        ):
            return interval_id, detune
    return None


class VoiceTrackPass(MacroPass):
    GATE_FLAGS = frozenset({"voice_track_pass"})

    def apply(self, df, args=None):
        if args is None or not getattr(args, "voice_track_pass", False):
            return df
        if df is None or len(df) == 0 or "op" not in df.columns:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        f_idx = _frame_index(df).to_numpy()
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        irq_default = _first_irq(df)
        n_frames = int(f_idx[-1]) + 1 if len(df) else 0

        freq_at = {
            v: self._forward_fill(v, regs, ops, subregs, vals, f_idx, n_frames)
            for v in range(VOICES)
        }
        tracker_sets = {
            v: self._tracker_sets(v, regs, ops, subregs, f_idx) for v in range(VOICES)
        }

        drop_idx = []
        new_rows = []
        for tracker in range(VOICES):
            self._collapse_voice(
                tracker,
                tracker_sets[tracker],
                freq_at,
                vals,
                diffs,
                irq_default,
                drop_idx,
                new_rows,
            )
        if not drop_idx:
            return df
        return _splice_rows(df, drop_idx, new_rows)

    @staticmethod
    def _forward_fill(voice, regs, ops, subregs, vals, f_idx, n_frames):
        reg = _FREQ_REGS[voice]
        per_frame = [None] * n_frames
        for i in range(len(regs)):
            if int(regs[i]) == reg and int(ops[i]) == SET_OP and int(subregs[i]) == -1:
                per_frame[int(f_idx[i])] = int(vals[i])
        cur = None
        for fr in range(n_frames):
            if per_frame[fr] is None:
                per_frame[fr] = cur
            else:
                cur = per_frame[fr]
        return per_frame

    @staticmethod
    def _tracker_sets(voice, regs, ops, subregs, f_idx):
        reg = _FREQ_REGS[voice]
        out = []
        for i in range(len(regs)):
            if int(regs[i]) == reg and int(ops[i]) == SET_OP and int(subregs[i]) == -1:
                out.append((int(f_idx[i]), i))
        return out

    def _collapse_voice(
        self, tracker, sets, freq_at, vals, diffs, irq_default, drop_idx, new_rows
    ):
        k = 0
        while k < len(sets):
            j = k
            while j + 1 < len(sets) and sets[j + 1][0] == sets[j][0] + 1:
                j += 1
            run = sets[k : j + 1]
            if len(run) >= TRACK_MIN_DURATION:
                self._emit_run(
                    tracker,
                    run,
                    freq_at,
                    vals,
                    diffs,
                    irq_default,
                    drop_idx,
                    new_rows,
                )
            k = j + 1

    def _emit_run(
        self, tracker, run, freq_at, vals, diffs, irq_default, drop_idx, new_rows
    ):
        frames = [fr for fr, _ in run]
        tracker_vals = [int(vals[i]) for _, i in run]
        for lead in range(VOICES):
            if lead == tracker:
                continue
            lead_vals = [freq_at[lead][fr] for fr in frames]
            if any(lv is None or lv <= 0 for lv in lead_vals):
                continue
            match = _match_interval(lead_vals, tracker_vals)
            if match is None:
                continue
            interval_id, detune = match
            first_idx = run[0][1]
            diff = int(diffs[first_idx]) if diffs is not None else 0
            atom = _track_rows(
                _FREQ_REGS[tracker],
                lead,
                interval_id,
                detune,
                len(run),
                diff,
                irq_default,
            )
            for nr in atom:
                nr["__pos"] = first_idx
            new_rows.extend(atom)
            drop_idx.extend(i for _, i in run)
            return
