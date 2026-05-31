"""WavetablePass: the pitched twin of StampPass (design/wavetable_codebook_encoding). The dominant
wavetable engines drive pitch ornament from a recurring per-frame note-relative offset program; this
post-skeleton proposer mines the SkeletonPass ORN-RESID dumps into an inline-redefinable WAVETABLE_DEF +
per-note WAVETABLE_REF codebook (held-ARP generalised to a cross-note loop), byte-identically replaying
the content-floor RESID it replaces or falling back to RESID. Opt-in (``wavetable_pass``), default OFF.
"""

__all__ = ["WavetablePass"]

from collections import defaultdict

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    _frame_index,
    MacroPass,
)
from preframr_tokens.macros.rle import run_length_encode
from preframr_tokens.macros.skeleton_pass import SkeletonPass
from preframr_tokens.macros.wavetable import factorise, program_key, unroll
from preframr_tokens.stfconstants import (
    FREQ_TRAJ_REGS,
    ORN_OP,
    ORN_SUBREG_P1,
    ORN_SUBREG_P2,
    ORN_SUBREG_TYPE,
    ORN_TYPE_RESID,
    WAVETABLE_DEF_OP,
    WAVETABLE_END_OP,
    WAVETABLE_ONESHOT_OP,
    WAVETABLE_REF_OP,
    WAVETABLE_STEP_OP,
    WT_ONESHOT_SUBREG_END,
    WT_ONESHOT_SUBREG_HOLD,
    WT_ONESHOT_SUBREG_LEN_HI,
    WT_ONESHOT_SUBREG_LEN_LO,
    WT_ONESHOT_SUBREG_OFFSET,
    WT_MINREP,
    WT_REF_SUBREG_ID,
    WT_REF_SUBREG_LEAD,
    WT_REF_SUBREG_LEADOFF,
    WT_REF_SUBREG_LEN_HI,
    WT_REF_SUBREG_LEN_LO,
    WT_SHORT_MAX,
    WT_STEP_SUBREG_HOLD,
    WT_STEP_SUBREG_LOOP,
    WT_STEP_SUBREG_OFFSET,
)

_WAVETABLE_PRIORITY = -20
_MIN_CORE = 2


def _row(reg, op, subreg, val, diff, irq):
    return {
        "reg": int(reg),
        "val": int(val),
        "diff": int(diff),
        "op": int(op),
        "subreg": int(subreg),
        "irq": int(irq),
        "description": 0,
    }


class WavetablePass(MacroPass):
    """Mine recurring note-relative offset programs from the skeleton ORN-RESID dumps and replace each
    with an inline WAVETABLE_DEF + WAVETABLE_REF (or an inline-structured one-shot), consuming the RESID
    atom rows and leaving the SKEL atom. Default OFF."""

    GATE_FLAGS = frozenset({"wavetable_pass", "wt_short", "wt_oneshot"})

    def apply(self, df, args=None):
        if args is None or not getattr(args, "wavetable_pass", False):
            return df
        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            return df
        irq = _first_irq(df)
        short_max = WT_SHORT_MAX if getattr(args, "wt_short", False) else 0
        oneshot = bool(getattr(args, "wt_oneshot", False))
        records = self._collect_resid(df, short_max, oneshot)
        if not records:
            return df
        programs = self._build_codebook(records, oneshot)
        drop_idx, new_rows = self._emit(records, programs, irq, oneshot)
        if not new_rows:
            return df
        return arbitrate(
            df,
            [
                Claim(
                    writes=tuple(drop_idx),
                    tokens=new_rows,
                    priority=_WAVETABLE_PRIORITY,
                    label="wavetable",
                )
            ],
        )

    @classmethod
    def _collect_resid(cls, df, short_max=0, oneshot=False):
        """Walk the post-skeleton df and pull each contiguous ORN-RESID atom (TYPE/P2/P1*) per freq
        reg into a record with its note-relative offsets, onset frame, and consumed row indices.
        ``short_max`` routes short residue to the literal codebook; ``oneshot`` keeps every residue
        note (even with no pitched core) so the inline one-shot can store it verbatim.
        """
        freq_regs = {int(r) for r in FREQ_TRAJ_REGS}
        ctrl_writes = SkeletonPass._context(df)[2]
        frames = _frame_index(df).to_numpy()
        ops = df["op"].to_numpy()
        regs = df["reg"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        idx = df.index.to_numpy()
        n = len(df)
        records = []
        i = 0
        while i < n:
            is_resid_head = (
                int(ops[i]) == ORN_OP
                and int(subregs[i]) == ORN_SUBREG_TYPE
                and int(vals[i]) == ORN_TYPE_RESID
                and int(regs[i]) in freq_regs
            )
            if not is_resid_head:
                i += 1
                continue
            reg = int(regs[i])
            onset_fr = int(frames[i])
            rows = [int(idx[i])]
            length = None
            offsets = []
            j = i + 1
            while j < n and int(ops[j]) == ORN_OP and int(regs[j]) == reg:
                rows.append(int(idx[j]))
                if int(subregs[j]) == ORN_SUBREG_P2:
                    length = int(vals[j]) & 0xFFFF
                elif int(subregs[j]) == ORN_SUBREG_P1:
                    v = int(vals[j]) & 0xFF
                    offsets.append(v if v < 128 else v - 256)
                j += 1
            i = j
            if length is None or len(offsets) != length or length < 1:
                continue
            rec = cls._make_record(
                reg, onset_fr, offsets, rows, ctrl_writes, short_max, oneshot
            )
            if rec is not None:
                records.append(rec)
        return records

    @staticmethod
    def _make_record(
        reg, onset_fr, offsets, rows, ctrl_writes, short_max=0, oneshot=False
    ):
        """Onset-strip the leading non-pitched (HR/test/noise) frames into a per-hit lead, leaving the
        pitched core the codebook keys on. Short residue (``len <= short_max``) keys on the verbatim
        offsets instead (no strip, no pitched-core gate). A note with no pitched core is kept (codebook
        ineligible) only when ``oneshot`` -- the inline one-shot stores its offsets verbatim.
        """
        base = {
            "reg": reg,
            "pos": rows[0],
            "rows": rows,
            "offsets": offsets,
            "length": len(offsets),
            "wt_id": None,
        }
        if 0 < short_max and len(offsets) <= short_max:
            base.update(lead=[], core=offsets, literal=True, codebook=True)
            return base
        cw = ctrl_writes.get(reg, [])
        lead = 0
        for k in range(len(offsets)):
            if SkeletonPass._is_pitched_frame(cw, onset_fr + 1 + k):
                break
            lead += 1
        core = offsets[lead:]
        if len(core) >= _MIN_CORE:
            base.update(lead=offsets[:lead], core=core, literal=False, codebook=True)
            return base
        if oneshot:
            base.update(lead=[], core=offsets, literal=False, codebook=False)
            return base
        return None

    @classmethod
    def _build_codebook(cls, records, oneshot=False):
        """Factorise each codebook-eligible core, share ids across keys recurring >= WT_MINREP, then
        verify-match the rest against shared programs (short/partial hits). Literal short records key
        on their verbatim RLE (loopless). When ``oneshot`` is off, structured one-shots get an inline
        single-use id; when on, they fall through to the inline one-shot. Returns ``{id: (steps,
        loop)}``; assigns ``rec['wt_id']`` byte-exactly (unroll == offsets).
        """
        cb = [rec for rec in records if rec["codebook"]]
        for rec in cb:
            if rec["literal"]:
                steps = run_length_encode(rec["core"])
                rec["steps"], rec["loop"] = steps, len(steps)
                rec["has_body"] = False
                continue
            steps, loop = factorise(rec["core"])
            rec["steps"], rec["loop"] = steps, loop
            rec["has_body"] = loop < len(steps)
        by_key = defaultdict(list)
        for rec in cb:
            by_key[program_key(rec["steps"], rec["loop"])].append(rec)
        qualifying = sorted(
            (min(r["pos"] for r in recs), key, recs)
            for key, recs in by_key.items()
            if len(recs) >= WT_MINREP
        )
        programs = {}
        shared = {}
        next_id = 0
        for _, key, recs in qualifying:
            rep = min(recs, key=lambda r: r["pos"])
            programs[next_id] = (rep["steps"], rep["loop"])
            shared[key] = next_id
            next_id += 1
        for rec in cb:
            wid = shared.get(program_key(rec["steps"], rec["loop"]))
            if wid is not None and cls._verify(rec, programs[wid]):
                rec["wt_id"] = wid
        for rec in cb:
            if rec["wt_id"] is not None:
                continue
            for wid in sorted(programs):
                if cls._verify(rec, programs[wid]):
                    rec["wt_id"] = wid
                    break
        if oneshot:
            return programs
        for rec in cb:
            if rec["wt_id"] is not None or not rec["has_body"]:
                continue
            program = (rec["steps"], rec["loop"])
            if cls._verify(rec, program):
                programs[next_id] = program
                rec["wt_id"] = next_id
                next_id += 1
        return programs

    @staticmethod
    def _verify(rec, program):
        steps, loop = program
        return unroll(steps, loop, rec["length"], rec["lead"]) == rec["offsets"]

    @classmethod
    def _emit(cls, records, programs, irq, oneshot=False):
        drop_idx, new_rows = [], []
        defined = set()
        for rec in sorted(records, key=lambda r: r["pos"]):
            wid = rec["wt_id"]
            if wid is None:
                if not oneshot:
                    continue
                rows = cls._oneshot_rows(rec["reg"], rec["offsets"], irq)
            else:
                rows = []
                if wid not in defined:
                    steps, loop = programs[wid]
                    rows.extend(cls._def_rows(wid, steps, loop, irq))
                    defined.add(wid)
                rows.extend(
                    cls._ref_rows(rec["reg"], wid, rec["length"], rec["lead"], irq)
                )
            for r in rows:
                r["__pos"] = rec["pos"]
                new_rows.append(r)
            drop_idx.extend(rec["rows"])
        return drop_idx, new_rows

    @staticmethod
    def _oneshot_rows(reg, offsets, irq):
        """Self-contained inline one-shot on ``reg``: LEN_HI/LO then the verbatim offsets as RLE
        OFFSET(/HOLD) atoms, terminated by END -- no codebook id, no DEF/REF indirection.
        """
        length = len(offsets) & 0xFFFF
        rows = [
            _row(
                reg,
                WAVETABLE_ONESHOT_OP,
                WT_ONESHOT_SUBREG_LEN_HI,
                (length >> 8) & 0xFF,
                irq,
                irq,
            ),
            _row(
                reg,
                WAVETABLE_ONESHOT_OP,
                WT_ONESHOT_SUBREG_LEN_LO,
                length & 0xFF,
                irq,
                irq,
            ),
        ]
        for off, hold in run_length_encode(offsets):
            rows.append(
                _row(
                    reg,
                    WAVETABLE_ONESHOT_OP,
                    WT_ONESHOT_SUBREG_OFFSET,
                    int(off) & 0xFF,
                    irq,
                    irq,
                )
            )
            if int(hold) != 1:
                rows.append(
                    _row(
                        reg,
                        WAVETABLE_ONESHOT_OP,
                        WT_ONESHOT_SUBREG_HOLD,
                        int(hold) & 0xFFFF,
                        irq,
                        irq,
                    )
                )
        rows.append(_row(reg, WAVETABLE_ONESHOT_OP, WT_ONESHOT_SUBREG_END, 0, irq, irq))
        return rows

    @staticmethod
    def _def_rows(wt_id, steps, loop, irq):
        rows = [_row(0, WAVETABLE_DEF_OP, -1, wt_id, irq, irq)]
        for off, hold in steps:
            rows.append(
                _row(
                    0,
                    WAVETABLE_STEP_OP,
                    WT_STEP_SUBREG_OFFSET,
                    int(off) & 0xFF,
                    irq,
                    irq,
                )
            )
            if int(hold) != 1:
                rows.append(
                    _row(
                        0,
                        WAVETABLE_STEP_OP,
                        WT_STEP_SUBREG_HOLD,
                        int(hold) & 0xFFFF,
                        irq,
                        irq,
                    )
                )
        rows.append(
            _row(
                0, WAVETABLE_STEP_OP, WT_STEP_SUBREG_LOOP, int(loop) & 0xFFFF, irq, irq
            )
        )
        rows.append(_row(0, WAVETABLE_END_OP, -1, wt_id, irq, irq))
        return rows

    @staticmethod
    def _ref_rows(reg, wt_id, length, lead, irq):
        rows = [
            _row(reg, WAVETABLE_REF_OP, WT_REF_SUBREG_ID, wt_id, irq, irq),
            _row(
                reg,
                WAVETABLE_REF_OP,
                WT_REF_SUBREG_LEN_HI,
                (length >> 8) & 0xFF,
                irq,
                irq,
            ),
            _row(reg, WAVETABLE_REF_OP, WT_REF_SUBREG_LEN_LO, length & 0xFF, irq, irq),
            _row(
                reg, WAVETABLE_REF_OP, WT_REF_SUBREG_LEAD, len(lead) & 0xFFFF, irq, irq
            ),
        ]
        for off in lead:
            rows.append(
                _row(
                    reg,
                    WAVETABLE_REF_OP,
                    WT_REF_SUBREG_LEADOFF,
                    int(off) & 0xFF,
                    irq,
                    irq,
                )
            )
        return rows
