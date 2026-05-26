"""Corpus-mined motif pass: a boundary-constrained, cross-composer motif
dictionary applied between the structural macros and Unigram. A motif is a
recurring atom sequence replaced losslessly by one MOTIF_OP atom; ``expand``
inverts it exactly. OFF by default and unwired (see release notes for the
pipeline / decode / constrained-decode integration)."""

import json
from collections import Counter

from preframr_tokens.macros.passes_base import MacroPass
from preframr_tokens.stfconstants import FRAME_REG, MOTIF_OP

__all__ = ["MotifDict", "mine_motifs", "MotifPass", "MOTIF_OP", "MOTIF_SUBREG"]

MOTIF_SUBREG = -1

_ATOM_KEYS = ("op", "reg", "subreg", "val")


def _as_atom(row):
    """Normalise a row / dict / tuple to an ``(op, reg, subreg, val)`` tuple."""
    if isinstance(row, tuple):
        return row
    return tuple(int(row[k]) for k in _ATOM_KEYS)


def _is_frame_advance(atom):
    """True when an atom advances a frame (a structural boundary)."""
    return atom[1] == FRAME_REG


def _motif_atom(mid):
    """The single MOTIF_OP atom standing in for motif ``mid`` (id in ``val``)."""
    return (MOTIF_OP, MOTIF_SUBREG, MOTIF_SUBREG, int(mid))


def _merge_run(seq, sym_a, sym_b, mid):
    """Replace every adjacent ``(sym_a, sym_b)`` in ``seq`` with motif ``mid``."""
    out = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == sym_a and seq[i + 1] == sym_b:
            out.append(mid)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _ncomposers(streams, composers, sym_a, sym_b):
    """Count distinct composers whose stream contains the pair ``(sym_a, sym_b)``."""
    seen = set()
    for s, cp in zip(streams, composers):
        for i in range(len(s) - 1):
            if s[i] == sym_a and s[i + 1] == sym_b:
                seen.add(cp)
                break
    return len(seen)


class MotifDict:
    """Frozen motif dictionary: ordered merges plus per-motif atom expansions."""

    def __init__(self, merges, expansions):
        self.merges = [tuple(m) for m in merges]
        self.expansions = {int(k): [tuple(a) for a in v] for k, v in expansions.items()}

    def __len__(self):
        return len(self.expansions)

    def to_json(self):
        """Serialize the dictionary to a JSON string artifact."""

        def enc(sym):
            return {"motif": sym} if isinstance(sym, int) else {"atom": list(sym)}

        return json.dumps(
            {
                "merges": [[enc(a), enc(b), mid] for a, b, mid in self.merges],
                "expansions": {
                    str(mid): [list(a) for a in seq]
                    for mid, seq in self.expansions.items()
                },
            }
        )

    @classmethod
    def from_json(cls, s):
        """Load a dictionary from a JSON string produced by ``to_json``."""
        d = json.loads(s)

        def dec(sym):
            return sym["motif"] if "motif" in sym else tuple(sym["atom"])

        merges = [(dec(a), dec(b), mid) for a, b, mid in d["merges"]]
        expansions = {int(k): [tuple(a) for a in v] for k, v in d["expansions"].items()}
        return cls(merges, expansions)

    def encode(self, atoms):
        """Replay the merges, collapsing merged runs to single MOTIF_OP atoms."""
        seq = [_as_atom(a) for a in atoms]
        for sym_a, sym_b, mid in self.merges:
            seq = _merge_run(seq, sym_a, sym_b, mid)
        return [a if isinstance(a, tuple) else _motif_atom(a) for a in seq]

    def expand(self, atoms):
        """Inverse of ``encode``: expand every MOTIF_OP atom (byte-exact)."""
        out = []
        for a in atoms:
            a = _as_atom(a)
            if a[0] == MOTIF_OP:
                out.extend(self.expansions[a[3]])
            else:
                out.append(a)
        return out


def mine_motifs(streams, composers, k=256, min_count=3, min_composers=3):
    """Mine a ``MotifDict`` from per-song atom streams with two greedy guards:
    a boundary guard (no motif ends on a frame-advance) and a cross-composer
    floor (a pair must span >= ``min_composers`` composers)."""
    seqs = [[_as_atom(a) for a in s] for s in streams]
    ends_fa = {}
    expand = {}
    merges = []
    next_id = 0

    def ends_frame_advance(sym):
        return ends_fa[sym] if isinstance(sym, int) else _is_frame_advance(sym)

    for _ in range(k):
        cnt = Counter()
        for s in seqs:
            cnt.update(zip(s, s[1:]))
        if not cnt:
            break
        picked = None
        for (sym_a, sym_b), count in cnt.most_common():
            if count < min_count:
                break
            if ends_frame_advance(sym_b):
                continue
            if _ncomposers(seqs, composers, sym_a, sym_b) < min_composers:
                continue
            picked = (sym_a, sym_b)
            break
        if picked is None:
            break
        sym_a, sym_b = picked
        mid = next_id
        next_id += 1
        exp_a = expand[sym_a] if isinstance(sym_a, int) else [sym_a]
        exp_b = expand[sym_b] if isinstance(sym_b, int) else [sym_b]
        expand[mid] = exp_a + exp_b
        ends_fa[mid] = ends_frame_advance(sym_b)
        merges.append((sym_a, sym_b, mid))
        seqs = [_merge_run(s, sym_a, sym_b, mid) for s in seqs]
    return MotifDict(merges, {mid: expand[mid] for _, _, mid in merges})


class MotifPass(MacroPass):
    """Pipeline pass substituting dictionary motifs in a token DataFrame; OFF
    unless ``args.motif_pass`` with a loaded ``args.motif_dict``."""

    GATE_FLAGS = frozenset({"motif_pass"})

    def apply(self, df, args=None):
        """Encode the row stream with the loaded motif dictionary, or pass through."""
        if args is None or not getattr(args, "motif_pass", False):
            return df
        motif_dict = getattr(args, "motif_dict", None)
        if motif_dict is None:
            return df
        atoms = [
            (int(r.op), int(r.reg), int(getattr(r, "subreg", -1)), int(r.val))
            for r in df.itertuples()
        ]
        encoded = motif_dict.encode(atoms)
        if len(encoded) == len(atoms):
            return df
        import pandas as pd

        out = pd.DataFrame(
            [
                {"op": a[0], "reg": a[1], "subreg": a[2], "val": a[3], "description": 0}
                for a in encoded
            ]
        )
        if "irq" in df.columns and len(df):
            out["irq"] = int(df["irq"].iloc[0])
        out.attrs.update(df.attrs)
        return out
