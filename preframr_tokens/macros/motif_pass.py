"""Corpus-mined motif pass: a boundary-constrained, cross-composer motif
dictionary applied between the structural macros and Unigram. A motif is a
recurring atom sequence replaced losslessly by one MOTIF_OP atom; ``expand``
inverts it exactly. OFF by default; the ``motif`` transform self-gates on
``args.motif_pass`` and needs a mined ``args.motif_dict``."""

import json
from collections import Counter

from preframr_tokens.macros.passes_base import MacroPass
from preframr_tokens.macros.transform import Transform, register
from preframr_tokens.stfconstants import FRAME_REG, MOTIF_ARG, MOTIF_OP

__all__ = [
    "MotifDict",
    "mine_motifs",
    "mine_templates",
    "MotifPass",
    "MotifTransform",
    "MOTIF_OP",
    "MOTIF_ARG",
    "get_motif_dict",
]

_ATOM_KEYS = ("op", "reg", "subreg", "val", "diff")


def get_motif_dict(args):
    """Resolve ``args.motif_dict`` to a ``MotifDict``. Accepts either a loaded
    ``MotifDict`` (tests / in-process callers) or a path string to a ``to_json``
    artifact (the ``--motif-dict`` CLI flag); the loaded object is cached back on
    ``args`` so a corpus parse reads the JSON once per worker. Returns ``None``
    when unset/empty."""
    if args is None:
        return None
    cached = getattr(args, "_motif_dict_obj", None)
    if cached is not None:
        return cached
    raw = getattr(args, "motif_dict", None)
    if raw is None or isinstance(raw, str) and not raw:
        return None
    if isinstance(raw, MotifDict):
        return raw
    with open(raw) as f:
        loaded = MotifDict.from_json(f.read())
    try:
        args._motif_dict_obj = loaded
    except AttributeError:
        pass
    return loaded


def _as_atom(row):
    """Normalise a row / dict / tuple to an ``(op, reg, subreg, val, diff)`` tuple."""
    if isinstance(row, tuple):
        return row
    return tuple(int(row[k]) for k in _ATOM_KEYS)


def _is_frame_advance(atom):
    """True when an atom advances a frame (a structural boundary)."""
    return atom[1] == FRAME_REG


def _motif_atom(mid):
    """The single MOTIF_OP atom standing in for motif ``mid`` (id in ``val``);
    filler fields are 0 to stay within the row df's unsigned dtypes."""
    return (MOTIF_OP, 0, 0, int(mid), 0)


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


def _arg_atom(val):
    """The MOTIF_ARG atom carrying one slot value for the preceding template."""
    return (MOTIF_ARG, 0, 0, int(val), 0)


def _shape(atoms):
    """The value-free ``(op, reg, subreg, diff)`` shape of an atom sequence."""
    return tuple((a[0], a[1], a[2], a[4]) for a in atoms)


def _norm_template(t):
    """Normalize one template dict: tuple shape, int positions / consts."""
    return {
        "id": int(t["id"]),
        "shape": tuple(tuple(s) for s in t["shape"]),
        "consts": {int(p): int(v) for p, v in t["consts"].items()},
        "slots": [int(p) for p in t["slots"]],
    }


def _matches(window, tmpl):
    """True when ``window`` matches a template's shape and constant vals."""
    return _shape(window) == tmpl["shape"] and all(
        window[p][3] == v for p, v in tmpl["consts"].items()
    )


def _encode_templates(seq, templates):
    """Greedy longest-first collapse: each match -> a MOTIF_OP atom plus one
    MOTIF_ARG atom per slot, in slot order."""
    out = []
    i = 0
    while i < len(seq):
        hit = next(
            (
                t
                for t in templates
                if i + len(t["shape"]) <= len(seq)
                and _matches(seq[i : i + len(t["shape"])], t)
            ),
            None,
        )
        if hit is None:
            out.append(seq[i])
            i += 1
            continue
        out.append(_motif_atom(hit["id"]))
        out.extend(_arg_atom(seq[i + p][3]) for p in hit["slots"])
        i += len(hit["shape"])
    return out


def _expand_templates(seq, by_id):
    """Inverse of ``_encode_templates`` (byte-exact)."""
    out = []
    i = 0
    while i < len(seq):
        if seq[i][0] != MOTIF_OP:
            out.append(seq[i])
            i += 1
            continue
        tmpl = by_id[seq[i][3]]
        slot_at = {p: seq[i + 1 + k][3] for k, p in enumerate(tmpl["slots"])}
        for pos, (op, reg, sub, diff) in enumerate(tmpl["shape"]):
            val = slot_at[pos] if pos in slot_at else tmpl["consts"][pos]
            out.append((op, reg, sub, val, diff))
        i += 1 + len(tmpl["slots"])
    return out


class MotifDict:
    """Frozen motif dictionary: v1 ordered merges + expansions, or v2 value-
    slotted templates (shape + constant vals + slots)."""

    def __init__(self, merges, expansions, templates=None):
        self.merges = [tuple(m) for m in merges]
        self.expansions = {int(k): [tuple(a) for a in v] for k, v in expansions.items()}
        self.templates = None
        self._by_id = {}
        if templates:
            self.templates = sorted(
                (_norm_template(t) for t in templates),
                key=lambda t: len(t["shape"]),
                reverse=True,
            )
            self._by_id = {t["id"]: t for t in self.templates}

    def __len__(self):
        return len(self.templates) if self.templates else len(self.expansions)

    def _to_json_v2(self):
        templates = self.templates or []
        return json.dumps(
            {
                "version": 2,
                "templates": [
                    {
                        "id": t["id"],
                        "shape": [list(s) for s in t["shape"]],
                        "consts": {str(p): v for p, v in t["consts"].items()},
                        "slots": list(t["slots"]),
                    }
                    for t in templates
                ],
            }
        )

    def to_json(self):
        """Serialize to a JSON string (v1 merges or v2 templates)."""
        if self.templates:
            return self._to_json_v2()

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
        """Load a dictionary (v1 merges or v2 templates) from JSON."""
        d = json.loads(s)
        if d.get("version") == 2:
            return cls([], {}, templates=d["templates"])

        def dec(sym):
            return sym["motif"] if "motif" in sym else tuple(sym["atom"])

        merges = [(dec(a), dec(b), mid) for a, b, mid in d["merges"]]
        expansions = {int(k): [tuple(a) for a in v] for k, v in d["expansions"].items()}
        return cls(merges, expansions)

    def encode(self, atoms):
        """Collapse motifs into MOTIF_OP rows (plus MOTIF_ARG slots in v2)."""
        seq = [_as_atom(a) for a in atoms]
        if self.templates:
            return _encode_templates(seq, self.templates)
        for sym_a, sym_b, mid in self.merges:
            seq = _merge_run(seq, sym_a, sym_b, mid)
        return [a if isinstance(a, tuple) else _motif_atom(a) for a in seq]

    def expand(self, atoms):
        """Inverse of ``encode`` (byte-exact)."""
        seq = [_as_atom(a) for a in atoms]
        if self.templates:
            return _expand_templates(seq, self._by_id)
        out = []
        for a in seq:
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


def _shape_stats(streams, composers):
    """Per-shape occurrence count, composer set, and per-position value sets
    over length-3/2 windows that do not end on a frame-advance."""
    stats = {}
    for s, cp in zip(streams, composers):
        atoms = [_as_atom(a) for a in s]
        for length in (3, 2):
            for i in range(len(atoms) - length + 1):
                window = atoms[i : i + length]
                if _is_frame_advance(window[-1]):
                    continue
                st = stats.setdefault(
                    _shape(window),
                    {"n": 0, "comp": set(), "vals": [set() for _ in range(length)]},
                )
                st["n"] += 1
                st["comp"].add(cp)
                for p, atom in enumerate(window):
                    st["vals"][p].add(atom[3])
    return stats


def mine_templates(streams, composers, k=256, min_count=3, min_composers=3):
    """Mine value-slotted templates: a (op,reg,subreg,diff) shape qualifies when
    it spans >= min_composers composers and >= min_count occurrences (pooled over
    value-instances); positions whose val varies become slots, constant positions
    are baked. Greedy longest-first at encode."""
    stats = _shape_stats(streams, composers)
    qualified = sorted(
        (
            (sh, st)
            for sh, st in stats.items()
            if st["n"] >= min_count and len(st["comp"]) >= min_composers
        ),
        key=lambda x: x[1]["n"],
        reverse=True,
    )
    templates = [
        {
            "id": tid,
            "shape": [list(a) for a in sh],
            "consts": {
                p: next(iter(v)) for p, v in enumerate(st["vals"]) if len(v) == 1
            },
            "slots": [p for p, v in enumerate(st["vals"]) if len(v) > 1],
        }
        for tid, (sh, st) in enumerate(qualified[:k])
    ]
    return MotifDict([], {}, templates=templates)


def _atoms_of(df):
    """Extract the ``(op,reg,subreg,val,diff)`` atom stream from a row df."""
    return [
        (int(r.op), int(r.reg), int(getattr(r, "subreg", -1)), int(r.val), int(r.diff))
        for r in df.itertuples()
    ]


def _rebuild_df(df, atoms):
    """Rebuild a row df from an atom stream, reusing ``df``'s columns and the
    per-song-constant ``irq``; ``description`` defaults to 0."""
    import pandas as pd

    irq = int(df["irq"].iloc[0]) if "irq" in df.columns and len(df) else -1
    rows = []
    for op, reg, subreg, val, diff in atoms:
        row = {"op": op, "reg": reg, "subreg": subreg, "val": val, "diff": diff}
        if "irq" in df.columns:
            row["irq"] = irq
        if "description" in df.columns:
            row["description"] = 0
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out[[c for c in df.columns if c in out.columns]]
    for col, dt in df.dtypes.items():
        if col in out.columns:
            try:
                out[col] = out[col].astype(dt)
            except (TypeError, ValueError):
                pass
    out.attrs.update(df.attrs)
    return out


class MotifPass(MacroPass):
    """Encode-side pass: substitute dictionary motifs; OFF unless
    ``args.motif_pass`` with a loaded ``args.motif_dict``."""

    GATE_FLAGS = frozenset({"motif_pass"})

    def apply(self, df, args=None):
        """Collapse dictionary motifs into MOTIF_OP rows, or pass through."""
        if args is None or not getattr(args, "motif_pass", False):
            return df
        motif_dict = get_motif_dict(args)
        if motif_dict is None or "op" not in df.columns or df.empty:
            return df
        encoded = motif_dict.encode(_atoms_of(df))
        if len(encoded) == len(df):
            return df
        return _rebuild_df(df, encoded)


@register("motif")
class MotifTransform(Transform):
    """Pipeline transform: forward collapses dictionary motifs into MOTIF_OP
    rows, inverse expands them back (lossless)."""

    NAME = "motif"
    TIER = "bit_exact"
    OP_CODES = frozenset({MOTIF_OP, MOTIF_ARG})
    DECOMPOSES_TO_ATOMS = True
    DECODES_VIA_DF = True
    LOSS_TIER = "zero"
    REQUIRES_ARGS = frozenset({"motif_pass"})

    def forward(self, df, args=None):
        return MotifPass().apply(df, args=args)

    def inverse(self, df, args=None):
        motif_dict = get_motif_dict(args)
        if motif_dict is None or "op" not in df.columns or df.empty:
            return df
        if not (df["op"] == MOTIF_OP).any():
            return df
        return _rebuild_df(df, motif_dict.expand(_atoms_of(df)))
