#!/usr/bin/env python3
"""Pass-by-pass pipeline tracer (torch-free): given a pipeline spec + a dump
parquet, run the real ``RegLogParser`` encode path with every ``MacroPass`` and
the parser stages instrumented, then report per stage which flag gated it,
whether it fired, and how the op mix changed. Text + JSON output; ``--isolate
FLAG`` re-runs with the flag off to prove what it changed. Trust the trace."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import shlex
import sys
import tempfile
from types import SimpleNamespace

import pandas as pd

from preframr_tokens import stfconstants as S
from preframr_tokens.macros.flag_registry import (
    ensure_passes_registered,
    macro_flag_names,
)
from preframr_tokens.reg_match import reg_class

_OP_NAME = {
    v: k[:-3] for k, v in vars(S).items() if k.endswith("_OP") and isinstance(v, int)
}
_SPECIAL_REG = {
    S.FRAME_REG: "FRAME",
    S.DELAY_REG: "DELAY",
    S.MODE_VOL_REG: "MODE_VOL",
    S.FILTER_REG: "FILTER_RES",
    S.FC_LO_REG: "FC",
}

_PIPELINE_NAME_TO_FLAG = {
    "hard_restart": "hard_restart_pass",
    "voice_block_order": "voice_canonical_block_order",
    "freq_trajectory": "freq_trajectory_pass",
    "loop": "loop_pass",
    "coarsen": "coarsen_pass",
    "fuzzy_loop": "fuzzy_loop_pass",
}

_BASE_PARSE_ARGS = dict(
    cents=50,
    exclude_list=None,
    min_irq=int(1.5e4),
    max_irq=int(2.5e4),
    min_song_tokens=256,
    diffq=4,
    loop_lookahead=3,
    coarsen_min_len=16,
    pipeline_spec="",
    meta_exclude_digi=False,
    meta_irq_lo=0,
    meta_irq_hi=0,
    meta_require=False,
)

_PARSER_STAGE_METHODS = (
    "_squeeze_changes",
    "_combine_regs",
    "_quantize_freq_to_cents",
    "_simplify_ctrl",
    "_simplify_pcm",
    "_squeeze_frame_regs",
    "_consolidate_frames",
    "_cap_delay",
    "_norm_pr_order",
    "_add_voice_reg",
)

_MAX_SIG_ROWS = 200_000


def _opname(op):
    return _OP_NAME.get(int(op), f"OP{int(op)}")


def _reglabel(reg):
    reg = int(reg)
    if reg in _SPECIAL_REG:
        return _SPECIAL_REG[reg]
    cls = reg_class(reg)
    if cls is not None:
        return f"v{cls[1]}.{cls[0]}"
    return f"reg{reg}"


def _ophist(df):
    if df is None or len(df) == 0 or "op" not in df.columns:
        return {}
    return {_opname(k): int(v) for k, v in df["op"].value_counts().to_dict().items()}


def _row_multiset(df):
    if df is None or len(df) == 0 or len(df) > _MAX_SIG_ROWS:
        return None
    cols = [c for c in ("reg", "op", "val", "subreg") if c in df.columns]
    return collections.Counter(map(tuple, df[cols].to_numpy().tolist()))


def _op_delta(before, after):
    keys = set(before) | set(after)
    return {
        k: after.get(k, 0) - before.get(k, 0)
        for k in keys
        if after.get(k, 0) != before.get(k, 0)
    }


def _decode_rows(df, limit):
    out = []
    if df is None or len(df) == 0:
        return out
    cols = df.columns
    for i, row in enumerate(df.itertuples(index=False)):
        if i >= limit:
            break
        d = dict(zip(cols, row))
        out.append(
            {
                "i": i,
                "reg": int(d.get("reg", -1)),
                "reg_label": _reglabel(d.get("reg", -1)),
                "op": _opname(d.get("op", 0)),
                "val": int(d.get("val", 0)),
                "subreg": int(d.get("subreg", -1)) if "subreg" in cols else -1,
            }
        )
    return out


class Tracer:
    """Wrap every ``MacroPass.apply`` + selected parser methods to record a
    before/after snapshot of each stage, in real call order."""

    def __init__(self, args):
        self.args = args
        self.records = []
        self._orig = []

    def _record(self, stage, kind, gate_set, before, after):
        before_ms = _row_multiset(before)
        after_ms = _row_multiset(after)
        nb = 0 if before is None else len(before)
        na = 0 if after is None else len(after)
        if before_ms is not None and after_ms is not None:
            content_changed = before_ms != after_ms
            order_changed = (not content_changed) and not before.equals(after)
        else:
            content_changed = nb != na or _ophist(before) != _ophist(after)
            order_changed = False
        changed = content_changed or order_changed or nb != na
        gate_flags = (
            {f: bool(getattr(self.args, f, False)) for f in sorted(gate_set)}
            if gate_set
            else {}
        )
        if changed:
            status = "FIRED"
        elif gate_set:
            status = "skip(off)" if not any(gate_flags.values()) else "on·nochange"
        else:
            status = "nochange"
        self.records.append(
            {
                "idx": len(self.records),
                "stage": stage,
                "kind": kind,
                "gate_flags": gate_flags,
                "rows_before": nb,
                "rows_after": na,
                "delta": na - nb,
                "op_delta": _op_delta(_ophist(before), _ophist(after)),
                "op_hist_after": _ophist(after),
                "order_changed": order_changed,
                "status": status,
                "branch": False,
            }
        )

    def _wrap_pass(self, cls):
        orig = cls.__dict__["apply"]
        gate_set = frozenset(getattr(cls, "GATE_FLAGS", frozenset()) or frozenset())
        name = cls.__name__

        def wrapped(inner_self, df, args=None):
            out = orig(inner_self, df, args=args)
            self._record(name, "macro", gate_set, df, out)
            return out

        cls.apply = wrapped
        self._orig.append((cls, "apply", orig))

    def _wrap_method(self, owner, attr, kind):
        orig = getattr(owner, attr)

        def wrapped(*a, **k):
            before = next((x for x in a if isinstance(x, pd.DataFrame)), None)
            out = orig(*a, **k)
            after = out if isinstance(out, pd.DataFrame) else before
            self._record(attr.lstrip("_"), kind, None, before, after)
            return out

        setattr(owner, attr, wrapped)
        self._orig.append((owner, attr, orig))

    def __enter__(self):
        ensure_passes_registered()
        from preframr_tokens import reglogparser
        from preframr_tokens.macros.passes_base import MacroPass

        seen = set()
        stack = list(MacroPass.__subclasses__())
        while stack:
            cls = stack.pop()
            stack.extend(cls.__subclasses__())
            if cls in seen or "apply" not in cls.__dict__:
                continue
            seen.add(cls)
            self._wrap_pass(cls)
        for attr in _PARSER_STAGE_METHODS:
            if hasattr(reglogparser.RegLogParser, attr):
                self._wrap_method(reglogparser.RegLogParser, attr, "parser")
        return self

    def __exit__(self, *exc):
        for owner, attr, orig in reversed(self._orig):
            setattr(owner, attr, orig)
        self._orig.clear()
        return False


def build_args(spec, cargs, overrides):
    """Build a torch-free args namespace: parse defaults + all macro flags off,
    then apply the pipeline-spec names, ``--cargs`` flags, and overrides. Returns
    the namespace plus a resolution report (every flag set + every name/flag the
    tool did not recognize)."""
    known = macro_flag_names()
    args = SimpleNamespace(**_BASE_PARSE_ARGS)
    for flag in known:
        setattr(args, flag, False)
    resolution = []
    unknown_names = []
    unknown_flags = []
    entries = []
    if spec:
        entries = spec.get("transforms", []) if isinstance(spec, dict) else spec
    for entry in entries:
        name = entry["name"] if isinstance(entry, dict) else entry
        params = entry.get("params", {}) if isinstance(entry, dict) else {}
        if name == "legato_per_cluster":
            for cluster in params.get("clusters", []):
                attr = f"legato_pass_c{int(cluster)}"
                setattr(args, attr, True)
                resolution.append((name, attr, True))
                if attr not in known:
                    unknown_flags.append(attr)
            continue
        flag = _PIPELINE_NAME_TO_FLAG.get(name)
        if flag is None:
            unknown_names.append(name)
            continue
        setattr(args, flag, True)
        resolution.append((name, flag, True))
        if flag not in known:
            unknown_flags.append(flag)
    for tok in cargs:
        if not tok.startswith("--"):
            continue
        val = True
        body = tok[2:]
        if body.startswith("no-"):
            val = False
            body = body[3:]
        attr = body.replace("-", "_")
        setattr(args, attr, val)
        resolution.append(("(cargs)", attr, val))
        if attr not in known and attr not in _BASE_PARSE_ARGS:
            unknown_flags.append(attr)
    for key, value in overrides.items():
        setattr(args, key, value)
    args.pipeline_spec = json.dumps(spec) if spec else ""
    return args, resolution, unknown_names, unknown_flags


def _mark_branches(records):
    """Flag stages whose output is discarded: the next stage receives the same
    input this stage did (e.g. the parser's filter-preview ``add_voice_reg`` on a
    throwaway copy), so this stage's change never feeds forward."""
    for cur, nxt in zip(records, records[1:]):
        if (
            cur["rows_after"] != cur["rows_before"]
            and cur["rows_before"] == nxt["rows_before"]
        ):
            cur["branch"] = True
    return records


def run_trace(dump, args, max_perm):
    """Run the instrumented parse over ``dump`` and return (records, final_df)."""
    from preframr_tokens.reglogparser import RegLogParser

    with Tracer(args) as tracer:
        parser = RegLogParser(args=args)
        dfs = list(
            parser.parse(dump, max_perm=max_perm, require_pq=False, reparse=True)
        )
    return _mark_branches(tracer.records), (dfs[-1] if dfs else None)


def flag_report(records, args):
    """For each active macro flag, the gated stages that read it and which
    fired (firing may be attributable to a co-flag; use --isolate to confirm)."""
    out = {}
    for flag in sorted(macro_flag_names()):
        if not getattr(args, flag, False):
            continue
        reading = [r for r in records if r["gate_flags"] and flag in r["gate_flags"]]
        fired = [r["stage"] for r in reading if r["status"] == "FIRED"]
        out[flag] = {
            "read_by": [r["stage"] for r in reading],
            "fired_in": fired,
            "effective": bool(fired),
        }
    return out


def isolate(dump, base_args, flag, max_perm):
    """Re-run with ``flag`` forced off and report the effect *introduced* at each
    stage (the marginal divergence, with downstream carry-forward subtracted), so
    a one-time change shows only where it happens, not smeared across every later
    stage. Also returns the net divergence at the final df."""
    base, final_on = run_trace(dump, base_args, max_perm)
    off_args = copy.copy(base_args)
    setattr(off_args, flag, False)
    off, final_off = run_trace(dump, off_args, max_perm)
    diffs = []
    prev_op, prev_rows = {}, 0
    for b, o in zip(base, off):
        if b["stage"] != o["stage"]:
            break
        cum_op = _op_delta(o["op_hist_after"], b["op_hist_after"])
        cum_rows = b["rows_after"] - o["rows_after"]
        marg_op = _op_delta(prev_op, cum_op)
        marg_rows = cum_rows - prev_rows
        if marg_op or marg_rows or b["status"] != o["status"]:
            diffs.append(
                {
                    "stage": b["stage"],
                    "status_off": o["status"],
                    "status_on": b["status"],
                    "rows_introduced": marg_rows,
                    "op_introduced": marg_op,
                }
            )
        prev_op, prev_rows = cum_op, cum_rows
    net = {
        "rows": (0 if final_on is None else len(final_on))
        - (0 if final_off is None else len(final_off)),
        "op": _op_delta(_ophist(final_off), _ophist(final_on)),
    }
    return {"sites": diffs, "net": net}


def _fmt_op_delta(delta):
    if not delta:
        return ""
    parts = [f"+{k}({v})" if v > 0 else f"-{k}({-v})" for k, v in sorted(delta.items())]
    return " ".join(parts)


def _fmt_gates(gate_flags):
    if not gate_flags:
        return "(unconditional)"
    return " ".join(f"{f}={'T' if v else 'F'}" for f, v in gate_flags.items())


def render_text(report, full, show_rows):
    out = []
    a = out.append
    a("=== preframr-tokens pipeline trace ===")
    a(f"dump: {report['dump']}")
    a(f"rows(final): {report['final_rows']}   max_perm: {report['max_perm']}")
    a("")
    a("PIPELINE SPEC RESOLUTION")
    for name, flag, val in report["resolution"]:
        a(f"  {name:<30} -> {flag}={'True' if val else 'False'}")
    for name in report["unknown_names"]:
        a(
            f"  !! UNRECOGNIZED spec name {name!r}: no flag set (silently no-ops in the real pipeline)"
        )
    for flag in report["unknown_flags"]:
        a(
            f"  !! FLAG {flag!r} is not read by any pass (typo / renamed?) -- would have no effect"
        )
    active = [f for f, v in report["active_flags"].items() if v]
    a(
        f"  ACTIVE MACRO FLAGS ({len(active)}): {', '.join(active) if active else '(none)'}"
    )
    a("")
    if report["final_rows"] == 0:
        a(
            "!! parse produced NO df (song filtered: check min_irq/max_irq/min_song_tokens, or the dump)"
        )
    a("STAGE TRACE")
    a(f"  {'#':>2}  {'stage':<26} {'rows':>7} {'Δ':>6}  {'status':<12} gate / op-delta")
    for r in report["records"]:
        line = (
            f"  {r['idx']:>2}  {r['stage']:<26} {r['rows_after']:>7} {r['delta']:>+6}  "
            f"{r['status']:<12} {_fmt_gates(r['gate_flags'])}"
        )
        od = _fmt_op_delta(r["op_delta"])
        if od:
            line += f"  | {od}"
        if r["branch"]:
            line += "   [branch: output discarded]"
        a(line)
    a("")
    a("FLAG EFFECT CHECK (gated stages that fired)")
    for flag, info in report["flag_report"].items():
        mark = "EFFECTIVE" if info["effective"] else "no fire on this dump"
        a(
            f"  {flag:<24} {mark:<22} read_by={info['read_by']} fired_in={info['fired_in']}"
        )
    a("")
    a("FINAL OP HISTOGRAM")
    for op, n in sorted(report["final_op_hist"].items(), key=lambda kv: -kv[1]):
        a(f"  {op:<22} {n}")
    if report.get("isolation") is not None:
        iso = report["isolation"]
        a("")
        a(f"ISOLATION: effect introduced by --{report['isolate_flag']} (on vs off)")
        net = iso["net"]
        a(
            f"  NET at final df: {net['rows']:+d} rows  | {_fmt_op_delta(net['op']) or '(no op change)'}"
        )
        if not iso["sites"]:
            a("  (no stage differs -- the flag had NO effect on this dump)")
        for d in iso["sites"]:
            st = (
                f" status {d['status_off']}->{d['status_on']}"
                if d["status_off"] != d["status_on"]
                else ""
            )
            a(
                f"  {d['stage']:<26} {d['rows_introduced']:>+6} rows{st}  "
                f"| {_fmt_op_delta(d['op_introduced'])}"
            )
    if full:
        a("")
        a(f"FINAL DECODED ROWS (first {show_rows})")
        for row in report["final_decoded"]:
            a(
                f"  {row['i']:>4}  {row['reg_label']:<10} {row['op']:<16} "
                f"val={row['val']:<8} subreg={row['subreg']}"
            )
    return "\n".join(out)


def _slice_dump(path, head, tail, rows):
    df = pd.read_parquet(path)
    if rows:
        lo, _, hi = rows.partition(":")
        df = df.iloc[int(lo or 0) : int(hi or len(df))]
    elif head:
        df = df.head(head)
    elif tail:
        df = df.tail(tail)
    tmp = tempfile.NamedTemporaryFile(suffix=".dump.parquet", delete=False)
    df.to_parquet(tmp.name)
    return tmp.name


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", help="path to a *.dump.parquet reglog dump")
    ap.add_argument(
        "--pipeline-spec",
        default="",
        help="inline JSON or @file: {'transforms':[{'name':...}]}",
    )
    ap.add_argument(
        "--cargs",
        default="",
        help="extra raw flags, e.g. '--ctrl-triple-pass --lonely-catch-all'",
    )
    ap.add_argument("--max-perm", type=int, default=1)
    ap.add_argument("--format", choices=("text", "json"), default="text")
    ap.add_argument("--full", action="store_true", help="dump the final decoded rows")
    ap.add_argument("--show-rows", type=int, default=80)
    ap.add_argument(
        "--isolate", default="", help="re-run with this flag forced off and diff"
    )
    ap.add_argument(
        "--head", type=int, default=0, help="trace only the first N raw dump rows"
    )
    ap.add_argument("--tail", type=int, default=0)
    ap.add_argument("--rows", default="", help="raw row slice LO:HI")
    ap.add_argument("--cents", type=int, default=_BASE_PARSE_ARGS["cents"])
    ap.add_argument("--min-irq", type=int, default=_BASE_PARSE_ARGS["min_irq"])
    ap.add_argument("--max-irq", type=int, default=_BASE_PARSE_ARGS["max_irq"])
    ap.add_argument(
        "--min-song-tokens", type=int, default=_BASE_PARSE_ARGS["min_song_tokens"]
    )
    cli = ap.parse_args(argv)

    spec = None
    if cli.pipeline_spec:
        raw = cli.pipeline_spec
        if raw.startswith("@"):
            with open(raw[1:], encoding="utf-8") as fh:
                raw = fh.read()
        spec = json.loads(raw)

    overrides = dict(
        cents=cli.cents,
        min_irq=cli.min_irq,
        max_irq=cli.max_irq,
        min_song_tokens=cli.min_song_tokens,
    )
    args, resolution, unknown_names, unknown_flags = build_args(
        spec, shlex.split(cli.cargs), overrides
    )

    dump = cli.dump
    if cli.head or cli.tail or cli.rows:
        dump = _slice_dump(dump, cli.head, cli.tail, cli.rows)

    records, final_df = run_trace(dump, args, cli.max_perm)

    report = {
        "dump": dump,
        "max_perm": cli.max_perm,
        "resolution": resolution,
        "unknown_names": unknown_names,
        "unknown_flags": unknown_flags,
        "active_flags": {
            f: bool(getattr(args, f, False)) for f in sorted(macro_flag_names())
        },
        "records": records,
        "flag_report": flag_report(records, args),
        "final_rows": 0 if final_df is None else len(final_df),
        "final_op_hist": _ophist(final_df),
        "final_decoded": _decode_rows(final_df, cli.show_rows) if cli.full else [],
        "isolation": None,
        "isolate_flag": cli.isolate,
    }
    if cli.isolate:
        report["isolation"] = isolate(dump, args, cli.isolate, cli.max_perm)

    if cli.format == "json":
        print(json.dumps(report, indent=1, default=str))
    else:
        print(render_text(report, cli.full, cli.show_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
