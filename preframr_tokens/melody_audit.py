"""Held-out next-symbol transfer probe for the melody-skeleton (AGENT_TASK_melody_skeleton.md §4): does
the key-invariant INTERVAL token generalize across tunes better than the absolute-pitch baseline? Train a
first-order Markov next-symbol predictor on a TRAIN split of dumps, measure next-symbol accuracy on a
held-out TEST split (cross-tune transfer). Self-contained (parser only); interval beating absolute is the
learnability lever (the work order's 0.52 interval vs 0.41 cross-tune ceiling)."""

from __future__ import annotations

from collections import Counter, defaultdict

from preframr_tokens.macros.generator_fit import unzig
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    MELODY_INTERVAL_OP,
    MELODY_INTERVAL_SUBREG_FIRST,
    MELODY_INTERVAL_SUBREG_INTERVAL_HI,
    MELODY_INTERVAL_SUBREG_INTERVAL_LO,
    MELODY_INTERVAL_SUBREG_VOICE,
)
from preframr_tokens.tokenizer_config import default_tokenizer_args

__all__ = [
    "extract_sequences",
    "extract_interleaved",
    "transfer_accuracy",
    "ceiling_2gram",
    "measure",
    "measure_triage",
]


def extract_sequences(df):
    """Per voice, the ordered ``{intervals, notes}`` of one tune's MELODY_INTERVAL onsets: ``intervals``
    the signed semitone deltas (the key-invariant token, FIRST excluded), ``notes`` the running absolute
    note (the baseline). Reconstructs the encoder's cur_note running sum from the token stream.
    """
    ops = df["op"].to_numpy()
    subs = df["subreg"].to_numpy()
    vals = df["val"].to_numpy()
    cur = {}
    out = defaultdict(lambda: {"intervals": [], "notes": []})
    pend = None
    for i in range(len(df)):
        if int(ops[i]) != MELODY_INTERVAL_OP:
            continue
        sub = int(subs[i])
        if sub == MELODY_INTERVAL_SUBREG_VOICE:
            pend = {"voice": int(vals[i]), "first": 0, "hi": 0, "lo": 0}
        if pend is None:
            continue
        if sub == MELODY_INTERVAL_SUBREG_FIRST:
            pend["first"] = int(vals[i])
        elif sub == MELODY_INTERVAL_SUBREG_INTERVAL_HI:
            pend["hi"] = int(vals[i])
        elif sub == MELODY_INTERVAL_SUBREG_INTERVAL_LO:
            pend["lo"] = int(vals[i])
            voice = pend["voice"]
            token = ((pend["hi"] & 0xFF) << 8) | (pend["lo"] & 0xFF)
            if pend["first"]:
                cur[voice] = token
            else:
                iv = unzig(token)
                cur[voice] = cur.get(voice, 0) + iv
                out[voice]["intervals"].append(iv)
            out[voice]["notes"].append(cur[voice])
            pend = None
    return dict(out)


def extract_interleaved(df):
    """One per-tune interval sequence in raw EMISSION (frame-major) order -- voices interleaved as the
    deployed model sees them before any de-multiplexing. The layer-3 triage baseline; a melody-onset's
    own previous interval is far away (other voices intervene), so next-interval transfer is degraded.
    """
    ops = df["op"].to_numpy()
    subs = df["subreg"].to_numpy()
    vals = df["val"].to_numpy()
    out = []
    pend = None
    for i in range(len(df)):
        if int(ops[i]) != MELODY_INTERVAL_OP:
            continue
        sub = int(subs[i])
        if sub == MELODY_INTERVAL_SUBREG_VOICE:
            pend = {"voice": int(vals[i]), "first": 0, "hi": 0, "lo": 0}
        if pend is None:
            continue
        if sub == MELODY_INTERVAL_SUBREG_FIRST:
            pend["first"] = int(vals[i])
        elif sub == MELODY_INTERVAL_SUBREG_INTERVAL_HI:
            pend["hi"] = int(vals[i])
        elif sub == MELODY_INTERVAL_SUBREG_INTERVAL_LO:
            token = ((pend["hi"] & 0xFF) << 8) | (int(vals[i]) & 0xFF)
            if not pend["first"]:
                out.append(unzig(token))
            pend = None
    return {"intervals": out, "notes": out}


def _markov_table(seqs, key):
    """First-order next-symbol table ``prev -> most-common next`` over a list of per-voice sequences."""
    counts = defaultdict(Counter)
    for s in seqs:
        xs = s[key]
        for a, b in zip(xs, xs[1:]):
            counts[a][b] += 1
    return {a: c.most_common(1)[0][0] for a, c in counts.items()}


def transfer_accuracy(train_seqs, test_seqs, key):
    """Held-out next-symbol accuracy: predict each test step from the TRAIN Markov table (an unseen prev
    symbol scores 0). The cross-tune transfer score for representation ``key``."""
    table = _markov_table(train_seqs, key)
    hit = tot = 0
    for s in test_seqs:
        xs = s[key]
        for a, b in zip(xs, xs[1:]):
            tot += 1
            if table.get(a) == b:
                hit += 1
    return hit / tot if tot else 0.0


def ceiling_2gram(seqs, key):
    """In-distribution 2-gram ceiling: train AND test on the same pool (the no-transfer upper bound)."""
    return transfer_accuracy(seqs, seqs, key)


def _parse(path):
    args = default_tokenizer_args(
        generator_pass=True, instrument_program=True, melody_skeleton=True
    )
    return next(
        RegLogParser(args=args).parse(path, max_perm=1, require_pq=False, reparse=True),
        None,
    )


def measure(paths, min_len=4):
    """Train/test-split the dumps by index parity (held-out by dump) and return the cross-tune transfer
    accuracy for intervals vs absolute notes, plus the interval in-pool 2-gram ceiling and counts.
    """
    train, test = [], []
    for k, path in enumerate(paths):
        df = _parse(path)
        if df is None:
            continue
        for _voice, seq in extract_sequences(df).items():
            if len(seq["intervals"]) < min_len:
                continue
            (train if k % 2 == 0 else test).append(seq)
    return {
        "n_train": len(train),
        "n_test": len(test),
        "interval_transfer": transfer_accuracy(train, test, "intervals"),
        "absolute_transfer": transfer_accuracy(train, test, "notes"),
        "interval_ceiling": ceiling_2gram(train + test, "intervals"),
    }


def measure_triage(paths, min_len=4):
    """Layer-3 pre-screen: held-out next-interval transfer for the de-multiplexed VOICE-major lanes vs
    the deployed FRAME-major interleaved stream. de_mux_gain > 0 means contiguous voice lanes surface
    the melody's own history and justify the byte-exact reorder (the work order's mandatory gate).
    """
    vm_train, vm_test, fm_train, fm_test = [], [], [], []
    for k, path in enumerate(paths):
        df = _parse(path)
        if df is None:
            continue
        for seq in extract_sequences(df).values():
            if len(seq["intervals"]) >= min_len:
                (vm_train if k % 2 == 0 else vm_test).append(seq)
        inter = extract_interleaved(df)
        if len(inter["intervals"]) >= min_len:
            (fm_train if k % 2 == 0 else fm_test).append(inter)
    voice_major = transfer_accuracy(vm_train, vm_test, "intervals")
    frame_major = transfer_accuracy(fm_train, fm_test, "intervals")
    return {
        "frame_major_transfer": frame_major,
        "voice_major_transfer": voice_major,
        "de_mux_gain": voice_major - frame_major,
    }
