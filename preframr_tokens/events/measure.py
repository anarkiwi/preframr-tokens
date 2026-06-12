"""Collapse / bits measurement for event token streams: compression
comes from BPE over a complete escape-free encoding, not a dictionary+escape. This trains a corpus-global
greedy BPE over the event token streams (the shared statistics ARE the dictionary) and reports the
order-0 coding cost in bits before/after, against the raw 2-byte-per-write floor. It is a measurement, not a
learned LM; the trained-model bits/write will be lower still.
"""

from __future__ import annotations

import collections
import math


def order0_bits(streams: list[list[int]]) -> float:
    """Total order-0 (unigram) coding cost in bits of a corpus of token streams."""
    freq = collections.Counter()
    for s in streams:
        freq.update(s)
    total = sum(freq.values())
    if total == 0:
        return 0.0
    bits = 0.0
    for c in freq.values():
        bits -= c * math.log2(c / total)
    return bits


def bpe_train(streams: list[list[int]], merges: int) -> list[tuple[int, int, int]]:
    """Greedy corpus-global BPE: repeatedly fuse the most frequent adjacent token pair into a new id.
    Returns the merge list ``[(a, b, new_id), ...]`` (apply with :func:`bpe_apply`). Frequent field-
    sequences (a repeated ORDER descriptor, a recurring gesture) collapse into single learned tokens.
    """
    streams = [list(s) for s in streams]
    next_id = (max((max(s) for s in streams if s), default=-1)) + 1
    out: list[tuple[int, int, int]] = []
    for _ in range(merges):
        pairs = collections.Counter()
        for s in streams:
            for i in range(len(s) - 1):
                pairs[(s[i], s[i + 1])] += 1
        if not pairs:
            break
        (a, b), best = pairs.most_common(1)[0]
        if best < 2:
            break
        new_id = next_id
        next_id += 1
        out.append((a, b, new_id))
        for s in streams:
            i = 0
            while i < len(s) - 1:
                if s[i] == a and s[i + 1] == b:
                    s[i] = new_id
                    del s[i + 1]
                else:
                    i += 1
    return out


def bpe_apply(stream: list[int], merges: list[tuple[int, int, int]]) -> list[int]:
    """Apply a trained merge list to one stream."""
    s = list(stream)
    for a, b, new_id in merges:
        i = 0
        while i < len(s) - 1:
            if s[i] == a and s[i + 1] == b:
                s[i] = new_id
                del s[i + 1]
            else:
                i += 1
    return s


def measure(streams: list[list[int]], n_writes: int, merges: int = 2000) -> dict:
    """Measure the collapse of a factored-token corpus against the raw 2-byte/write dump floor."""
    atomic_bits = order0_bits(streams)
    merge_list = bpe_train(streams, merges)
    bpe_streams = [bpe_apply(s, merge_list) for s in streams]
    bpe_bits = order0_bits(bpe_streams)
    raw_bits = n_writes * 16
    n_tok = sum(len(s) for s in streams)
    n_bpe_tok = sum(len(s) for s in bpe_streams)
    return {
        "writes": n_writes,
        "atomic_tokens": n_tok,
        "bpe_tokens": n_bpe_tok,
        "atomic_bits_per_write": atomic_bits / n_writes,
        "bpe_bits_per_write": bpe_bits / n_writes,
        "raw_bits_per_write": 16.0,
        "collapse_vs_raw": raw_bits / bpe_bits,
        "merges": len(merge_list),
    }
