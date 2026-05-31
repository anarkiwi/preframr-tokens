"""Pure helpers for the WAVETABLE codebook (design/wavetable_codebook_encoding): a note-relative offset
program is RLE (offset, hold) steps with a loop point (prefix replays once, body cycles); ``factorise``
mines the smallest looping unit and ``unroll`` replays it to a frame length after a per-hit onset-strip
lead, shared by the encoder verify and ``WavetableDecoder`` so they cannot disagree (offsets stored
exactly -> byte-identical to the RESID replaced)."""

from preframr_tokens.macros.rle import run_length_encode

__all__ = ["factorise", "unroll", "program_key"]


def factorise(core):
    """Mine ``(steps, loop)`` for a per-frame offset ``core``: RLE to steps, then the smallest body
    period ``b`` (then smallest prefix ``p``) whose step-tail repeats at least twice is a loop (store
    prefix + one body period, ``loop=p``); else the whole core is a one-shot prefix. The tail from
    ``p`` is period-``b`` iff no ``steps[i] != steps[i-b]`` mismatch sits at/after ``p+b``, so the last
    such mismatch fixes the smallest valid ``p`` in one O(m) pass (O(m^2), vs the cubic scan).
    """
    if not core:
        return [], 0
    steps = run_length_encode(core)
    m = len(steps)
    for b in range(1, m // 2 + 1):
        bad = -1
        for i in range(b, m):
            if steps[i] != steps[i - b]:
                bad = i
        p = max(0, bad - b + 1)
        if p <= m - 2 * b:
            return steps[: p + b], p
    return steps, m


def unroll(steps, loop, length, lead=()):
    """Replay a program to exactly ``length`` per-frame offsets: emit ``lead`` verbatim (the onset
    strip), the prefix ``steps[:loop]`` once, then the body ``steps[loop:]`` cyclically, truncating at
    ``length``. A body that makes no progress (no frames) terminates the loop."""
    out = [int(o) for o in lead]
    if len(out) >= length:
        return out[:length]
    prefix = steps[:loop]
    body = steps[loop:]
    for off, hold in prefix:
        for _ in range(int(hold)):
            if len(out) >= length:
                return out[:length]
            out.append(int(off))
    if not body:
        return out
    while len(out) < length:
        before = len(out)
        for off, hold in body:
            for _ in range(int(hold)):
                if len(out) >= length:
                    return out[:length]
                out.append(int(off))
        if len(out) == before:
            break
    return out[:length]


def program_key(steps, loop):
    """Hashable canonical key for codebook grouping: the program is transposition-invariant (offsets
    are note-relative), so the same wavetable at any base note shares this key."""
    return (tuple((int(o), int(h)) for o, h in steps), int(loop))
