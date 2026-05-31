"""Pure helpers for the WAVETABLE codebook (design/wavetable_codebook_encoding): a note-relative offset
program is RLE (offset, hold) steps with a loop point (prefix replays once, body cycles); ``factorise``
mines the smallest looping unit and ``unroll`` replays it to a frame length after a per-hit onset-strip
lead, shared by the encoder verify and ``WavetableDecoder`` so they cannot disagree (offsets stored
exactly -> byte-identical to the RESID replaced)."""

__all__ = ["rle", "factorise", "unroll", "program_key"]


def rle(seq):
    """Run-length encode a per-frame sequence into ``[(value, hold)]`` of consecutive duplicates."""
    steps = []
    for x in seq:
        x = int(x)
        if steps and steps[-1][0] == x:
            steps[-1] = (x, steps[-1][1] + 1)
        else:
            steps.append((x, 1))
    return steps


def _step_period(steps, p, b):
    return all(steps[i] == steps[p + (i - p) % b] for i in range(p, len(steps)))


def factorise(core):
    """Mine ``(steps, loop)`` for a per-frame offset ``core``: RLE to steps (a hold is not a loop),
    then the smallest body period ``b`` (then smallest prefix ``p``) whose step-tail repeats at least
    twice is a loop -- store prefix + one body period, ``loop=p``; else the whole core is a one-shot
    prefix (``loop=len(steps)``)."""
    if not core:
        return [], 0
    steps = rle(core)
    m = len(steps)
    for b in range(1, m // 2 + 1):
        for p in range(0, m - 2 * b + 1):
            if _step_period(steps, p, b):
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
