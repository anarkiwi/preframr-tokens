"""Per-voice INSTRUMENT-program recovery for the ctrl/AD/SR (env) write stream: a
voice's ordered env writes are segmented into note instances by gate-on edges, each
split into an onset-anchored HEAD + a note-end-anchored TAIL (the release block). An
instrument ``(head, tail)`` is DEFined inline then REFerenced backward; a non-fitting
note falls back to RAW (so the stream stays byte-exact); pre-onset writes are a LEAD."""

from __future__ import annotations

import numpy as np

from .oracle import ENV_REGS

VOICE = {4: 0, 5: 0, 6: 0, 11: 1, 12: 1, 13: 1, 18: 2, 19: 2, 20: 2}
CTRL = {0: 4, 1: 11, 2: 18}

PW_LO = {0: 2, 1: 9, 2: 16}
PW_HI = {0: 3, 1: 10, 2: 17}
PW_REGS = frozenset(PW_LO.values()) | frozenset(PW_HI.values())

_ENV_SET = frozenset(ENV_REGS)
_PW_VOICE = {**{r: v for v, r in PW_LO.items()}, **{r: v for v, r in PW_HI.items()}}

PW_SYNC_REUSE = 0.4
PW_SYNC_HEAD = 6
PW_SYNC_MIN_ONSETS = 6


def _voice_of(reg):
    return VOICE.get(reg, _PW_VOICE.get(reg))


def synced_pw_voices(grid: np.ndarray, ew) -> frozenset[int]:
    """Voices whose pulse-width is SYNCED to the instrument (resets at each onset, so the
    PW trajectory belongs with the note rather than a free-running lane). Detected from
    the settled grid: the per-onset PW head trajectories repeat (distinct heads / onsets
    below :data:`PW_SYNC_REUSE`). A free-running or static PW voice is excluded -- it
    keeps PW in its own settled lane."""
    if grid.shape[0] < 8:
        return frozenset()
    onsets_by_voice = _voice_onsets(ew)
    out = set()
    for v in range(3):
        base = 7 * v
        ctrl = grid[:, base + 4]
        pw = (grid[:, base + 2] + 256 * (grid[:, base + 3] & 0xF)).astype(np.int64)
        if int(np.sum((ctrl & 0x40) > 0)) < 20 or len(np.unique(pw)) < 3:
            continue
        onsets = [
            f
            for f in onsets_by_voice.get(v, [])
            if 2 <= f < grid.shape[0] - PW_SYNC_HEAD
        ]
        if len(onsets) < PW_SYNC_MIN_ONSETS:
            continue
        heads = [tuple((pw[f : f + PW_SYNC_HEAD] - pw[f]).tolist()) for f in onsets]
        if len(set(heads)) / len(heads) < PW_SYNC_REUSE:
            out.add(v)
    return frozenset(out)


def _voice_onsets(ew) -> dict[int, list[int]]:
    """Gate-on onset frames per voice from the ordered env writes."""
    out: dict[int, list[int]] = {0: [], 1: [], 2: []}
    gate = {0: 0, 1: 0, 2: 0}
    for f, r, val in ew:
        if r in CTRL.values():
            v = VOICE[r]
            ng = val & 1
            if ng == 1 and gate[v] == 0:
                out[v].append(f)
            gate[v] = ng
    return out


def pw_writes(grid: np.ndarray, voices) -> list[tuple[int, int, int]]:
    """Per-frame PW-change pseudo-writes (``(frame, reg, val)`` on the pw lo/hi regs) for
    the given synced voices, so they can be folded into those voices' instruments."""
    out: list[tuple[int, int, int]] = []
    for v in sorted(voices):
        for reg in (PW_LO[v], PW_HI[v]):
            col = grid[:, reg]
            prev = 0
            for f in range(grid.shape[0]):
                if int(col[f]) != prev:
                    out.append((f, int(reg), int(col[f])))
                    prev = int(col[f])
    return out


def pw_skip_lanes(voices) -> tuple[int, ...]:
    """The non-env lane ids to skip when the given voices' PW is folded (pw lanes are
    ``NUM_NONENV//... ``: lanes 3,4,5 for voices 0,1,2)."""
    return tuple(3 + v for v in sorted(voices))


def _merge_frame_stable(pw, env):
    """Merge folded pw pseudo-writes with env writes: per frame the pw writes (a non-env
    lane, emitted before the env writes in the corrected order) come first, then the env
    writes in their source order. Both inputs are frame-ascending."""
    by_frame_pw: dict[int, list] = {}
    for w in pw:
        by_frame_pw.setdefault(w[0], []).append(w)
    by_frame_env: dict[int, list] = {}
    for w in env:
        by_frame_env.setdefault(w[0], []).append(w)
    out = []
    for f in sorted(set(by_frame_pw) | set(by_frame_env)):
        out.extend(by_frame_pw.get(f, []))
        out.extend(by_frame_env.get(f, []))
    return out


def segment_voice(ew, v, extra=()):
    """Segment one voice's ordered env writes (plus any folded ``extra`` writes, e.g.
    synced PW) into note instances by gate-on edges on its ctrl reg. Returns
    ``(instances, lead, onsets)`` where an instance is ``(onset, end, sig)`` (``end`` is
    the next onset frame or ``None`` for the last), ``sig`` the instance's ordered writes
    (stable by frame then reg), and ``lead`` the writes before the first onset."""
    env = [(f, r, val) for f, r, val in ew if VOICE.get(r) == v]
    pw = sorted((w for w in extra if _voice_of(w[1]) == v), key=lambda w: (w[0], w[1]))
    writes = _merge_frame_stable(pw, env)
    ctrl_reg = CTRL[v]
    gate = 0
    onsets = []
    for f, r, val in writes:
        if r == ctrl_reg:
            ng = val & 1
            if ng == 1 and gate == 0:
                onsets.append(f)
            gate = ng
    instances = []
    for i, on in enumerate(onsets):
        end = onsets[i + 1] if i + 1 < len(onsets) else None
        hi = end if end is not None else 1 << 30
        sig = [(f, r, val) for f, r, val in writes if on <= f < hi]
        instances.append((on, end, sig))
    lead = [(f, r, val) for f, r, val in writes if not onsets or f < onsets[0]]
    return instances, lead, onsets


def split_head_tail(sig, on, dur):
    """Split an instance's writes into an onset-anchored head ``(offset>=0, reg, val)``
    + a note-end-anchored tail ``(offset_from_end<0, reg, val)``: each write anchors to
    whichever of the onset or the note end (``on + dur``) it sits closer to, so the same
    instrument reproduces byte-exact at any duration. ``dur is None`` (the last note)
    anchors everything to the onset."""
    if dur is None:
        return [(f - on, r, val) for f, r, val in sig], []
    head, tail = [], []
    end = on + dur
    for f, r, val in sig:
        from_end = f - end
        from_start = f - on
        if -from_end <= from_start:
            tail.append((from_end, r, val))
        else:
            head.append((from_start, r, val))
    return head, tail


def _replay(head, tail, on, dur):
    """Replay a ``(head, tail)`` instrument at ``onset``/``dur`` -> its ordered writes."""
    rep = [(on + off, r, val) for off, r, val in head]
    if dur is not None:
        rep += [(on + dur + off, r, val) for off, r, val in tail]
    return rep


def encode_voice(ew, v, extra=()):
    """One voice's ordered env writes (plus folded ``extra`` synced-PW writes) -> its
    instrument-item event stream: a ``VSEL`` head, an optional ``LEAD`` seed, then per
    note a ``DEF`` (first use), ``REF`` (backward, by inter-onset ``dt`` and duration),
    or ``RAW`` (non-fitting) item. Byte-exact: replaying reproduces the voice's writes.
    """
    instances, lead, _ = segment_voice(ew, v, extra=extra)
    if not instances and not lead:
        return []
    items: list[tuple] = [("VSEL", v)]
    if lead:
        items.append(("LEAD", tuple(lead)))
    key_to_id: dict[tuple, int] = {}
    prev_on = lead[-1][0] if lead else 0
    for on, end, sig in instances:
        dur = (end - on) if end is not None else None
        head, tail = split_head_tail(sig, on, dur)
        dt = on - prev_on
        prev_on = on
        durtok = dur if dur is not None else 0
        if _replay(head, tail, on, dur) != list(sig):
            items.append(("RAW", dt, tuple((f - on, r, val) for f, r, val in sig)))
            continue
        key = (tuple(head), tuple(tail))
        iid = key_to_id.get(key)
        if iid is not None:
            items.append(("REF", dt, iid, durtok))
        else:
            key_to_id[key] = len(key_to_id)
            items.append(("DEF", dt, durtok, tuple(head), tuple(tail)))
    return items


def env_events(ew, pw_extra=()):
    """All three voices' instrument-item event streams, concatenated voice by voice.
    This is the env-half event stream (the per-note instrument program) that replaces
    the raw per-frame env ``WRITE`` events; ``pw_extra`` are folded synced-PW writes."""
    out: list[tuple] = []
    for v in range(3):
        out.extend(encode_voice(ew, v, extra=pw_extra))
    return out


def decode_voice(items):
    """Replay one voice's instrument-item stream -> its ordered ``(frame, reg, val)``
    env writes (absolute frame). Inverse of :func:`encode_voice`."""
    ew: list[tuple[int, int, int]] = []
    inst_table: list[tuple[tuple, tuple]] = []
    frame = 0
    for it in items:
        kind = it[0]
        if kind == "VSEL":
            frame = 0
            continue
        if kind == "LEAD":
            for f, r, val in it[1]:
                ew.append((f, r, val))
            frame = it[1][-1][0] if it[1] else 0
            continue
        if kind == "RAW":
            _, dt, sig = it
            on = frame + dt
            for off, r, val in sig:
                ew.append((on + off, r, val))
            frame = on
            continue
        if kind == "DEF":
            _, dt, durtok, head, tail = it
            inst_table.append((head, tail))
            iid = len(inst_table) - 1
        else:
            _, dt, iid, durtok = it
            head, tail = inst_table[iid]
        on = frame + dt
        dur = durtok if durtok else None
        for off, r, val in head:
            ew.append((on + off, r, val))
        if dur is not None:
            end = on + dur
            for off, r, val in tail:
                ew.append((end + off, r, val))
        frame = on
    return ew


def decode_env(events):
    """Replay an env-half instrument-item event stream (all voices) -> ``(env_writes,
    pw_writes)``: the merged ordered ctrl/AD/SR writes (voice-major, within-voice source
    order) and the folded synced-PW pseudo-writes split back out by reg. Inverse of
    :func:`env_events`."""
    voices: list[list[tuple]] = [[], [], []]
    cur = -1
    for it in events:
        kind = it[0]
        if kind == "VSEL":
            cur = it[1]
            voices[cur].append(it)
        elif kind in ("LEAD", "DEF", "REF", "RAW") and cur >= 0:
            voices[cur].append(it)
    env_out: list[tuple[int, int, int]] = []
    pw_out: list[tuple[int, int, int]] = []
    for v in range(3):
        for f, r, val in decode_voice(voices[v]):
            (pw_out if r in PW_REGS else env_out).append((f, r, val))
    return env_out, pw_out
