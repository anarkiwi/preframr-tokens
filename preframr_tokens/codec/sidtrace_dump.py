"""Produce the per-frame SID register dump from the ``preframr-sidtrace`` tool.

This is the headlessvice-replacement dump source (Phase 2a). ``sidtrace`` is a
cycle-accurate libsidplayfp recorder that, given a ``.sid``, emits a
``<prefix>.sidwr.bin`` of every SID register write as packed 12-byte records
``{int64 cycle, uint16 addr, uint8 reg, uint8 val}`` (see the preframr-sidtrace
README). ``write_dump()`` converts that into the *same* ``.dump.parquet`` schema
the VICE ``vsiddump.py`` corpus path emits and that
``lane_grammar.per_frame_state`` consumes: columns ``clock, irq, chipno, reg,
val``.

Dump contract (``design/infra/libsidplay_callgraph_recovery_design.md`` and the
superseded-but-contract-valid ``sid_to_dump_emulator_design.md`` in preframr-xpt):

  * keep regs 0-24 (``reg = addr & 0x1F``; D400-D418 maps to 0-24);
  * apply the SID don't-care masks so the logged byte equals what the chip sees,
    matching VICE's logged value (a pure logging-convention alignment, NOT an
    emulation change): PW-high (regs 3/10/17) ``& 0x0F``; filter-cutoff-low
    (reg 21) ``& 0x07``; res/filt-route (reg 23) ``& 0xF7``;
  * the ``irq`` frame id is the play-call burst-start cycle that the write falls
    in (the same way the VICE dump keys rows by the IRQ cycle); ``per_frame_state``
    re-bins ``clock`` onto the cpf grid itself, so the dump is NOT pre-squeezed --
    it carries every write, exactly like the VICE corpus dump.

The sidtrace binary is obtained, in order: ``$SIDTRACE_BIN`` (explicit path), a
``sidtrace`` on ``$PATH``, ``$PREFRAMR_SIDTRACE_ROOT/build/sidtrace``, or a
default sibling clone ``../preframr-sidtrace/build/sidtrace``. ``ensure_binary``
will ``git clone`` + ``make`` preframr-sidtrace on demand when allowed.
"""

import os
import subprocess

import numpy as np
import pandas as pd

# Packed little-endian sidwr record (mirrors src/sidtrace.cpp / sidtrace_records.py).
SIDWR_DT = np.dtype([("cycle", "<i8"), ("addr", "<u2"), ("reg", "u1"), ("val", "u1")])

SID_BASE = 0xD400
NREG = 25

# SID don't-care masks: bits the chip ignores, which VICE logs already masked.
PW_HI = (3, 10, 17)  # pulse-width high nibble: 12-bit PW -> & 0x0F
FC_LO = 21  # filter cutoff low: 3-bit -> & 0x07
RES = 23  # res/filt routing: bit 3 unused -> & 0xF7

SIDTRACE_REPO = "https://github.com/anarkiwi/preframr-sidtrace"


def _mask(reg, val):
    """Apply the SID don't-care mask for ``reg`` so it equals VICE's logged byte."""
    if reg in PW_HI:
        return val & 0x0F
    if reg == FC_LO:
        return val & 0x07
    if reg == RES:
        return val & 0xF7
    return val


def _burst_starts(cyc, gap=2000):
    """First-write cycle of each play-call burst (inter-write gap > ``gap``)."""
    if len(cyc) == 0:
        return np.empty(0, dtype=np.int64)
    bnd = np.nonzero(np.diff(cyc) > gap)[0]
    return np.concatenate(([cyc[0]], cyc[bnd + 1])).astype(np.int64)


def sidwr_to_frame(sidwr_path):
    """Read a ``.sidwr.bin`` into the (clock, irq, chipno, reg, val) columns.

    Returns a ``pandas.DataFrame`` with the same schema/dtypes as the VICE
    ``.dump.parquet``. ``irq`` is the burst-start cycle of the write's play call
    (frame id); ``chipno`` is 0 (sidtrace traces SID #0).
    """
    a = np.fromfile(sidwr_path, dtype=SIDWR_DT)
    # Keep SID #0 register space ($D400-$D41F); reg already = addr & 0x1F.
    keep = (a["addr"] >= SID_BASE) & (a["addr"] < SID_BASE + 0x20)
    a = a[keep]
    cyc = a["cycle"].astype(np.int64)
    reg = a["reg"].astype(np.int64)
    val = a["val"].astype(np.int64)
    val = np.array([_mask(int(r), int(v)) for r, v in zip(reg, val)], dtype=np.int64)
    # Frame id == the play-call burst start each write belongs to (VICE's irq).
    starts = _burst_starts(cyc)
    # searchsorted(side="right")-1 maps each cycle to the burst start <= it.
    idx = np.clip(np.searchsorted(starts, cyc, side="right") - 1, 0, None)
    irq = starts[idx] if len(starts) else cyc
    return pd.DataFrame(
        {
            "clock": pd.array(cyc, dtype="UInt32"),
            "irq": pd.array(irq, dtype="UInt32"),
            "chipno": pd.array(np.zeros(len(cyc), dtype=np.int64), dtype="UInt8"),
            "reg": pd.array(reg, dtype="UInt8"),
            "val": pd.array(val, dtype="UInt8"),
        }
    )


def write_dump(sidwr_path, out_parquet):
    """Convert ``sidwr_path`` to a VICE-schema ``.dump.parquet`` at ``out_parquet``."""
    df = sidwr_to_frame(sidwr_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_parquet)) or ".", exist_ok=True)
    df.to_parquet(out_parquet)
    return out_parquet


def _candidate_bins():
    if os.environ.get("SIDTRACE_BIN"):
        yield os.environ["SIDTRACE_BIN"]
    root = os.environ.get("PREFRAMR_SIDTRACE_ROOT")
    if root:
        yield os.path.join(root, "build", "sidtrace")
    # Default: a sibling clone next to this repo.
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))  # .../preframr-tokens
    yield os.path.join(os.path.dirname(repo), "preframr-sidtrace", "build", "sidtrace")


def find_binary():
    """Return a usable ``sidtrace`` binary path, or ``None`` if none is present."""
    for cand in _candidate_bins():
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    from shutil import which

    return which("sidtrace")


def ensure_binary(build=False):
    """Locate the sidtrace binary; clone+make preframr-sidtrace if ``build``.

    The clone target is ``$PREFRAMR_SIDTRACE_ROOT`` or a sibling ``preframr-sidtrace``
    dir. Building needs the documented toolchain (C++17, autotools, ``xa65``).
    """
    found = find_binary()
    if found:
        return found
    if not build:
        return None
    root = os.environ.get("PREFRAMR_SIDTRACE_ROOT")
    if not root:
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.dirname(os.path.dirname(here))
        root = os.path.join(os.path.dirname(repo), "preframr-sidtrace")
    if not os.path.isdir(os.path.join(root, ".git")):
        subprocess.run(["git", "clone", SIDTRACE_REPO, root], check=True)
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"], cwd=root, check=True
    )
    subprocess.run(["make"], cwd=root, check=True)
    return os.path.join(root, "build", "sidtrace")


def render_sidwr(sid_path, subtune, nframes, out_prefix, roms=(), build=False):
    """Run sidtrace on ``sid_path`` -> ``<out_prefix>.sidwr.bin``; return its path.

    ``SIDTRACE_NOBUS=1`` is set so the (large) ``.bus.bin`` is skipped: only the
    register-write log is needed for the dump.
    """
    binary = ensure_binary(build=build)
    if not binary:
        raise RuntimeError(
            "sidtrace binary not found; set SIDTRACE_BIN or PREFRAMR_SIDTRACE_ROOT, "
            "or call ensure_binary(build=True)"
        )
    cli = [binary, sid_path, str(subtune), str(nframes), out_prefix, *roms]
    env = dict(os.environ, SIDTRACE_NOBUS="1")
    subprocess.run(cli, check=True, capture_output=True, env=env)
    sidwr = out_prefix + ".sidwr.bin"
    if not os.path.exists(sidwr):
        raise RuntimeError(f"sidtrace produced no sidwr.bin for {sid_path}")
    return sidwr


def render_dump(sid_path, subtune, nframes, out_parquet, roms=(), build=False):
    """End-to-end: sidtrace ``sid_path`` -> VICE-schema ``out_parquet``."""
    import tempfile

    with tempfile.TemporaryDirectory() as work:
        prefix = os.path.join(work, "st")
        sidwr = render_sidwr(sid_path, subtune, nframes, prefix, roms=roms, build=build)
        return write_dump(sidwr, out_parquet)


def boot_lag(a, b, span=80):
    """Integer-frame offset of per-frame-state ``a`` vs ``b`` (max-overlap match).

    The two emulators' init/boot prologs differ (VICE logs an init write at the
    very first cycles; libsidplayfp's first writes can appear dozens of frames in),
    so ``per_frame_state`` anchors their frame-0 a constant integer number of
    frames apart. This recovers that lag the same way the corpus census does. A
    wide ``span`` is needed: some players (Jetta) have a ~54-frame boot prolog.
    """
    best = (-1, 0)
    for lag in range(-span, span + 1):
        n = ok = 0
        for i in range(max(0, lag), min(len(a), len(b) + lag)):
            j = i - lag
            if 0 <= j < len(b) and np.array_equal(a[i], b[j]):
                ok += 1
            if 0 <= j < len(b):
                n += 1
        if n > 20 and ok > best[0]:
            best = (ok, lag)
    return best[1]


def frame_parity(a, b, span=80):
    """(matched, overlap, lag) of two per-frame-state arrays at the boot lag."""
    lag = boot_lag(a, b, span=span)
    n = ok = 0
    for i in range(max(0, lag), min(len(a), len(b) + lag)):
        j = i - lag
        if 0 <= j < len(b):
            n += 1
            if np.array_equal(a[i], b[j]):
                ok += 1
    return ok, n, lag
