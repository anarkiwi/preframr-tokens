"""Single-file ``.sid`` -> (per-frame register dump, bus trace) via preframr-sidtrace.

One run of the deterministic ``preframr-sidtrace`` binary (PR #5) over a ``.sid``
emits BOTH artifacts the generic recovery needs, internally self-consistent
(same emulator, same run):

  * ``<prefix>.sidwr.bin`` -- every SID register write as packed records
    ``int64 cyc, uint16 addr, uint8 reg, uint8 val`` -- the per-frame register
    DUMP (the same role the corpus ``.dump.parquet`` plays), and
  * ``<prefix>.bus.bin`` -- the full CPU bus trace (:data:`bustrace.BUS_DT`) --
    the provenance substrate the GENERIC recovery reads.

:func:`sid_to_dump_and_bustrace` runs the binary once (subprocess, exactly as the
hand backends shell out to ``headlessvice``) and returns the dump as an
``(nframes, 25)`` register array -- the SAME schema and SAME blit-group framing
:func:`busstate.per_frame_state_from_bus` produces, so the in-process dump and
the bus-state are byte-identical by construction (no pre-rendered dump file, no
fragile frame-shift).  The binary location is taken from ``SIDTRACE_BIN`` (the
built path as default); a clear error is raised when it is absent so the default,
self-contained CI -- which never invokes the binary -- is unaffected (HARD RULE
#0: the dump is the real emulator output, never fabricated).
"""

import os
import subprocess
import tempfile

import numpy as np

from preframr_tokens.bacc.generic.busstate import NREG, PW_HI, _cadence_state
from preframr_tokens.codec.lsp_validate import (
    BURST_GAP,
    detect_play_period,
    first_play_cycle,
)

# Packed ``.sidwr.bin`` record: int64 cyc, uint16 addr, uint8 reg, uint8 val.
SIDWR_DT = np.dtype([("cyc", "<i8"), ("addr", "<u2"), ("reg", "u1"), ("val", "u1")])

# The built binary in the sibling preframr-sidtrace checkout (REUSE; never rebuilt).
_DEFAULT_SIDTRACE_BIN = "/scratch/anarkiwi/preframr/preframr-sidtrace/build/sidtrace"

# Real C64 ROM images (KERNAL / BASIC / CHARGEN).  An RSID tune with NO play
# address (``play == 0``) installs its OWN interrupt handler during init and the
# REAL KERNAL/BASIC ROM runs it (a BASIC-stub tune is literally a BASIC program at
# $0801 the BASIC ROM interprets); without the real ROMs libsidplayfp's internal
# pseudo-ROM never reaches the player and the trace collapses to a single boot SID
# write (the "sidtrace produced no frames" failure).  A PSID tune with an explicit
# play address never executes ROM code, so passing the ROMs is byte-exact-identical
# for the whole PSID corpus and ONLY rescues the self-IRQ / BASIC-stub RSID tunes.
# The standard VICE install layout; override the directory with ``SIDTRACE_ROM_DIR``
# (or any of the three files individually) -- absent ROMs degrade gracefully to the
# prior no-ROM behaviour (the binary's ROM args are optional positional args).
_VICE_ROM_DIRS = (
    "/usr/local/share/vice/C64",
    "/usr/share/vice/C64",
    "/usr/lib/vice/C64",
)
_ROM_FILES = (
    ("SIDTRACE_KERNAL", "kernal-901227-03.bin"),
    ("SIDTRACE_BASIC", "basic-901226-01.bin"),
    ("SIDTRACE_CHARGEN", "chargen-901225-01.bin"),
)


def sidtrace_bin():
    """The ``preframr-sidtrace`` binary path (``SIDTRACE_BIN`` env or the built
    default).  Returns ``None`` when no binary is present -- the caller skips the
    sid-only path so the default render-free CI gate stays self-contained."""
    cand = os.environ.get("SIDTRACE_BIN", _DEFAULT_SIDTRACE_BIN)
    return cand if cand and os.path.exists(cand) else None


def _rom_dir():
    """Directory holding the real C64 ROM images (``SIDTRACE_ROM_DIR`` env or the
    first standard VICE install path that exists), or ``None``."""
    env = os.environ.get("SIDTRACE_ROM_DIR")
    cands = (env,) + _VICE_ROM_DIRS if env else _VICE_ROM_DIRS
    for cand in cands:
        if cand and os.path.isdir(cand):
            return cand
    return None


def sidtrace_roms():
    """The ``[kernal, basic, chargen]`` ROM-image paths to pass to the binary, or
    ``None`` when the real ROMs are not installed.

    The binary takes the three ROMs as optional positional args (5..7); a tune that
    installs its own IRQ or is a BASIC stub (``play == 0``) only RUNS its player when
    the real KERNAL/BASIC ROM is present.  Discovery is via ``SIDTRACE_ROM_DIR`` (or
    per-file ``SIDTRACE_KERNAL`` / ``SIDTRACE_BASIC`` / ``SIDTRACE_CHARGEN``
    overrides) falling back to the standard VICE layout.  All three must resolve to
    existing files; otherwise ``None`` is returned and the binary runs with its
    internal pseudo-ROM exactly as before."""
    rom_dir = _rom_dir()
    paths = []
    for env_key, default_file in _ROM_FILES:
        cand = os.environ.get(env_key)
        if cand is None and rom_dir is not None:
            cand = os.path.join(rom_dir, default_file)
        if not cand or not os.path.exists(cand):
            return None
        paths.append(cand)
    return paths


def _tune_play_addr(sid_path):
    """The tune's PSID/RSID play address (header bytes ``$0c..$0d``), or ``None``
    when the header cannot be read.  ``play == 0`` is the "no play address -- run
    from the installed IRQ vector" convention (a self-IRQ player or a BASIC stub)."""
    try:
        with open(sid_path, "rb") as handle:
            head = handle.read(14)
    except OSError:
        return None
    if len(head) < 14 or head[0:4] not in (b"PSID", b"RSID"):
        return None
    return (head[12] << 8) | head[13]


def _tune_needs_roms(sid_path):
    """Whether a tune requires the REAL C64 ROMs to trace (vs libsidplayfp's
    internal pseudo-ROM).

    ONLY a ``play == 0`` tune -- one with no host-called play routine, which instead
    installs its own CIA/raster IRQ during init (or is a BASIC program the BASIC ROM
    interprets) -- needs the real KERNAL/BASIC to run its player; without them the
    trace collapses to a single boot SID write.  A tune with an explicit play address
    is driven directly and must NOT be given the real ROMs: some such PSID tunes read
    ROM/CIA cells as data (RNG seed, jump tables) and the real-ROM values perturb the
    byte-exact per-frame state.  Gating ROM injection on ``play == 0`` rescues exactly
    the self-IRQ / BASIC-stub tunes while leaving every play-addressed tune (the whole
    committed corpus) byte-identical to the prior no-ROM behaviour."""
    return _tune_play_addr(sid_path) == 0


def sidwr_state(sidwr_path, t0=None):
    """Parse a ``.sidwr.bin`` into the per-frame ``(nframes, 25)`` register array.

    Vectorised load of the packed records, then framing at the tune's detected play
    PERIOD with forward-fill -- one row per play-call, the last value written to each
    register surviving as that frame's value and held between play-calls -- the SAME
    grid the corpus training dumps use (:func:`codec.lane_grammar.per_frame_state` and
    :func:`busstate.per_frame_state_from_bus`).  The PW-high registers are masked to
    their 4 SID-significant bits (a pure chip semantic) and frame 0 is the play-call at
    the tune's first steady play cycle (``t0``).

    Two framings reproduce that grid; the cheaper, byte-identical one is chosen:

      * BLIT-GROUP (the default): a SID-write gap above :data:`BURST_GAP` cycles starts
        a new play-call.  An EVERY-FRAME writer (Grid_Runner, A Mind, ...) exposes one
        gap-group per play period, so this yields exactly one row per period -- the
        grid -- and is kept verbatim (byte-identical to the prior behaviour, so the
        every-frame tunes and the env-gated recovery fixtures are unchanged).
      * CADENCE: ``round((cyc - t0) / cpf)`` bins writes onto the play-period grid and
        forward-fills held values down the frame axis (:func:`busstate._cadence_state`,
        identical to the corpus :func:`per_frame_state`).  A SPARSE writer that plays
        EVERY raster frame but only WRITES the SID on a fraction of them (Master
        Composer: 234 write-bursts across ~2300 play-calls) collapses under blit-group
        framing to one row per WRITE -- dropping the held-value frames between writes --
        which blows the fixed boot/pool/header costs up to several tok/frame.  When the
        blit-group count is materially below the play-period span (the bursts span far
        more periods than there are bursts), the tune is this sparse writer and the
        cadence framing -- one row per play-call, values held between writes -- is used
        instead, so the fixed costs amortize over the true frame count.

    Returns ``(state, t0)``; ``state`` is ``None`` for a trace with no SID writes.
    """
    recs = np.fromfile(sidwr_path, dtype=SIDWR_DT)
    recs = recs[recs["reg"] < NREG]
    if len(recs) == 0:
        return None, None
    cyc = recs["cyc"].astype(np.int64)
    reg = recs["reg"].astype(int)
    val = recs["val"].astype(int).copy()
    for pw_reg in PW_HI:
        val[reg == pw_reg] &= 0x0F
    cpf = detect_play_period(cyc)
    starts = _frame_starts(cyc)
    ends = np.concatenate((starts[1:], [len(cyc)]))
    gstart_cyc = cyc[starts]
    if t0 is None:
        t0 = first_play_cycle(cyc, cpf)
    boot_off = int(np.searchsorted(gstart_cyc, t0 - cpf / 2))
    nframes = len(starts) - boot_off
    # SPARSE/DENSE-WRITER DETECTION.  The play-period span (play-calls from frame 0
    # to the last write) is the true frame count when the player calls play every
    # period; the blit-group count is the number of WRITE bursts.  Two ways blit-group
    # framing miscounts a continuously-played tune, both fixed by re-framing at the
    # play period (cadence) with forward-fill -- exactly how the corpus dump is framed:
    #
    #   * SPARSE writer (Master Composer: 234 bursts across ~2300 play-calls): many
    #     bursts but covering FAR more periods than there are bursts, so blit-group
    #     framing (one row per write) drops the held frames between writes.
    #   * DENSE writer (Master_Blaster / MacGyver / Das_Model / James_Brown / a
    #     self-IRQ RSID player): writes the SID CONTINUOUSLY with no inter-play gap
    #     above BURST_GAP, so EVERY play-call coalesces into ONE blit group and
    #     blit-group framing degenerates to a single frame even though the trace spans
    #     thousands of cadence frames.  This is the "sidtrace produced no frames"
    #     failure (``len(state) < 2``) -- not a tiny tune, the player just never leaves
    #     a quiet gap.  Mirror :func:`busstate.per_frame_state_from_bus`'s same guard.
    span_frames = int(round((int(cyc[-1]) - t0) / cpf)) + 1 if cpf else 0
    sparse = nframes >= 2 and span_frames > nframes and nframes < 0.9 * span_frames
    dense = nframes < 2 and span_frames >= 2
    if cpf and (sparse or dense):
        # Cadence framing (corpus-consistent): bin on the play-period grid + forward-fill.
        # Anchored at ``t0`` so frame 0 is the first steady play-call, identical to the
        # blit-group anchor and to :func:`per_frame_state` / :func:`_cadence_state`.
        return _cadence_state(cyc, reg, val, t0, cpf), t0
    seq = np.zeros((nframes, NREG), dtype=np.int64)
    cur = [0] * NREG
    for group in range(len(starts)):
        for k in range(starts[group], ends[group]):
            cur[reg[k]] = val[k]
        if group >= boot_off:
            seq[group - boot_off] = cur
    return seq, t0


def _frame_starts(cyc, gap=BURST_GAP):
    """Indices of the first write of each play-call burst (blit-group boundary)."""
    if len(cyc) == 0:
        return np.empty(0, dtype=np.int64)
    big = np.nonzero(np.diff(cyc) > gap)[0]
    return np.concatenate(([0], big + 1))


def run_sidtrace(sid_path, out_prefix, subtune=1, nframes=200, sidtrace_path=None):
    """Run ``preframr-sidtrace`` once over ``sid_path`` (subtune is 1-based),
    emitting ``<out_prefix>.sidwr.bin`` (the small timestamped SID-write stream,
    the render gate) and ``<out_prefix>.distill.bin`` (the compact SDST artifact
    the SMC-correct recovery consumes -- a few KB, NOT the retired multi-GB raw
    bus trace).

    Returns ``(sidwr_path, distill_path)``.  Raises :class:`FileNotFoundError`
    when no binary is available (env-gated; default CI never reaches here)."""
    binary = sidtrace_path or sidtrace_bin()
    if binary is None:
        raise FileNotFoundError(
            "preframr-sidtrace binary not found; set SIDTRACE_BIN to the built "
            f"'sidtrace' (looked for {_DEFAULT_SIDTRACE_BIN})"
        )
    argv = [
        binary,
        str(sid_path),
        str(int(subtune)),
        str(int(nframes)),
        str(out_prefix),
    ]
    # Append the real C64 ROM images ONLY for a tune that needs them: a ``play == 0``
    # self-IRQ / BASIC-stub tune installs its own interrupt handler during init and
    # only RUNS its player when the real KERNAL/BASIC ROM is present (otherwise the
    # trace is a single boot SID write -- the "produced no frames" failure).  A
    # play-addressed tune is driven directly and is left on the internal pseudo-ROM
    # so its byte-exact per-frame state is unchanged (some such tunes read ROM/CIA
    # cells as data, which the real ROMs would perturb).
    if _tune_needs_roms(sid_path):
        roms = sidtrace_roms()
        if roms is not None:
            argv.extend(roms)
    subprocess.run(
        argv,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return f"{out_prefix}.sidwr.bin", f"{out_prefix}.distill.bin"


def sidwr_to_bus(sidwr_path):
    """Synthesise a SID-write-only :data:`bustrace.BUS_DT` record array from a
    ``.sidwr.bin``.

    The full cycle-by-cycle bus trace is no longer emitted (it was GBs/tune); the
    legacy per-register generic recovery (:func:`recover_generic`) needs only the
    SID-write substream to reconstruct the per-frame register state, which the
    small ``.sidwr.bin`` carries verbatim.  Reads are absent, so the optional
    note-table-by-read enhancer simply does not fire -- the per-register fit path
    still covers the output.  The SMC-correct song-data recovery uses the
    ``.distill.bin`` artifact, not this synthesised stream.
    """
    from preframr_tokens.bacc.generic.bustrace import BUS_DT  # local: avoid cycle

    recs = np.fromfile(sidwr_path, dtype=SIDWR_DT)
    bus = np.zeros(len(recs), dtype=BUS_DT)
    bus["cyc"] = recs["cyc"]
    bus["addr"] = recs["addr"]
    bus["val"] = recs["val"]
    bus["rw"] = 1  # SID writes
    return bus


def sid_to_dump_and_bustrace(
    sid_path, subtune=1, nframes=200, sidtrace_path=None, out_prefix=None
):
    """From a ``.sid`` ALONE, produce the per-frame register dump, a SID-write
    bus-state array, and the compact distill artifact via ONE
    ``preframr-sidtrace`` run -- no pre-rendered ``.dump.parquet`` input and no
    multi-GB raw trace.

    Returns ``(dump_state, bus, t0, distill_path)`` where ``dump_state`` is the
    ``(nframes, 25)`` register array (the dump), ``bus`` is the SID-write-only
    :data:`bustrace.BUS_DT` array (synthesised from ``.sidwr.bin`` for the legacy
    per-register recovery), ``t0`` is the dump's frame-0 anchor, and
    ``distill_path`` is the SDST artifact for the SMC-correct identity recovery.

    When ``out_prefix`` is given the artifacts persist there; otherwise they land
    in a temporary directory the caller owns.
    """
    if out_prefix is None:
        out_prefix = os.path.join(
            tempfile.mkdtemp(prefix="preframr_sidtrace_"), "trace"
        )
    sidwr_path, distill_path = run_sidtrace(
        sid_path, out_prefix, subtune, nframes, sidtrace_path
    )
    dump_state, t0 = sidwr_state(sidwr_path)
    bus = sidwr_to_bus(sidwr_path)
    return dump_state, bus, t0, distill_path
