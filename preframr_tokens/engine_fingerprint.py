"""Engine fingerprint vector for a single .dump.parquet."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from preframr_tokens.stfconstants import (
    FC_LO_REG,
    FILTER_REG,
    MAX_REG,
    VOICE_CTRL_REG,
)

DEFAULT_FINGERPRINT_WRITES = 4000

REG_DENSITY_DIM = MAX_REG + 1
DELTA_BUCKETS = 10
CTRL_2GRAM_DIM = 64
CTRL_3GRAM_DIM = 32
FILTER_DIM = 1
FEATURE_DIM = (
    REG_DENSITY_DIM + DELTA_BUCKETS + CTRL_2GRAM_DIM + CTRL_3GRAM_DIM + FILTER_DIM
)

SLICE_REG_DENSITY = slice(0, REG_DENSITY_DIM)
SLICE_DELTA = slice(REG_DENSITY_DIM, REG_DENSITY_DIM + DELTA_BUCKETS)
SLICE_CTRL_2GRAM = slice(
    REG_DENSITY_DIM + DELTA_BUCKETS,
    REG_DENSITY_DIM + DELTA_BUCKETS + CTRL_2GRAM_DIM,
)
SLICE_CTRL_3GRAM = slice(
    REG_DENSITY_DIM + DELTA_BUCKETS + CTRL_2GRAM_DIM,
    FEATURE_DIM - FILTER_DIM,
)
IDX_FILTER_TOUCH = FEATURE_DIM - 1

DELTA_EDGES = np.array(
    [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000],
    dtype=np.int64,
)
assert len(DELTA_EDGES) == DELTA_BUCKETS - 1

CTRL_REGS = frozenset(VOICE_CTRL_REG.values())
FILTER_REGS_3 = frozenset({FC_LO_REG, FC_LO_REG + 1, FILTER_REG})


def _ctrl_state(val: int) -> int:
    """Collapse a CTRL byte into an 8-state code: waveform-bit-index
    (0..3 for TRI/SAW/PULSE/NOISE -- highest set bit in upper nibble)
    times 2 plus the gate bit. Mute / multi-waveform configs collapse
    to the lowest-set-bit waveform (matches how the engine cycles
    waveforms; the dominant musical waveform is captured by the
    """
    waveform_nibble = (val >> 4) & 0xF
    if waveform_nibble == 0:
        wave_idx = 0
    else:
        wave_idx = (waveform_nibble & -waveform_nibble).bit_length() - 1
        wave_idx = min(wave_idx, 3)
    gate = val & 0x1
    return (wave_idx << 1) | gate


def _read_writes(parquet_path: Path, n_writes: int) -> np.ndarray | None:
    """Return the first ``n_writes`` rows as ``(N, 4)`` int64 array,
    columns (clock, irq, reg, val). Returns None on read failure or
    empty parquet (caller decides how to handle).
    """
    try:
        pf = pq.ParquetFile(parquet_path)
    except (OSError, FileNotFoundError) as e:
        logging.warning("%s: open failed: %s", parquet_path, e)
        return None
    if pf.num_row_groups == 0:
        return None
    try:
        table = pf.read(columns=["clock", "irq", "reg", "val"])
    except (KeyError, OSError) as e:
        logging.warning("%s: read failed: %s", parquet_path, e)
        return None
    n = min(n_writes, table.num_rows)
    if n == 0:
        return None
    cols = [
        table.column(c).to_numpy()[:n].astype(np.int64)
        for c in ("clock", "irq", "reg", "val")
    ]
    return np.stack(cols, axis=1)


def _reg_density(regs: np.ndarray) -> np.ndarray:
    """L1-normalised histogram of writes per register address.
    Dropping reg < 0 / reg > MAX_REG defends against sentinel rows
    leaking into the dump (raw .dump.parquet shouldn't carry them).
    """
    valid = (regs >= 0) & (regs <= MAX_REG)
    counts = np.bincount(regs[valid], minlength=REG_DENSITY_DIM).astype(np.float64)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def _delta_histogram(clocks: np.ndarray) -> np.ndarray:
    """L1-normalised log-bucket histogram of successive clock deltas."""
    if clocks.size < 2:
        return np.zeros(DELTA_BUCKETS, dtype=np.float64)
    deltas = np.diff(clocks)
    deltas[deltas < 0] = 0
    buckets = np.digitize(deltas, DELTA_EDGES)
    counts = np.bincount(buckets, minlength=DELTA_BUCKETS).astype(np.float64)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def _ctrl_ngrams(regs: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CTRL 2-gram (64 buckets) and 3-gram (feature-hashed into 32
    buckets) over the per-voice CTRL streams. Each voice's CTRL
    write sequence is collapsed via ``_ctrl_state`` and n-grammed
    independently; the counts are summed across voices before
    L1 normalisation so a voice that's silent on this dump doesn't
    """
    bigram = np.zeros(CTRL_2GRAM_DIM, dtype=np.float64)
    trigram = np.zeros(CTRL_3GRAM_DIM, dtype=np.float64)
    for voice_ctrl_reg in CTRL_REGS:
        mask = regs == voice_ctrl_reg
        if not mask.any():
            continue
        v_vals = vals[mask]
        states = np.array([_ctrl_state(int(v)) for v in v_vals], dtype=np.int64)
        if states.size >= 2:
            pairs = states[:-1] * 8 + states[1:]
            counts = np.bincount(pairs, minlength=CTRL_2GRAM_DIM)
            bigram += counts.astype(np.float64)
        if states.size >= 3:
            triplets = states[:-2] * 64 + states[1:-1] * 8 + states[2:]
            for t in triplets:
                h = hashlib.blake2b(int(t).to_bytes(2, "little"), digest_size=4)
                bucket = int.from_bytes(h.digest(), "little") % CTRL_3GRAM_DIM
                trigram[bucket] += 1.0
    bg_sum = bigram.sum()
    if bg_sum > 0:
        bigram /= bg_sum
    tg_sum = trigram.sum()
    if tg_sum > 0:
        trigram /= tg_sum
    return bigram, trigram


def _filter_touch_ratio(regs: np.ndarray) -> float:
    """Fraction of writes targeting any filter register (21/22/23).
    Engines without a filter routine emit ~0; engines that animate
    the filter (Galway sweeps, Hubbard timbral) emit measurably
    higher values.
    """
    if regs.size == 0:
        return 0.0
    valid = (regs >= 0) & (regs <= MAX_REG)
    if not valid.any():
        return 0.0
    valid_regs = regs[valid]
    filter_mask = np.isin(valid_regs, list(FILTER_REGS_3))
    return float(filter_mask.sum() / valid.sum())


def compute_fingerprint(
    parquet_path: Path,
    n_writes: int = DEFAULT_FINGERPRINT_WRITES,
) -> np.ndarray | None:
    """Compute the engine fingerprint vector for one dump. Returns
    a length-``FEATURE_DIM`` float64 array, or None if the parquet
    couldn't be read / had fewer than 2 writes.
    """
    writes = _read_writes(parquet_path, n_writes)
    if writes is None or writes.shape[0] < 2:
        return None
    clocks = writes[:, 0]
    regs = writes[:, 2]
    vals = writes[:, 3]
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    vec[SLICE_REG_DENSITY] = _reg_density(regs)
    vec[SLICE_DELTA] = _delta_histogram(clocks)
    bigram, trigram = _ctrl_ngrams(regs, vals)
    vec[SLICE_CTRL_2GRAM] = bigram
    vec[SLICE_CTRL_3GRAM] = trigram
    vec[IDX_FILTER_TOUCH] = _filter_touch_ratio(regs)
    return vec


ENGINE_FP_K = 7
UNKNOWN_CLUSTER = 0

_ENGINE_FAMILIES_CANDIDATES = (
    Path("/integration_tests/data/prodlike/engine_families.json"),
    Path(__file__).resolve().parents[2]
    / "integration_tests"
    / "data"
    / "prodlike"
    / "engine_families.json",
)


@functools.lru_cache(maxsize=1)
def _load_composer_to_cluster() -> dict[str, int]:
    """Composer name -> cluster id (1..K) at ``ENGINE_FP_K=7``. Empty
    dict if ``engine_families.json`` isn't reachable (e.g. running
    outside any expected layout) -- callers fall back to ``UNKNOWN_CLUSTER``.
    """
    for path in _ENGINE_FAMILIES_CANDIDATES:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logging.warning("%s: engine_families read failed: %s", path, e)
                return {}
            try:
                stats = data["composer_stats"]
                labels = data["cluster_assignments"][str(ENGINE_FP_K)]
            except KeyError as e:
                logging.warning("%s: engine_families missing key %s", path, e)
                return {}
            if len(stats) != len(labels):
                logging.warning(
                    "%s: composer_stats/labels mismatch (%d vs %d)",
                    path,
                    len(stats),
                    len(labels),
                )
                return {}
            return {s["name"]: int(c) for s, c in zip(stats, labels)}
    return {}


def cluster_for_composer(name: str | None) -> int:
    """Map composer name -> cluster id (1..K), or ``UNKNOWN_CLUSTER``
    if the composer isn't in the audit's top-N list. Composer names
    are case-sensitive and match the audit's ``composer_stats`` ``name``
    field (HVSC ``MUSICIANS/<L>/<composer>`` directory names with
    spaces underscored).
    """
    if not name:
        return UNKNOWN_CLUSTER
    return _load_composer_to_cluster().get(name, UNKNOWN_CLUSTER)


def composer_from_dump_path(path: Path | str) -> str | None:
    """Heuristic composer name extraction for typical HVSC + training-
    dump layouts: ``.../<composer>/<song>.dump.parquet``. Returns the
    parent directory's basename; ``None`` if the path has no parent.
    Test fixtures and ad-hoc paths whose parent dir isn't a real
    composer name resolve to ``UNKNOWN_CLUSTER`` via the lookup miss
    """
    parent = Path(path).resolve().parent
    name = parent.name
    return name or None


def cluster_for_path(path: Path | str) -> int:
    """Convenience: composer-name extract + cluster lookup in one call."""
    return cluster_for_composer(composer_from_dump_path(path))


_CANONICAL_PALETTES_CANDIDATES = (
    Path("/integration_tests/data/mini/engine_fp_palettes.json"),
    Path("/integration_tests/data/canonical/engine_fp_palettes.json"),
    Path("/integration_tests/data/prodlike/engine_fp_palettes.json"),
    Path(__file__).resolve().parents[2]
    / "integration_tests"
    / "data"
    / "mini"
    / "engine_fp_palettes.json",
    Path(__file__).resolve().parents[2]
    / "integration_tests"
    / "data"
    / "canonical"
    / "engine_fp_palettes.json",
    Path(__file__).resolve().parents[2]
    / "integration_tests"
    / "data"
    / "prodlike"
    / "engine_fp_palettes.json",
)


def _resolve_palettes_path(explicit: Path | str | None) -> Path | None:
    """Pick the canonical-palette JSON to read: explicit path wins, else
    first candidate that exists. Returns None if nothing reachable.
    """
    if explicit is not None:
        p = Path(explicit)
        return p if p.exists() else None
    for cand in _CANONICAL_PALETTES_CANDIDATES:
        if cand.exists():
            return cand
    return None


@functools.lru_cache(maxsize=4)
def _load_palettes_cached(path_str: str | None) -> dict[int, tuple]:
    """Inner cache keyed by stringified path (Path isn't hashable
    across lru_cache keys in older Pythons; stringify defensively).
    """
    path = _resolve_palettes_path(path_str)
    if path is None:
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning("%s: canonical palette read failed: %s", path, e)
        return {}
    try:
        clusters_raw = data["clusters"]
    except KeyError:
        logging.warning("%s: missing 'clusters' key", path)
        return {}
    out: dict[int, tuple] = {}
    for cid_str, progs in clusters_raw.items():
        try:
            cid = int(cid_str)
        except ValueError:
            logging.warning("%s: non-int cluster key %r", path, cid_str)
            continue
        out[cid] = tuple(
            tuple(tuple(int(x) for x in triple) for triple in prog) for prog in progs
        )
    return out


def load_canonical_cluster_palettes(
    path: Path | str | None = None,
) -> dict[int, tuple]:
    """Return ``{cluster_id: (program_tuple, ...)}`` from the canonical
    JSON artifact. Empty dict when the artifact isn't reachable.
    """
    return _load_palettes_cached(str(path) if path is not None else None)
