"""Engine fingerprint vector + caller-provided cluster lookup."""

from __future__ import annotations

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

__all__ = [
    "compute_fingerprint",
    "composer_from_dump_path",
    "ClusterTable",
    "UNKNOWN_CLUSTER",
    "ENGINE_FP_K",
    "FEATURE_DIM",
    "DEFAULT_FINGERPRINT_WRITES",
]

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

ENGINE_FP_K = 7
UNKNOWN_CLUSTER = 0


def _ctrl_state(val: int) -> int:
    """Collapse a CTRL byte into an 8-state code: waveform-bit-index times 2 plus the gate bit."""
    waveform_nibble = (val >> 4) & 0xF
    if waveform_nibble == 0:
        wave_idx = 0
    else:
        wave_idx = (waveform_nibble & -waveform_nibble).bit_length() - 1
        wave_idx = min(wave_idx, 3)
    gate = val & 0x1
    return (wave_idx << 1) | gate


def _read_writes(parquet_path: Path, n_writes: int) -> np.ndarray | None:
    """Return the first n_writes rows as (N, 4) int64 array."""
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
    """L1-normalised histogram of writes per register address."""
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
    """CTRL 2-gram (64 buckets) + 3-gram (feature-hashed into 32 buckets) per voice, summed across voices, L1-normalised."""
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
    """Fraction of writes targeting any filter register (21/22/23)."""
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
    """Engine fingerprint vector for one dump. Returns length-FEATURE_DIM float64 array, or None on read failure / <2 writes."""
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


def composer_from_dump_path(path: Path | str) -> str | None:
    """Heuristic composer-name extraction for HVSC / training-dump layouts."""
    parent = Path(path).resolve().parent
    name = parent.name
    return name or None


class ClusterTable:
    """Composer-name -> engine-cluster-id (1..K) lookup. Caller provides the data file; library carries no opinions about which clustering snapshot to use."""

    def __init__(
        self,
        families_json: Path | str | None = None,
        k: int = ENGINE_FP_K,
    ):
        self.k = k
        self._table: dict[str, int] = {}
        if families_json is None:
            return
        path = Path(families_json)
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logging.warning("%s: engine_families read failed: %s", path, e)
            return
        try:
            stats = data["composer_stats"]
            labels = data["cluster_assignments"][str(k)]
        except KeyError as e:
            logging.warning("%s: engine_families missing key %s", path, e)
            return
        if len(stats) != len(labels):
            logging.warning(
                "%s: composer_stats/labels mismatch (%d vs %d)",
                path,
                len(stats),
                len(labels),
            )
            return
        self._table = {s["name"]: int(c) for s, c in zip(stats, labels)}

    def cluster_for_composer(self, name: str | None) -> int:
        if not name:
            return UNKNOWN_CLUSTER
        return self._table.get(name, UNKNOWN_CLUSTER)

    def cluster_for_path(self, path: Path | str) -> int:
        return self.cluster_for_composer(composer_from_dump_path(path))

    def __len__(self) -> int:
        return len(self._table)

    def __bool__(self) -> bool:
        return bool(self._table)
