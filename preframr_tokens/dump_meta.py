"""DumpMeta sidecar: per-dump metadata cache with a code-hash staleness gate."""

from __future__ import annotations

import hashlib
import inspect
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from preframr_tokens.stfconstants import (
    DEFAULT_IRQ_CYCLES,
    DUMP_SUFFIX,
    MODE_VOL_REG,
    VOICES,
    VOICE_REG_SIZE,
)

__all__ = [
    "DumpMeta",
    "meta_path_for",
    "meta_code_hash",
    "write_meta",
    "read_meta",
    "filter_dump_paths",
    "META_SUFFIX",
]

LOGGER = logging.getLogger(__name__)

META_SUFFIX = ".meta.parquet"

DIGI_DENSITY_THRESHOLD = 12.0

_FIELDS = (
    "meta_code_hash",
    "is_digi",
    "irq",
    "n_frames",
    "vol_changes_per_frame_max",
    "ctrl_changes_per_frame_max",
    "freq_writes_per_frame_max",
    "pw_changes_per_frame_max",
    "reg_write_density_max",
)


def meta_path_for(dump_path: str | Path) -> Path:
    return Path(str(dump_path).replace(DUMP_SUFFIX, META_SUFFIX))


def _span_frames(raw_df: pd.DataFrame, n_frames: int) -> float:
    """Player-frame span over the dump's wall-clock window (the digi density
    denominator): the settled per-frame view erases a digi (a sample is rewritten
    dozens of times per frame, last write wins), so density is measured against the
    raw clock span, not the irq-unique frame count. Falls back to ``n_frames`` when
    no usable clock column is present (synthetic single-clock-per-frame inputs)."""
    if "clock" not in raw_df.columns:
        return float(max(1, n_frames))
    clk = raw_df["clock"].to_numpy(np.int64)
    if clk.size == 0:
        return float(max(1, n_frames))
    span_cycles = float(clk.max() - clk.min())
    if span_cycles < DEFAULT_IRQ_CYCLES:
        return float(max(1, n_frames))
    return max(1.0, span_cycles / DEFAULT_IRQ_CYCLES)


def _build_meta_from_raw(dump_path: Path, raw_df: pd.DataFrame) -> dict[str, Any]:
    """Compute every DumpMeta field from a raw register-write dump. Treat this body as the canonical generator; its source-hash drives ``stale``."""
    regs = raw_df["reg"].to_numpy()
    irqs = raw_df["irq"].to_numpy()
    irq_unique = sorted(set(int(x) for x in irqs if x))
    irq_value = int(irq_unique[0]) if irq_unique else 0
    n_frames = int(len(set(irqs)))
    vol_mask = regs == MODE_VOL_REG
    vol_counts = (
        pd.Series(irqs[vol_mask]).value_counts()
        if vol_mask.any()
        else pd.Series(dtype=int)
    )
    vol_max = int(vol_counts.max()) if len(vol_counts) else 0
    voice_ctrl_regs = [v * VOICE_REG_SIZE + 4 for v in range(VOICES)]
    voice_freq_regs = [v * VOICE_REG_SIZE for v in range(VOICES)] + [
        v * VOICE_REG_SIZE + 1 for v in range(VOICES)
    ]
    ctrl_mask = np.isin(regs, voice_ctrl_regs)
    freq_mask = np.isin(regs, voice_freq_regs)
    ctrl_counts = (
        pd.Series(irqs[ctrl_mask]).value_counts()
        if ctrl_mask.any()
        else pd.Series(dtype=int)
    )
    freq_counts = (
        pd.Series(irqs[freq_mask]).value_counts()
        if freq_mask.any()
        else pd.Series(dtype=int)
    )
    ctrl_max = int(ctrl_counts.max()) if len(ctrl_counts) else 0
    freq_max = int(freq_counts.max()) if len(freq_counts) else 0
    voice_pw_regs = [v * VOICE_REG_SIZE + 2 for v in range(VOICES)] + [
        v * VOICE_REG_SIZE + 3 for v in range(VOICES)
    ]
    pw_mask = np.isin(regs, voice_pw_regs)
    pw_counts = (
        pd.Series(irqs[pw_mask]).value_counts()
        if pw_mask.any()
        else pd.Series(dtype=int)
    )
    pw_max = int(pw_counts.max()) if len(pw_counts) else 0
    reg_write_density_max = 0.0
    span_frames = _span_frames(raw_df, n_frames)
    if span_frames and len(regs):
        reg_counts = pd.Series(regs).value_counts()
        reg_write_density_max = float(reg_counts.max()) / span_frames
    is_digi = bool(
        vol_max >= 40
        or ctrl_max >= 20
        or pw_max >= 40
        or reg_write_density_max > DIGI_DENSITY_THRESHOLD
    )
    return {
        "meta_code_hash": meta_code_hash(),
        "is_digi": is_digi,
        "irq": irq_value,
        "n_frames": n_frames,
        "vol_changes_per_frame_max": vol_max,
        "ctrl_changes_per_frame_max": ctrl_max,
        "freq_writes_per_frame_max": freq_max,
        "pw_changes_per_frame_max": pw_max,
        "reg_write_density_max": round(reg_write_density_max, 3),
    }


def meta_code_hash() -> str:
    src = inspect.getsource(_build_meta_from_raw)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def raw_is_digi(raw_df: pd.DataFrame) -> bool:
    """The ``is_digi`` classification computed directly from a raw dump DataFrame (no sidecar)."""
    return bool(_build_meta_from_raw(Path(""), raw_df)["is_digi"])


def write_meta(dump_path: str | Path, raw_df: pd.DataFrame) -> Path:
    meta_path = meta_path_for(dump_path)
    fields = _build_meta_from_raw(Path(dump_path), raw_df)
    pd.DataFrame([fields]).to_parquet(meta_path, index=False)
    return meta_path


def read_meta(dump_path: str | Path) -> Optional["DumpMeta"]:
    mp = meta_path_for(dump_path)
    if not mp.exists():
        return None
    try:
        row = pd.read_parquet(mp).iloc[0].to_dict()
    except Exception as exc:
        LOGGER.warning("read_meta failed for %s: %s", mp, exc)
        return None
    return DumpMeta(dump_path=str(dump_path), fields=row)


class DumpMeta:
    """Read-only view of one dump's cached meta fields."""

    def __init__(self, dump_path: str, fields: dict[str, Any]):
        self.dump_path = dump_path
        self.fields = dict(fields)

    @property
    def stale(self) -> bool:
        return str(self.fields.get("meta_code_hash", "")) != meta_code_hash()

    @property
    def is_digi(self) -> bool:
        return bool(self.fields.get("is_digi", False))

    @property
    def irq(self) -> int:
        return int(self.fields.get("irq", 0))

    @property
    def n_frames(self) -> int:
        return int(self.fields.get("n_frames", 0))

    @property
    def vol_changes_per_frame_max(self) -> int:
        return int(self.fields.get("vol_changes_per_frame_max", 0))

    @property
    def ctrl_changes_per_frame_max(self) -> int:
        return int(self.fields.get("ctrl_changes_per_frame_max", 0))

    @property
    def freq_writes_per_frame_max(self) -> int:
        return int(self.fields.get("freq_writes_per_frame_max", 0))

    @property
    def pw_changes_per_frame_max(self) -> int:
        return int(self.fields.get("pw_changes_per_frame_max", 0))

    @property
    def reg_write_density_max(self) -> float:
        return float(self.fields.get("reg_write_density_max", 0.0))


def filter_dump_paths(
    dump_paths: Iterable[str],
    exclude_digi: bool = False,
    irq_range: Optional[tuple[int, int]] = None,
    require_meta: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Filter a list of dump paths using their DumpMeta sidecars. Paths with missing or stale metas are KEPT by default (so the caller can fall back to re-parsing); set ``require_meta=True`` to drop them. Returns (admitted_paths, dropped_reasons)."""
    admitted: list[str] = []
    dropped: dict[str, str] = {}
    cur_hash = meta_code_hash()
    for path in dump_paths:
        meta = read_meta(path)
        if meta is None:
            if require_meta:
                dropped[path] = "no_meta"
                continue
            admitted.append(path)
            continue
        if str(meta.fields.get("meta_code_hash", "")) != cur_hash:
            if require_meta:
                dropped[path] = "stale_meta"
                continue
            admitted.append(path)
            continue
        if exclude_digi and meta.is_digi:
            dropped[path] = "digi"
            continue
        if irq_range is not None:
            lo, hi = irq_range
            if not (lo <= meta.irq <= hi):
                dropped[path] = f"irq_out_of_range_{meta.irq}"
                continue
        admitted.append(path)
    return admitted, dropped
