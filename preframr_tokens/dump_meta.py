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
    DUMP_SUFFIX,
    MODE_VOL_REG,
    VOICES,
    VOICE_REG_SIZE,
)

LOGGER = logging.getLogger(__name__)

META_SUFFIX = ".meta.parquet"

_FIELDS = (
    "meta_code_hash",
    "is_digi",
    "irq",
    "n_frames",
    "vol_changes_per_frame_max",
    "ctrl_changes_per_frame_max",
    "freq_writes_per_frame_max",
)


def meta_path_for(dump_path: str | Path) -> Path:
    return Path(str(dump_path).replace(DUMP_SUFFIX, META_SUFFIX))


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
    is_digi = bool(vol_max >= 40 or ctrl_max >= 20)
    return {
        "meta_code_hash": meta_code_hash(),
        "is_digi": is_digi,
        "irq": irq_value,
        "n_frames": n_frames,
        "vol_changes_per_frame_max": vol_max,
        "ctrl_changes_per_frame_max": ctrl_max,
        "freq_writes_per_frame_max": freq_max,
    }


def meta_code_hash() -> str:
    src = inspect.getsource(_build_meta_from_raw)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


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
