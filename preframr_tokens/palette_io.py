"""JSON sidecar IO for engine-fingerprint / engine-fp-cluster ``df.attrs``. Decoupled from ``df.attrs`` because pandas / pyarrow can't serialise tuple-keyed attrs to parquet metadata; the sidecar is a ``<parquet>.palettes.json`` file written alongside each parsed parquet."""

from __future__ import annotations

import json
import os


def _palettes_sidecar_path(parquet_path: str) -> str:
    return parquet_path + ".palettes.json"


def dump_palettes_attrs(attrs, parquet_path: str) -> None:
    """Write the engine fingerprint / cluster portion of ``attrs`` to ``parquet_path``'s sidecar JSON if either key is present. No-op if ``attrs`` is empty or missing both keys."""
    if not attrs:
        return
    out: dict = {}
    ef = attrs.get("engine_fingerprint")
    if ef is not None:
        out["engine_fingerprint"] = list(ef)
    if "engine_fp_cluster" in attrs:
        out["engine_fp_cluster"] = int(attrs["engine_fp_cluster"])
    if not out:
        return
    with open(_palettes_sidecar_path(parquet_path), "w") as f:
        json.dump(out, f)


def load_palettes_attrs(parquet_path: str) -> dict:
    """Inverse of :func:`dump_palettes_attrs`. Returns a dict suitable for assignment to ``df.attrs``; empty if the sidecar doesn't exist."""
    sidecar = _palettes_sidecar_path(parquet_path)
    if not os.path.exists(sidecar):
        return {}
    with open(sidecar) as f:
        raw = json.load(f)
    out: dict = {}
    if "engine_fingerprint" in raw:
        out["engine_fingerprint"] = [float(x) for x in raw["engine_fingerprint"]]
    if "engine_fp_cluster" in raw:
        out["engine_fp_cluster"] = int(raw["engine_fp_cluster"])
    return out
