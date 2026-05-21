"""Standalone helpers used by the reglog parser pipeline."""

import json
import os

from preframr_tokens.stfconstants import (
    DELAY_REG,
    DESCRIPTION_PDTYPE,
    FC_LO_REG,
    FRAME_REG,
    SUBREG_PDTYPE,
    VOICES,
    VOICE_REG_SIZE,
)


def vreg_match(vreg):
    """Set of absolute reg ids for one voice-relative register offset
    across all ``VOICES``. E.g. ``vreg_match(0)`` = freq-lo regs
    ``{0, 7, 14}`` for VOICES=3, VOICE_REG_SIZE=7.
    """
    return {(v * VOICE_REG_SIZE) + vreg for v in range(VOICES)}


def freq_match(df):
    return df["reg"].isin(vreg_match(0))


def pcm_match(df):
    return df["reg"].isin(vreg_match(2))


def ctrl_match(df):
    return df["reg"].isin(vreg_match(4))


def adsr_match(df):
    return df["reg"].isin(vreg_match(5) | vreg_match(6))


def ad_match(df):
    return df["reg"].isin(vreg_match(5))


def sr_match(df):
    return df["reg"].isin(vreg_match(6))


def filter_match(df):
    return df["reg"] == FC_LO_REG


def frame_match(df):
    return (df["reg"] == FRAME_REG) | (df["reg"] == DELAY_REG)


def tighten_persist_dtypes(df):
    """Cast ``subreg`` to Int16 and ``description`` to Int8 in-place if
    they're wider. The macro pipeline creates these columns at int64
    by default (numpy literal broadcast); the parsed-parquet write
    site casts them down before persistence. Saves a sizeable fraction
    of parsed-form in-memory footprint at parse + preload.
    """
    if "subreg" in df.columns and df["subreg"].dtype != SUBREG_PDTYPE:
        df["subreg"] = df["subreg"].astype(SUBREG_PDTYPE)
    if "description" in df.columns and df["description"].dtype != DESCRIPTION_PDTYPE:
        df["description"] = df["description"].astype(DESCRIPTION_PDTYPE)
    return df


def _palettes_sidecar_path(parquet_path):
    """Sidecar JSON path that carries the macro-pipeline palettes
    (``gate_palette``, ``instrument_palette``) for one parsed parquet.
    Decoupled from ``df.attrs`` because pandas / pyarrow can't
    serialise tuple-keyed attrs to parquet metadata.
    """
    return parquet_path + ".palettes.json"


def dump_palettes_attrs(attrs, parquet_path):
    """Write engine fingerprint / cluster attrs to ``parquet_path``'s
    sidecar JSON when present.
    """
    if not attrs:
        return
    out = {}
    ef = attrs.get("engine_fingerprint")
    if ef is not None:
        out["engine_fingerprint"] = list(ef)
    if "engine_fp_cluster" in attrs:
        out["engine_fp_cluster"] = int(attrs["engine_fp_cluster"])
    if not out:
        return
    with open(_palettes_sidecar_path(parquet_path), "w") as f:
        json.dump(out, f)


def load_palettes_attrs(parquet_path):
    """Inverse of :func:`dump_palettes_attrs`. Returns a dict suitable
    for assignment to ``df.attrs``; empty if the sidecar doesn't
    exist.
    """
    sidecar = _palettes_sidecar_path(parquet_path)
    if not os.path.exists(sidecar):
        return {}
    with open(sidecar) as f:
        raw = json.load(f)
    out = {}
    if "engine_fingerprint" in raw:
        out["engine_fingerprint"] = [float(x) for x in raw["engine_fingerprint"]]
    if "engine_fp_cluster" in raw:
        out["engine_fp_cluster"] = int(raw["engine_fp_cluster"])
    return out


def wrapbits(x, reglen):
    """Bit-rotate-left by 1 within a ``reglen``-bit window."""
    base = (x << 1) & (2**reglen - 1)
    lsb = (x >> (reglen - 1)) & 1
    return base ^ lsb
