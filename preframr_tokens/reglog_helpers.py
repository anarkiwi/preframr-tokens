"""Helpers used by the reglog parser pipeline. Palette sidecar IO has moved to ``preframr_tokens.palette_io`` and ``wrapbits`` has moved to ``preframr_tokens.utils``; both are re-exported here for back-compat. New consumers should import from the source modules."""

from preframr_tokens.palette_io import (
    dump_palettes_attrs,
    load_palettes_attrs,
)
from preframr_tokens.stfconstants import (
    DEFAULT_IRQ_CYCLES,
    DELAY_REG,
    DESCRIPTION_PDTYPE,
    FC_LO_REG,
    FRAME_REG,
    SUBREG_PDTYPE,
    VOICES,
    VOICE_REG_SIZE,
)
from preframr_tokens.utils import wrapbits

__all__ = [
    "vreg_match",
    "freq_match",
    "pcm_match",
    "ctrl_match",
    "adsr_match",
    "ad_match",
    "sr_match",
    "filter_match",
    "frame_match",
    "read_initial_irq",
    "tighten_persist_dtypes",
    "dump_palettes_attrs",
    "load_palettes_attrs",
    "wrapbits",
]


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


def read_initial_irq(df, default: int = DEFAULT_IRQ_CYCLES) -> int:
    """Read the initial IRQ cycle interval from a parser-output df by taking the first FRAME_REG row's ``diff`` column. Returns ``default`` (canonical SID ~50.1 Hz raster) if no FRAME rows present."""
    frame_rows = df[df["reg"] == FRAME_REG]
    if frame_rows.empty:
        return int(default)
    return int(frame_rows["diff"].iloc[0])


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
