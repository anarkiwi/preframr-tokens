"""Voice-relative register classification: raw ``reg`` id → boolean row mask. The parse-domain sibling of ``preframr_tokens.macros.roles`` (which classifies macro ``(op, subreg)`` pairs); both are pure ``stfconstants`` decision layers so a classification matches the data by construction instead of consumers re-encoding reg ids inline."""

from preframr_tokens.stfconstants import (
    DELAY_REG,
    FC_LO_REG,
    FRAME_REG,
    VOICES,
    VOICE_REG_SIZE,
)

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
    "reg_class",
]

_REG_KIND_BY_OFFSET = {0: "FREQ", 2: "PW", 4: "CTRL", 5: "AD", 6: "SR"}


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


def reg_class(reg):
    """``(kind, voice)`` for an absolute ``reg`` id, or ``None`` if ``reg``
    is not a classified voice register. ``kind`` is one of ``"FREQ"``,
    ``"PW"``, ``"CTRL"``, ``"AD"``, ``"SR"``. Scalar counterpart of the
    boolean ``*_match`` predicates above (e.g. ``reg_class(r)[0] == "FREQ"``
    iff ``freq_match`` would select ``r``).
    """
    reg = int(reg)
    if reg < 0:
        return None
    voice, offset = divmod(reg, VOICE_REG_SIZE)
    if voice >= VOICES:
        return None
    kind = _REG_KIND_BY_OFFSET.get(offset)
    if kind is None:
        return None
    return (kind, voice)
