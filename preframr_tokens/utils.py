import logging


def get_logger(level=None):
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
    if level is not None:
        level = getattr(logging, level.upper())
        logger.setLevel(level)
    return logger


def wrapbits(x: int, reglen: int) -> int:
    """Bit-rotate-left by 1 within a ``reglen``-bit window."""
    base = (x << 1) & (2**reglen - 1)
    lsb = (x >> (reglen - 1)) & 1
    return base ^ lsb
