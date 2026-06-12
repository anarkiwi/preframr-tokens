"""Complete, escape-free integer codec shared by every numeric field: a
signed int is zig-zagged then split into BIG-ENDIAN base-16 digits (most-significant first -- the coarse,
context-predictable part is committed before the noisy fine digit, which autoregressive models prefer),
each a token 0..15 with a high continue bit (token 0..31, ``& 16`` => more follow). A common small value
is one token; a rare large value is more digits of the same alphabet, never a different path.
"""

from __future__ import annotations

CONT = 0x10
DIGIT_MASK = 0x0F


def zigzag(n: int) -> int:
    """Map a signed int to a non-negative int (0,-1,1,-2,2,... -> 0,1,2,3,4,...)."""
    n = int(n)
    return n * 2 if n >= 0 else -n * 2 - 1


def unzigzag(z: int) -> int:
    """Inverse of :func:`zigzag`."""
    z = int(z)
    return (z >> 1) ^ -(z & 1)


def encode_unsigned(u: int) -> list[int]:
    """A non-negative int as big-endian base-16 digit tokens (high bit = continue). At least one token."""
    u = int(u)
    if u < 0:
        raise ValueError(f"encode_unsigned got negative {u}")
    digits: list[int] = []
    while True:
        digits.append(u & DIGIT_MASK)
        u >>= 4
        if not u:
            break
    digits.reverse()
    return [d | CONT for d in digits[:-1]] + [digits[-1]]


def encode_signed(n: int) -> list[int]:
    """A signed int as digit tokens (zig-zag then unsigned)."""
    return encode_unsigned(zigzag(n))


def decode_unsigned(tokens: list[int], pos: int = 0) -> tuple[int, int]:
    """Decode one unsigned varint from ``tokens`` at ``pos``; returns ``(value, next_pos)``. Raises
    ``ValueError`` on a truncated run (continue bit set but no following token), never silently tolerated.
    """
    u = 0
    n = len(tokens)
    while True:
        if pos >= n:
            raise ValueError("truncated varint (continue bit set at end of stream)")
        tok = tokens[pos]
        pos += 1
        u = (u << 4) | (tok & DIGIT_MASK)
        if not (tok & CONT):
            return u, pos
