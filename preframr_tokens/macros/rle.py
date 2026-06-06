"""Run-length encoding used by the skeleton held-ARP cycle detector (``skeleton_pass``): the
``(value, run-length)`` collapse and its inverse live in one tested place rather than being
re-implemented per caller."""

__all__ = ["run_length_encode", "run_length_decode"]


def run_length_encode(seq):
    """Collapse ``seq`` into ``[(value, run_length)]`` of maximal consecutive-equal runs."""
    out = []
    for x in seq:
        if out and out[-1][0] == x:
            out[-1] = (x, out[-1][1] + 1)
        else:
            out.append((x, 1))
    return out


def run_length_decode(pairs):
    """Inverse of ``run_length_encode``: expand ``[(value, run_length)]`` back to a flat list."""
    out = []
    for value, run in pairs:
        out.extend([value] * int(run))
    return out
