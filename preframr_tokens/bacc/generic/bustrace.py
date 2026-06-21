"""Parse a preframr-sidtrace ``.bus.bin`` CPU bus trace into the ``(cyc, addr,
val, rw)`` record array the generic recovery reads.

preframr-sidtrace (``build/sidtrace``) runs a ``.sid`` through a lightly-patched,
cycle-accurate libsidplayfp 6510/SID emulation and emits the full CPU bus trace
as **packed little-endian** 12-byte records -- exactly the :data:`BUS_DT` layout
below, with NO magic header (each field is ``fwrite`` individually by the
recorder)::

    cyc   int64  8   absolute PHI1 CPU cycle of the access
    addr  uint16 2   bus address
    val   uint8  1   byte on the bus
    rw    uint8  1   0 = read, 1 = write

The SID-write substream is the subset with ``rw == 1`` and
``0xD400 <= addr <= 0xD418`` (:func:`busstate.sid_writes`).  This is the trusted,
deterministic provenance substrate for driver-agnostic BACC recovery; the traces
are large (tens of MB) and are NEVER committed.

The same on-disk layout is also emitted by the legacy raw ``BUS_DT`` dumper, so a
raw array loads with the same reader.  An RBT1-framed trace (revice
``vsiddump.py --bustrace``, magic ``RBT1``, 10-byte delta-cycle records) is a
DIFFERENT format and is intentionally not read here -- preframr-sidtrace is the
single native source for this module.
"""

import numpy as np

BUS_DT = np.dtype([("cyc", "<i8"), ("addr", "<u2"), ("val", "u1"), ("rw", "u1")])


def load_bus(path):
    """Load a preframr-sidtrace ``.bus.bin`` (native packed :data:`BUS_DT`).

    Returns the ``BUS_DT`` record array with absolute cycles.  Raises
    :class:`ValueError` for an RBT1-framed trace (the wrong, non-native source)
    or a file whose size is not a whole number of 12-byte records.
    """
    with open(path, "rb") as handle:
        head = handle.read(4)
    if head == b"RBT1":
        raise ValueError(
            f"{path} is an RBT1-framed trace; this module reads only native "
            "preframr-sidtrace .bus.bin records"
        )
    records = np.fromfile(path, dtype=BUS_DT)
    return records
