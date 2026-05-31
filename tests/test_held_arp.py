"""held_arp: the held-step wavetable-ARP reclassification of would-be-RESID notes. Unit-tests the RLE
cycle detector/inverse and the ``_orn_rows`` reclassification -- when held_arp is on, a note whose
floored offsets are a held cycle (a frame-period the skeleton's ARP misses, forced to RESID by an
off-pitch accent) emits a HELD_ARP ornament instead of a raw RESID dump, replaying the same content
floor (byte-identical), validated on real tunes in the dev harness."""

from preframr_tokens.macros.skeleton_pass import (
    LUT,
    SkeletonPass,
    held_cycle,
    held_cycle_offsets,
)
from preframr_tokens.stfconstants import (
    ORN_OP,
    ORN_SUBREG_TYPE,
    ORN_TYPE_HELD_ARP,
    ORN_TYPE_RESID,
)

_IRQ = 19656


def test_held_cycle_finds_period_and_inverts():
    target = [0, 0, 12, 12, 24, 24] * 3
    hc = held_cycle(target)
    assert hc is not None
    period, holds = hc
    assert period == (0, 12, 24)
    assert held_cycle_offsets(period, holds) == target


def test_held_cycle_rejects_aperiodic():
    assert held_cycle([0, 0, 12, 12, 24, 24, 7, 3, 9]) is None


def test_held_cycle_rejects_giant_hold():
    assert held_cycle([5] * 300 + [9] * 300) is None


def _orn_types(rows):
    return [
        r["val"] for r in rows if r["op"] == ORN_OP and r["subreg"] == ORN_SUBREG_TYPE
    ]


def _held_cycle_per_frame():
    """Per-frame settled freqs that floor to a held cycle [+0, +24, accent] each held 2 frames x3;
    the accent (freq<8 -> no semitone) forces the pitched fitter to RESID."""
    base = 48
    step = [
        int(LUT[base]),
        int(LUT[base]),
        int(LUT[base + 24]),
        int(LUT[base + 24]),
        4,
        4,
    ]
    return step * 3


def test_orn_rows_reclassifies_resid_to_held_arp_when_enabled():
    note, per = 48, _held_cycle_per_frame()
    SkeletonPass._held_arp = False
    off = _orn_types(
        SkeletonPass._orn_rows(0, note, per, ORN_TYPE_RESID, (), _IRQ, _IRQ)
    )
    assert off == [ORN_TYPE_RESID], ("held_arp OFF keeps RESID", off)
    SkeletonPass._held_arp = True
    try:
        on = _orn_types(
            SkeletonPass._orn_rows(0, note, per, ORN_TYPE_RESID, (), _IRQ, _IRQ)
        )
    finally:
        SkeletonPass._held_arp = False
    assert on == [ORN_TYPE_HELD_ARP], on
