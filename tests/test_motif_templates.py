"""Tests for MotifDict v2 value-slotted templates: lossless round-trip across
value-shifts, template-id consistency, and JSON version dispatch. Atoms are
``(op, reg, subreg, val, diff)``."""

from preframr_tokens.macros.motif_pass import MotifDict
from preframr_tokens.stfconstants import MOTIF_ARG, MOTIF_OP

_TEMPLATE = {
    "id": 0,
    "shape": [[0, 2, -1, 32], [0, 5, 0, 32]],
    "consts": {0: 33},
    "slots": [1],
}
A = (0, 2, -1, 33, 32)
OTHER = (0, 6, -1, 0, 40)


def _dict():
    return MotifDict([], {}, templates=[_TEMPLATE])


def _stream():
    return [A, (0, 5, 0, 15, 32), A, (0, 5, 0, 99, 32), OTHER]


def test_v2_roundtrip_lossless():
    md = _dict()
    stream = _stream()
    assert md.expand(md.encode(stream)) == stream


def test_v2_collapses_value_variants_to_one_template():
    enc = _dict().encode(_stream())
    motifs = [a[3] for a in enc if a[0] == MOTIF_OP]
    args = [a[3] for a in enc if a[0] == MOTIF_ARG]
    assert motifs == [0, 0]
    assert args == [15, 99]
    assert enc[-1] == OTHER


def test_v2_json_version_dispatch_roundtrip():
    s = _dict().to_json()
    assert '"version": 2' in s
    md = MotifDict.from_json(s)
    assert len(md) == 1
    stream = _stream()
    assert md.expand(md.encode(stream)) == stream


def test_v1_still_loads_via_from_json():
    md = MotifDict.from_json('{"merges": [], "expansions": {}}')
    assert md.templates is None
    assert len(md) == 0
