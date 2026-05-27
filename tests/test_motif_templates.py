"""Tests for MotifDict v2 value-slotted templates: lossless round-trip across
value-shifts, template-id consistency, and JSON version dispatch. Atoms are
``(op, reg, subreg, val, diff)``."""

from preframr_tokens.macros.motif_pass import MotifDict, mine_templates
from preframr_tokens.stfconstants import FRAME_REG, MOTIF_ARG, MOTIF_OP

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


_TARGET = ((0, 2, -1, 32), (0, 5, 0, 32))


def _mine_corpus():
    head = (0, 2, -1, 33, 32)
    streams = [[head, (0, 5, 0, v, 32)] * 3 for v in (10, 20, 30)]
    return streams, ["c1", "c2", "c3"]


def _target(md):
    return next((t for t in (md.templates or []) if t["shape"] == _TARGET), None)


def test_mine_templates_finds_slotted_shape():
    md = mine_templates(*_mine_corpus(), k=16, min_count=3, min_composers=3)
    tmpl = _target(md)
    assert tmpl is not None
    assert tmpl["slots"] == [1]
    assert tmpl["consts"] == {0: 33}


def test_mine_templates_roundtrip():
    md = mine_templates(*_mine_corpus(), k=16, min_count=3, min_composers=3)
    stream = [(0, 2, -1, 33, 32), (0, 5, 0, 77, 32), (0, 9, -1, 1, 5)]
    assert md.expand(md.encode(stream)) == stream


def test_mine_templates_composer_floor():
    md = mine_templates(*_mine_corpus(), k=16, min_count=3, min_composers=4)
    assert _target(md) is None


def test_mine_templates_skips_frame_advance_end():
    frame = (0, FRAME_REG, -1, 1, 100)
    streams = [[(0, 2, -1, 33, 32), frame] * 3 for _ in range(3)]
    md = mine_templates(streams, ["c1", "c2", "c3"], min_count=3, min_composers=3)
    bad = ((0, 2, -1, 32), (0, FRAME_REG, -1, 100))
    assert all(t["shape"] != bad for t in (md.templates or []))
