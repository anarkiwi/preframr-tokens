"""Tests for the motif pass: lossless round-trip, boundary and cross-composer
guards, determinism, JSON serialization, and unknown-atom passthrough. Atoms are
``(op, reg, subreg, val, diff)`` so timing round-trips exactly."""

from types import SimpleNamespace

from preframr_tokens.macros.motif_pass import MotifDict, get_motif_dict, mine_motifs
from preframr_tokens.stfconstants import FRAME_REG

FRAME = (0, FRAME_REG, -1, 57, 19656)
A = (0, 2, -1, 33, 32)
B = (0, -126, -1, 0, 32)
C = (0, 5, 0, 15, 32)
DD = (0, 6, -1, 0, 32)


def _corpus():
    idiom = [FRAME, A, B]
    return (
        [idiom * 5 + [C, DD] * 5, idiom * 5, idiom * 5],
        ["c1", "c2", "c3"],
    )


def _mine():
    streams, comps = _corpus()
    return streams, mine_motifs(streams, comps, k=32, min_count=3, min_composers=2)


def test_roundtrip_lossless():
    streams, d = _mine()
    assert len(d) >= 1
    for s in streams:
        assert d.expand(d.encode(s)) == s


def test_compresses():
    streams, d = _mine()
    assert len(d.encode(streams[1])) < len(streams[1])


def test_boundary_guard_no_motif_ends_on_frame_advance():
    _, d = _mine()
    for seq in d.expansions.values():
        assert seq[-1][1] != FRAME_REG


def test_cross_composer_floor_excludes_single_composer_pair():
    streams, d = _mine()
    assert [C, DD] not in d.expansions.values()
    enc = d.encode(streams[0])
    assert enc.count(C) == 5 and enc.count(DD) == 5


def test_determinism():
    streams, comps = _corpus()
    d1 = mine_motifs(streams, comps, k=32, min_count=3, min_composers=2)
    d2 = mine_motifs(streams, comps, k=32, min_count=3, min_composers=2)
    assert d1.merges == d2.merges and d1.expansions == d2.expansions


def test_json_roundtrip():
    streams, d = _mine()
    d2 = MotifDict.from_json(d.to_json())
    for s in streams:
        assert d2.expand(d2.encode(s)) == s


def test_unknown_atoms_passthrough():
    _, d = _mine()
    novel = (0, 14, -1, 99, 32)
    s = [FRAME, A, B, novel, FRAME, A, B]
    assert d.expand(d.encode(s)) == s


def test_get_motif_dict_object_and_path(tmp_path):
    _, d = _mine()
    assert get_motif_dict(SimpleNamespace(motif_dict=d)) is d
    assert get_motif_dict(SimpleNamespace(motif_dict="")) is None
    assert get_motif_dict(SimpleNamespace()) is None
    path = tmp_path / "motif_dict.json"
    path.write_text(d.to_json())
    args = SimpleNamespace(motif_dict=str(path))
    loaded = get_motif_dict(args)
    assert loaded.merges == d.merges and loaded.expansions == d.expansions
    assert get_motif_dict(args) is loaded  # cached on args, not re-read
