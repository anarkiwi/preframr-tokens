"""melody-merge-split: the is_melody_pitch_atom classifier + the pure
split_cross_boundary_merges function (cross-boundary merges expand, pure-melody and
pure-non-melody merges are kept, single base atoms are kept)."""

import numpy as np

from preframr_tokens.regtokenizer import (
    is_melody_pitch_atom,
    split_cross_boundary_merges,
)
from preframr_tokens.stfconstants import (
    FREQ_TRAJ_OP,
    FT_SUBREG_V0_HI,
    FT_SUBREG_V0_LO,
    ORN_OP,
    SKEL_OP,
)


def test_is_melody_pitch_atom():
    assert is_melody_pitch_atom(FREQ_TRAJ_OP, 0, FT_SUBREG_V0_HI)
    assert is_melody_pitch_atom(FREQ_TRAJ_OP, 7, FT_SUBREG_V0_LO)
    assert is_melody_pitch_atom(SKEL_OP, 14, -1)
    assert is_melody_pitch_atom(ORN_OP, 0, -1)
    assert not is_melody_pitch_atom(FREQ_TRAJ_OP, 0, 6)
    assert not is_melody_pitch_atom(FREQ_TRAJ_OP, 2, FT_SUBREG_V0_HI)
    assert not is_melody_pitch_atom(SKEL_OP, 2, -1)
    assert not is_melody_pitch_atom(0, 0, -1)


def test_split_cross_boundary_merges_keeps_singles_and_pure_merges():
    melody_ids = {0, 1}
    decode = {
        10: [5],
        11: [0, 1],
        12: [2, 3],
        13: [0, 2],
        14: [0, 1, 2],
    }
    uni_atom_id = {0: 100, 1: 101, 2: 102, 3: 103, 5: 105}
    seq = np.array([10, 11, 12, 13, 14, 5], dtype=np.int32)
    out = split_cross_boundary_merges(
        seq,
        decode_to_base_ids=lambda uid: decode.get(uid, [uid]),
        base_to_unigram_id=lambda b: uni_atom_id.get(b),
        is_melody=lambda b: b in melody_ids,
        n_atoms=10,
    )
    assert out.tolist() == [10, 11, 12, 100, 102, 100, 101, 102, 5]


def test_split_emits_unigram_atom_ids_not_base_ids():
    melody_ids = {7}
    decode = {99: [7, 8]}
    uni = {7: 700, 8: 800}
    seq = np.array([99], dtype=np.int32)
    out = split_cross_boundary_merges(
        seq,
        decode_to_base_ids=lambda uid: decode[uid],
        base_to_unigram_id=lambda b: uni.get(b),
        is_melody=lambda b: b in melody_ids,
        n_atoms=10,
    )
    assert out.tolist() == [700, 800]


def test_split_dtype_preserved():
    seq = np.array([5], dtype=np.uint32)
    out = split_cross_boundary_merges(
        seq,
        decode_to_base_ids=lambda uid: [uid],
        base_to_unigram_id=lambda b: b,
        is_melody=lambda b: False,
        n_atoms=10,
        dtype=seq.dtype,
    )
    assert out.dtype == np.uint32
