"""Tests for the shared run-length codec (``macros/rle``) used by the skeleton held-ARP cycle detector
and the WAVETABLE codebook factoriser."""

from preframr_tokens.macros.rle import run_length_decode, run_length_encode


def test_encode_collapses_consecutive_runs():
    assert run_length_encode([3, 3, 1, 1, 1, 2]) == [(3, 2), (1, 3), (2, 1)]
    assert run_length_encode([]) == []
    assert run_length_encode([5]) == [(5, 1)]
    assert run_length_encode([0, 0, 12, 12, 24, 24]) == [(0, 2), (12, 2), (24, 2)]


def test_decode_inverts_encode():
    for seq in ([], [5], [3, 3, 1, 1, 1, 2], [0, 12, 0, 12], [7, 7, 7, -1, -1]):
        assert run_length_decode(run_length_encode(seq)) == seq


def test_no_merge_across_unequal_neighbours():
    assert run_length_encode([1, 2, 1, 2]) == [(1, 1), (2, 1), (1, 1), (2, 1)]
