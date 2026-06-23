"""Self-contained unit tests for the PWLK-driven generic structure recovery.

These exercise the S1+S2+S4 mechanism (relocation resolve, the (zp),Y pointer-walk
pattern discovery, the four-dialect row grammar, the accumulator-fit) on SYNTHETIC
:class:`Distill` artifacts -- no ``preframr-sidtrace`` binary -- so the default CI
covers the generic recovery the env-gated whole-tune proofs validate on real tunes.

The invariants pinned here:
  * the (zp),Y walk's pointer-value stream IS the resolved orderlist->pattern bank;
    relocation maps a runtime pointer to its image (SNAP) address by the RELO delta;
  * the four row-grammar dialects collapse to ONE decode skeleton, selected by the
    byte-exact slice (a clean EOP termination within the song data);
  * the accumulator-fit reduces a ramp / quadratic STSQ cell to its generator and an
    un-fit cell to a verbatim store (the honest fallback); and
  * ``recover_structure`` selects the byte-exact candidate with the fewest tokens.
"""

import numpy as np

from preframr_tokens.bacc.generic import structure_recover as SR
from preframr_tokens.bacc.generic.distill import (
    ACC_READ_PLAY,
    Distill,
    PtrWalk,
    ReloCopy,
    StsqCell,
)

_LOAD = 0x1000
_LOAD_LEN = 0x1000  # image [0x1000, 0x2000)


def _blank_distill(nframes=64):
    acc = np.zeros(65536, dtype=np.uint8)
    ram = np.zeros(65536, dtype=np.uint8)
    return Distill(
        version=1,
        init_addr=_LOAD,
        play_addr=_LOAD + 3,
        load_addr=_LOAD,
        subtune=1,
        nframes=nframes,
        cycles_per_frame=19656,
        t0_cycle=0,
        load_len=_LOAD_LEN,
        acc=acc,
        ram=ram,
    )


def _newplayer_pattern(note_base):
    """A small NewPlayer pattern: instr=$A1, dur=$82, two notes, EOP $7F."""
    return [0xA1, 0x82, note_base, note_base + 2, 0x7F]


def _place(d, addr, bytez, read=True):
    for i, b in enumerate(bytez):
        d.ram[addr + i] = b
        if read:
            d.acc[addr + i] |= ACC_READ_PLAY


def test_reloc_delta_candidates_includes_relo_copies():
    d = _blank_distill()
    assert SR.reloc_delta_candidates(d) == [0]
    d.relo_copies = [
        ReloCopy(
            0,
            0,
            src_base=0x3FC4,
            dst_base=0xAACE,
            src_stride=2,
            dst_stride=1,
            idx_min=0,
            idx_max=254,
            count=255,
        )
    ]
    cands = SR.reloc_delta_candidates(d)
    assert 0 in cands and 0x6B0A in cands  # dst - src


def test_grammar_eops_and_pattern_len():
    dialects = SR._dialects()
    assert SR._grammar_eops(None) == {SR._END_OF_PATTERN}
    # NewPlayer EOPs are 0x7F and 0xFF.
    assert SR._grammar_eops(dialects["newplayer"]) == {0x7F, 0xFF}
    # TFX / FC EOP is 0xFF only.
    assert SR._grammar_eops(dialects["tfx"]) == {0xFF}
    ram = np.zeros(65536, dtype=np.uint8)
    pat = _newplayer_pattern(0x10)
    ram[0x1500 : 0x1500 + len(pat)] = pat

    class _S:
        grammar = dialects["newplayer"]

    assert SR._pattern_len(ram, 0x1500, _S()) == len(pat)


def test_discover_patterns_pwlk_in_place():
    d = _blank_distill()
    # three patterns at 0x1500/0x1520/0x1540, walked by a (zp),Y load.
    bases = [0x1500, 0x1520, 0x1540]
    for i, base in enumerate(bases):
        _place(d, base, _newplayer_pattern(0x10 + i))
    # the orderlist walks 0,1,2,1,0 (pattern 1 reused).
    seq = [bases[i] for i in (0, 1, 2, 1, 0)]
    d.ptr_walks = [
        PtrWalk(
            zp=0xFB,
            is_load=True,
            is_store=False,
            y_min=0,
            y_max=4,
            count=100,
            ptr_vals=seq,
            adv_frames=list(range(len(seq))),
        )
    ]
    pw = SR.discover_patterns_pwlk(d)
    assert pw is not None
    assert pw["pattern_src"] == bases
    assert pw["reloc_delta"] == 0
    assert pw["orderlist"] == [0, 1, 2, 1, 0]
    assert pw["grammar_name"] == "newplayer"


def test_discover_patterns_pwlk_relocated():
    d = _blank_distill()
    delta = 0x6B0A
    bases = [0x1500, 0x1520]  # image (SNAP) addresses
    for i, base in enumerate(bases):
        _place(d, base, _newplayer_pattern(0x10 + i))
    d.relo_copies = [
        ReloCopy(
            0,
            0,
            src_base=0x1500,
            dst_base=(0x1500 + delta) & 0xFFFF,
            src_stride=1,
            dst_stride=1,
            idx_min=0,
            idx_max=31,
            count=32,
        )
    ]
    # the player walks RUNTIME pointers (image + delta); recovery resolves via reloc.
    seq = [(b + delta) & 0xFFFF for b in (bases[0], bases[1], bases[0])]
    d.ptr_walks = [
        PtrWalk(
            zp=0xFC,
            is_load=True,
            is_store=False,
            y_min=0,
            y_max=4,
            count=100,
            ptr_vals=seq,
            adv_frames=[0, 1, 2],
        )
    ]
    pw = SR.discover_patterns_pwlk(d)
    assert pw is not None
    assert pw["reloc_delta"] == delta
    assert pw["pattern_src"] == bases  # read from the image, not the runtime base
    assert pw["orderlist"] == [0, 1, 0]


def test_fit_accumulator_ramp_quadratic_and_raw():
    ramp = np.array([100 + 5 * i for i in range(40)], dtype=np.int64) & 0xFFFF
    kind, seed, p1, _p2, _p3 = SR.fit_accumulator(ramp)
    assert (kind, seed, p1) == (SR.ACC_RAMP, 100, 5)
    quad = np.array([(i * i + 2 * i) & 0xFFFF for i in range(40)], dtype=np.int64)
    kind, _seed, _p1, p2, _p3 = SR.fit_accumulator(quad)
    assert kind == SR.ACC_QUADRATIC and p2 == 2
    noisy = np.array([0, 5, 1, 9, 2, 7, 3], dtype=np.int64)  # no closed-form fit
    assert SR.fit_accumulator(noisy)[0] == SR.ACC_RAW


def test_decode_patterns_grammar_matches_newplayer():
    d = _blank_distill()
    base = 0x1500
    pat = _newplayer_pattern(0x10)
    _place(d, base, pat)
    gram = SR._dialects()["newplayer"]
    patterns, instr_refs, notes, _cmds, durs, n_rows, n_bytes = (
        SR._decode_patterns_grammar(d, [base], gram)
    )
    assert n_rows == 2 and n_bytes == len(pat)
    assert notes == [0x10, 0x12]
    assert instr_refs == [1] and durs == [2]
    # rows carry the running (note, instr, dur, cmd) state.
    assert patterns[0][0] == (0x10, 1, 2, None)


def test_recover_structure_picks_fewest_tokens(monkeypatch):
    # both paths succeed; recover_structure must keep the lower-token candidate.
    d = _blank_distill()

    def fake_load(_p):
        return d

    monkeypatch.setattr(SR, "load_distill", fake_load)
    monkeypatch.setattr(SR, "read_sddf_slices", lambda _p: [])

    big = SR.RecoveredStructure(ok=True, ram=d.ram, nframes=64, n_patterns=9)
    small = SR.RecoveredStructure(ok=True, ram=d.ram, nframes=64, n_patterns=2)

    def fake_pwlk(_d, struct, _s):
        struct.__dict__.update(small.__dict__)
        return True

    def fake_legacy(_d, struct, _s):
        struct.__dict__.update(big.__dict__)
        return True

    monkeypatch.setattr(SR, "_recover_structure_pwlk", fake_pwlk)
    monkeypatch.setattr(SR, "_recover_structure_legacy", fake_legacy)
    monkeypatch.setattr(SR, "token_budget", lambda s: (s.n_patterns * 100, {}))

    got = SR.recover_structure("x")
    assert got.ok and got.n_patterns == 2  # the fewest-tokens candidate won


def test_recover_structure_no_structure_falls_back():
    # neither path yields a structure -> ok=False with a falsifiable reason.
    d = _blank_distill()
    import preframr_tokens.bacc.generic.structure_recover as M

    orig_load = M.load_distill
    try:
        M.load_distill = lambda _p: d
        M.read_sddf_slices = lambda _p: []
        got = M.recover_structure("x")
        assert not got.ok and "pure-code" in got.reason
    finally:
        M.load_distill = orig_load


def _synthetic_structured_distill(nframes=64):
    """A synthetic distill with a PWLK pattern walk, a freq accumulator STSQ cell,
    and a small program span -- enough to drive ``recover_structure`` end to end."""
    d = _blank_distill(nframes)
    bases = [0x1500, 0x1520, 0x1540]
    for i, base in enumerate(bases):
        _place(d, base, _newplayer_pattern(0x10 + i))
    seq = [bases[i] for i in (0, 1, 2, 1, 0)]
    d.ptr_walks = [
        PtrWalk(
            zp=0xFB,
            is_load=True,
            is_store=False,
            y_min=0,
            y_max=4,
            count=100,
            ptr_vals=seq,
            adv_frames=list(range(len(seq))),
        )
    ]
    # a shared program table (read-as-data, outside the patterns) -> a program span.
    _place(d, 0x1800, list(range(8)))
    # a ramp accumulator cell pair (lo at 0x40, hi at 0x41) the freq fit reduces;
    # the STSQ cells carry the per-frame BYTE values (low byte / high byte).
    ramp = np.array([(7 * i) & 0xFF for i in range(nframes)], dtype=np.uint8)
    zero = np.zeros(nframes, dtype=np.uint8)
    d.stsq_cells = [
        StsqCell(addr=0x0040, flags=0, first_seen=0, samples=ramp),
        StsqCell(addr=0x0041, flags=0, first_seen=0, samples=zero),
    ]
    return d, bases


def test_recover_structure_end_to_end_synthetic(monkeypatch):
    d, bases = _synthetic_structured_distill()
    monkeypatch.setattr(SR, "load_distill", lambda _p: d)
    monkeypatch.setattr(SR, "read_sddf_slices", lambda _p: [])
    struct = SR.recover_structure("x")
    assert struct.ok
    assert struct.pattern_src == bases
    assert struct.n_patterns == 3
    # the program span (the shared table) was carved out of the song-data mask.
    assert struct.program_spans  # at least the 0x1800 table
    # the token budget computes and is finite.
    total, brk = SR.token_budget(struct, frames=64)
    assert total > 0 and brk["n_patterns"] == 3
    # clean_pitches_residual / accumulator_generators consume the STSQ cells.
    gens = SR.accumulator_generators("x", np.zeros((64, 25), dtype=np.int64))
    assert gens is not None


# --- read-coverage (nibble / bit-packed dialect) pattern discovery ------------
def _nibble_pattern(seed):
    """A GoatTracker-style nibble/bit-packed pattern: bytes carry no value-range EOP
    marker (no 0x7F / 0xFF), so the value-range dialects cannot slice it -- only the
    observed READ-COVERAGE run sites it.  Deterministic on ``seed``."""
    return [(0x40 + ((seed + i * 7) & 0x3D)) for i in range(12 + (seed & 3))]


def test_discover_patterns_pwlk_read_coverage_nibble():
    """A nibble-grammar walk (no value-range EOP) is sited by read-coverage: the bank
    is the in-image read-as-data pointers, lengths are the observed read extents, and a
    phantom (non-data) pointer the walk visited is DROPPED, not fatal."""
    d = _blank_distill()
    bases = [0x1500, 0x1540, 0x1580]
    pats = [_nibble_pattern(i) for i in range(3)]
    for base, pat in zip(bases, pats):
        _place(d, base, pat)
    # the orderlist reuses pattern 1 and 0, and visits a PHANTOM pointer 0x1700 that
    # was never read as data (a null/init pointer) -- it must be dropped from the bank.
    seq = [bases[0], 0x1700, bases[1], bases[2], bases[1], bases[0]]
    d.ptr_walks = [
        PtrWalk(
            zp=0x20,
            is_load=True,
            is_store=False,
            y_min=0,
            y_max=11,
            count=200,
            ptr_vals=seq,
            adv_frames=list(range(len(seq))),
        )
    ]
    pw = SR.discover_patterns_pwlk(d)
    assert pw is not None
    assert pw["grammar_name"] == "readcov" and pw["grammar"] is None
    assert pw["pattern_src"] == bases  # the phantom 0x1700 is not in the bank
    assert pw["pattern_lens"] == [len(p) for p in pats]  # observed read extents
    # the orderlist indexes the bank; the phantom advance is 0xFF (no pattern).
    assert pw["orderlist"] == [0, 0xFF, 1, 2, 1, 0]


def test_read_coverage_rejects_streaming_cursor():
    """A per-frame streaming pointer (a sample / wavetable cursor: ``y == 0`` always,
    every advance a DISTINCT pointer) is NOT an orderlist->pattern walk and the read-
    coverage path must reject it, so it never fabricates a pseudo-bank from a cursor.

    Both structural tells are exercised: ``y_min == y_max`` (no within-pattern row
    indexing) AND the ~1:1 distinct/advance ratio (no pattern reuse)."""
    d = _blank_distill()
    # 40 distinct, never-reused pointers into a long read-as-data region (nibble bytes,
    # no value-range EOP so this is purely the read-coverage path's responsibility).
    _place(d, 0x1500, [(0x40 + (i % 0x3D)) for i in range(0x200)])
    seq = [0x1500 + 4 * i for i in range(40)]
    # y fixed at 0 -> the y-span guard alone rejects it.
    d.ptr_walks = [
        PtrWalk(
            zp=0x51,
            is_load=True,
            is_store=False,
            y_min=0,
            y_max=0,
            count=40,
            ptr_vals=seq,
            adv_frames=list(range(40)),
        )
    ]
    assert SR._pwlk_candidate_readcov(d, d.ptr_walks[0], 0, d.song_data_mask()) is None
    # even WITH row indexing, the ~1:1 distinct/advance ratio (no reuse) is rejected.
    d.ptr_walks[0] = PtrWalk(
        zp=0x51,
        is_load=True,
        is_store=False,
        y_min=0,
        y_max=8,
        count=40,
        ptr_vals=seq,
        adv_frames=list(range(40)),
    )
    assert SR._pwlk_candidate_readcov(d, d.ptr_walks[0], 0, d.song_data_mask()) is None


def test_read_coverage_structure_ir_roundtrips(monkeypatch):
    """A read-coverage (nibble) structure assembles into a StructureIR whose token
    serialization round-trips EQUAL (the codec invariant) and whose pattern bytes are
    exactly the observed read-coverage runs."""
    from preframr_tokens.bacc.generic import structure_ir as SI

    d = _blank_distill()
    bases = [0x1500, 0x1540, 0x1580]
    pats = [_nibble_pattern(i) for i in range(3)]
    for base, pat in zip(bases, pats):
        _place(d, base, pat)
    seq = [bases[0], bases[1], bases[2], bases[1], bases[0]]
    d.ptr_walks = [
        PtrWalk(
            zp=0x20,
            is_load=True,
            is_store=False,
            y_min=0,
            y_max=11,
            count=200,
            ptr_vals=seq,
            adv_frames=list(range(len(seq))),
        )
    ]
    monkeypatch.setattr(SR, "load_distill", lambda _p: d)
    monkeypatch.setattr(SR, "read_sddf_slices", lambda _p: [])
    struct = SR.recover_structure("x")
    assert struct.ok and struct.pattern_lens == [len(p) for p in pats]
    ir = SI.build_structure_ir(struct, None, "x")
    # the stored pattern bytes ARE the read-coverage runs.
    assert ir.pattern_bytes == [list(p) for p in pats]
    # the codec round-trips every serialized field EQUAL.
    SI.assert_ids_roundtrip(ir)
