"""Re-canonicalisation oracle guards: recanon projects an n-space atom stream onto its canonical form --
identity on a canonical keyframe-free block, idempotent, write-preserving, and -- on the windowed,
keyframe-led, possibly mid-event streams DAgger actually rolls out -- prior-state-aware (the leading
KEYFRAME seeds the SID state so the body's relative deltas decode against it). The Tier-4 DAgger oracle
that maps a model rollout onto the nearest valid SID state."""

import glob

import numpy as np
import pytest

from preframr_tokens.events import dataset, generate, oracle, stream
from preframr_tokens.macros import pitch_grid

_REAL_BLOCKS_GLOB = (
    "/scratch/tmp/preframr_experiments/aug_ab_v1/results/generalize_aug_ab/"
    "instrument_full/seed0/eval_b_*/**/*.blocks.npy"
)


def _block():
    """A small voice-0 tune as a keyframe-free n-space atom block (block_to_ids of its oracle)."""
    writes = []
    for f, nt in enumerate((49, 56, 50, 58)):
        fr = pitch_grid.note_freq_at(nt, 0.0)
        writes += [(f, 0, fr & 0xFF), (f, 1, (fr >> 8) & 0xFF), (f, 4, 0x41)]
    ow = oracle.ordered_writes(generate.writes_to_dump_df(writes))
    return [a + 1 for a in stream.encode(ow, verify=False)]


def _whole_tune_atoms():
    """A multi-voice multi-frame tune as a whole-tune atom stream through the real encode path: per voice
    a freq/gate/envelope and constant globals (constant so the global lane is per-frame STEPs, the case a
    constant-init keyframe seed reproduces byte-exact)."""
    writes = []
    notes = (49, 56, 50, 58, 53, 60, 55, 62, 57, 64, 52, 59)
    for f, nt in enumerate(notes):
        for v in range(3):
            fr = pitch_grid.note_freq_at(nt + 3 * v, 0.0)
            lo, hi = 7 * v, 7 * v + 1
            writes += [
                (f, lo, fr & 0xFF),
                (f, hi, (fr >> 8) & 0xFF),
                (f, 7 * v + 4, 0x41),
                (f, 7 * v + 5, 0x49),
                (f, 7 * v + 6, 0xA8),
            ]
        writes += [(f, 24, 0x1F), (f, 22, 40)]
    return stream.encode(oracle.writes_to_ordered(writes), verify=False)


class _FrameWalk(stream._Decoder):  # pylint: disable=protected-access
    """A decoder recording each frame group's ``(atom_pos, abs_frame)`` so a test can build a window at a
    real frame-group boundary."""

    def __init__(self, tokens):
        super().__init__(tokens)
        self.frame_pos = []

    def _parse_body_groups(self, start_f=0, tolerant=False):
        t = self.t
        cur_f = start_f
        last_f = start_f
        m = len(t)
        while self.pos < m:
            p = self.pos
            cur_f += self._u()
            self.frame_pos.append((p, cur_f))
            while self.pos < m and stream._is_voice(t[self.pos]):
                voice = t[self.pos] - stream.VOICE_BASE
                self.pos += 1
                while self.pos < m and t[self.pos] in stream._EVENT_KINDS:
                    self._parse_event(cur_f, voice)
            last_f = cur_f
        return last_f


def _keyframe_window(atoms, frame_index):
    """A keyframe-led window starting at the ``frame_index``-th frame group of ``atoms`` (the production
    chunk_keyframe-prefixed slice) as an n-space id stream; returns ``(n_ids, snapshot_frame)``.
    """
    walk = _FrameWalk(atoms)
    walk.parse()
    apos = walk.frame_pos[frame_index][0]
    kf = stream.chunk_keyframe(atoms, apos)
    assert kf and kf[0] == stream.KEYFRAME
    snap = stream._Decoder(atoms[:apos]).parse(tolerant=True)[1]
    return [a + 1 for a in kf + atoms[apos:]], snap


def _frame_map(writes):
    out = {}
    for f, r, v in writes:
        out.setdefault(f, []).append((r, v))
    return {f: sorted(rv) for f, rv in out.items()}


def test_recanon_identity_on_canonical():
    block = _block()
    assert generate.recanon(block) == block


def test_recanon_idempotent():
    block = _block()
    once = generate.recanon(block)
    assert generate.recanon(once) == once


def test_recanon_preserves_writes():
    block = _block()
    assert dataset.ids_to_writes(generate.recanon(block)) == dataset.ids_to_writes(
        block
    )


def test_recanon_drops_pad():
    block = _block()
    assert generate.recanon(block + [0, 0, 0]) == generate.recanon(block)


def test_recanon_output_is_keyframe_free():
    assert (stream.KEYFRAME + 1) not in generate.recanon(_block())


def test_seed_keyframe_restores_decoder_state():
    """The keyframe seed (inverse of chunk_keyframe) reconstructs the whole-tune decoder's state at the
    snapshot frame EXACTLY: per-voice note-index/freq-residual/CTRL/AD/SR and every global plus the active
    flags -- the binding property the whole prior-state oracle rests on."""
    atoms = _whole_tune_atoms()
    walk = _FrameWalk(atoms)
    walk.parse()
    apos = walk.frame_pos[5][0]
    whole = stream._Decoder(atoms[:apos])
    n, sf = whole.parse(tolerant=True)
    whole.replay(min(sf + 1, n))
    seed = stream._Decoder(stream.chunk_keyframe(atoms, apos))
    seed.seed_keyframe()
    for v in range(3):
        assert seed.freq_active[v] == whole.freq_active[v]
        assert seed.ni[v].at(0) == whole.ni[v].at(sf)
        assert seed.fd[v].at(0) == whole.fd[v].at(sf)
        assert seed._seed_cas[0][v] == whole._ctrl_state[v]
        assert seed._seed_cas[1][v] == whole._ad_state[v]
        assert seed._seed_cas[2][v] == whole._sr_state[v]
    for reg in stream.GLOBAL_REGS:
        assert seed.g_active[reg] == whole.g_active[reg]
        assert seed.g[reg].at(0) == whole.g[reg].at(sf)


def test_decode_windowed_matches_whole_tune_body():
    """The decisive prior-state proof: a keyframe-led window decodes the body to the SAME absolute writes
    the whole tune produces over those frames (frame 0 carries the seeded snapshot as absolute writes, so
    only frames >= 1 are compared). Impossible before -- strip_keyframes decoded the body from zero.
    """
    atoms = _whole_tune_atoms()
    n_ids, sf = _keyframe_window(atoms, 4)
    body = stream.decode_windowed([t - 1 for t in n_ids])
    whole = _frame_map(stream.decode(atoms))
    body_max = max(f for f, _, _ in body)
    for k in range(1, body_max + 1):
        got = sorted((r, v) for f, r, v in body if f == k)
        assert got == whole.get(sf + k, [])


def test_recanon_keyframe_window_idempotent_and_kf_free():
    atoms = _whole_tune_atoms()
    n_ids, _ = _keyframe_window(atoms, 4)
    once = generate.recanon(n_ids)
    assert (stream.KEYFRAME + 1) not in once
    assert generate.recanon(once) == once


def test_recanon_keyframe_window_write_preserving():
    """recanon(window) preserves the seeded (prior-state-correct) absolute writes up to canonicalisation:
    decoding recanon's output equals the canonical form of the windowed decode."""
    atoms = _whole_tune_atoms()
    n_ids, _ = _keyframe_window(atoms, 4)
    body = stream.decode_windowed([t - 1 for t in n_ids])
    canon = stream.decode(stream.encode(oracle.writes_to_ordered(body), verify=False))
    assert dataset.ids_to_writes(generate.recanon(n_ids)) == canon


def test_recanon_window_repromptable():
    """recanon(window) is a valid keyframe-free prompt: it decodes cleanly (extend=True, generation
    continues past the declared frame count) and re-encodes self-consistently -- a model can prime from it
    and continue."""
    atoms = _whole_tune_atoms()
    n_ids, _ = _keyframe_window(atoms, 4)
    rc = generate.recanon(n_ids)
    cont = dataset.ids_to_writes(rc, extend=True)
    assert cont
    again = [a + 1 for a in stream.encode(oracle.writes_to_ordered(cont), verify=False)]
    assert dataset.ids_to_writes(again) == dataset.ids_to_writes(rc, extend=True)


def test_recanon_trims_mid_event_truncation():
    """A fixed-length rollout ends mid-event; recanon(trim=True) trims back to the last whole frame and
    still yields a valid canonical window (problem #1)."""
    atoms = _whole_tune_atoms()
    n_ids, _ = _keyframe_window(atoms, 4)
    rc = generate.recanon(n_ids[:-3], trim=True)
    assert (stream.KEYFRAME + 1) not in rc
    assert generate.recanon(rc) == rc
    assert dataset.ids_to_writes(rc)


def _real_keyframe_rows(limit_files=8):
    rows = []
    for path in sorted(glob.glob(_REAL_BLOCKS_GLOB, recursive=True))[:limit_files]:
        for row in np.load(path):
            atoms = [int(x) - 1 for x in row if int(x) > 0]
            if atoms and atoms[0] == stream.KEYFRAME and stream.KEYFRAME in atoms[1:]:
                rows.append([a + 1 for a in atoms])
    return rows


def test_recanon_on_real_keyframe_blocks():
    """End-to-end on the REAL windowed keyframe-led blocks the experiment produced (skipped when the
    corpus mount is absent): every decodable row trims+seeds, recanon is idempotent + keyframe-free, and
    the body carries real multi-frame content (not just the snapshot) on the bulk of rows.
    """
    rows = _real_keyframe_rows()
    if not rows:
        pytest.skip("real keyframe blocks not mounted")
    decoded = 0
    multiframe = 0
    for n_ids in rows:
        head, writes = stream.trim_to_decodable([t - 1 for t in n_ids])
        if head is None:
            continue
        decoded += 1
        if max(f for f, _, _ in writes) > 0:
            multiframe += 1
        rc = generate.recanon(n_ids, trim=True)
        assert (stream.KEYFRAME + 1) not in rc
        assert generate.recanon(rc) == rc
    assert decoded == len(rows)
    assert multiframe >= len(rows) // 2
