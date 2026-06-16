"""Re-canonicalisation oracle guards: recanon projects an n-space atom stream onto its canonical form --
identity on a canonical keyframe-free block, idempotent, write-preserving, and keyframe-stripping (the
Tier-4 DAgger oracle that maps a model rollout onto the nearest valid SID state)."""

from preframr_tokens.events import dataset, generate, oracle, stream
from preframr_tokens.macros import pitch_grid


def _block():
    """A small voice-0 tune as a keyframe-free n-space atom block (block_to_ids of its oracle)."""
    writes = []
    for f, nt in enumerate((49, 56, 50, 58)):
        fr = pitch_grid.note_freq_at(nt, 0.0)
        writes += [(f, 0, fr & 0xFF), (f, 1, (fr >> 8) & 0xFF), (f, 4, 0x41)]
    ow = oracle.ordered_writes(generate.writes_to_dump_df(writes))
    return [a + 1 for a in stream.encode(ow, verify=False)]


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
