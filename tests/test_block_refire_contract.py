"""Block-decoder reference contract: every pass emitting a back-ref / loop / inline-codebook-id op must
be in the per-block re-fire pipeline, so a self-contained window the model trains on RE-EMITS the
reference instead of expanding it to literal. Guards the bug where LoopPass (DO_LOOP /
PATTERN_*) lived only in ``run_post_norm_pre_voice_passes`` and was never re-fired on blocks.
"""

from preframr_tokens.macros import block_refire_pass_names
from preframr_tokens.macros.op_contracts import (
    OP_PRODUCER,
    reference_op_producers,
    reference_ops,
)


def test_every_reference_op_declares_a_producer():
    ops = reference_ops()
    assert ops, "expected back-ref / loop / codebook-ref ops in the contract registry"
    assert ops <= set(OP_PRODUCER), "a reference op is missing an OP_PRODUCER mapping"
    assert reference_op_producers()


def test_block_refire_covers_every_reference_producer():
    """The structural contract: the per-block re-fire pipeline includes every pass that emits a
    reference op. Fails until LoopPass (and any future ref-emitting pass) is re-fired on blocks.
    """
    missing = reference_op_producers() - block_refire_pass_names()
    assert not missing, (
        "reference/ID-emitting passes NOT re-fired on self-contained blocks "
        f"(their refs would expand to literal and never reach the model): {sorted(missing)}"
    )
