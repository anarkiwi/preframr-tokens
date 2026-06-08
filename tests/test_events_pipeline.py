"""Events-native tokenization pipeline guards (REDESIGN_optionB §7.1): frame-window blocking is byte-exact
per block and reassembles to the whole tune, and the block array is the fixed-shape model input.
"""

import numpy as np

from preframr_tokens.events import oracle, pipeline


def _ow(writes, n):
    return oracle.OrderedWrites(
        frame=np.array([f for f, _, _ in writes], dtype=np.int64),
        reg=np.array([r for _, r, _ in writes], dtype=np.int64),
        val=np.array([v for _, _, v in writes], dtype=np.int64),
        n_frames=n,
        irq=np.arange(n, dtype=np.int64),
    )


def _synthetic():
    writes = []
    base = 0x1240
    for f in range(64):
        fr = base if f < 8 else base + int(6 * ((f % 4) - 2))
        writes.append((f, 0, fr & 0xFF))
        writes.append((f, 1, (fr >> 8) & 0xFF))
        writes.append((f, 4, 0x40 if f < 4 else (0x41 if f < 40 else 0x10)))
        writes.append((f, 5, 0x08))
        writes.append((f, 6, 0xA9))
        writes.append((f, 2, 64 + f))
    return _ow(sorted(writes, key=lambda t: t[0]), 64)


def test_windows_are_byte_exact_and_reassemble():
    ow = _synthetic()
    fpb = 16
    wins = list(pipeline.iter_windows(ow, fpb))
    recon = []
    for i, w in enumerate(wins):
        triples = pipeline.block_writes(pipeline.block_tokens(w))
        assert triples == w.triples(), f"block {i} diverged"
        recon += [(f + i * fpb, r, v) for f, r, v in triples]
    assert (
        recon == ow.triples()
    ), "non-overlapping windows must reassemble the whole tune"


def test_block_array_shape_and_alphabet():
    ow = _synthetic()
    block_size = 129
    arr = pipeline.block_array(_synth_df(ow), block_size, frames_per_block=16)
    assert arr.dtype == np.int32 and arr.shape[1] == block_size
    assert arr.shape[0] == len(list(pipeline.iter_windows(ow, 16)))
    assert int(arr.max()) < pipeline.VOCAB_SIZE
    assert pipeline.atoms() == list(range(pipeline.VOCAB_SIZE))


def _synth_df(ow):
    import pandas as pd

    return pd.DataFrame(
        {
            "clock": np.arange(len(ow), dtype=np.int64),
            "irq": ow.frame.astype(np.int64),
            "chipno": np.zeros(len(ow), dtype=np.int64),
            "reg": ow.reg,
            "val": ow.val,
        }
    )
