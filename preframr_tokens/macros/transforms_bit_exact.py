"""Bit-exact Transform wrappers: forward delegates to existing MacroPass, inverse decomposes per-row."""

from __future__ import annotations

from typing import Any

import pandas as pd

from preframr_tokens.macros.decoders import (
    CtrlBigramDecoder,
    HardRestartDecoder,
    LegatoCluster2Decoder,
    LegatoCluster3Decoder,
    LegatoCluster4Decoder,
    LegatoCluster7Decoder,
    SubregFlushDecoder,
)
from preframr_tokens.macros.gate_slope_shift_pass import GateSlopeShiftPass
from preframr_tokens.macros.local_macros import CtrlBigramPass
from preframr_tokens.macros.passes import (
    HardRestartPass,
    LegatoPerClusterPass,
    SubregPass,
    VoiceBlockOrderPass,
)
from preframr_tokens.macros.transform import Transform, register
from preframr_tokens.stfconstants import (
    CTRL_BIGRAM_OP,
    CTRL_BIGRAM_TABLE,
    HARD_RESTART_OP,
    LEGATO_OP_CLUSTER_2,
    LEGATO_OP_CLUSTER_3,
    LEGATO_OP_CLUSTER_4,
    LEGATO_OP_CLUSTER_7,
    SET_OP,
    SUBREG_FLUSH_OP,
)

_LEGATO_DECODERS = {
    LEGATO_OP_CLUSTER_2: LegatoCluster2Decoder(),
    LEGATO_OP_CLUSTER_3: LegatoCluster3Decoder(),
    LEGATO_OP_CLUSTER_4: LegatoCluster4Decoder(),
    LEGATO_OP_CLUSTER_7: LegatoCluster7Decoder(),
}


def _expand_op_rows(df: pd.DataFrame, op_codes, expand_fn) -> pd.DataFrame:
    if "op" not in df.columns or df.empty:
        return df
    out_rows = []
    for row in df.itertuples(index=False):
        if int(getattr(row, "op")) in op_codes:
            out_rows.extend(expand_fn(row))
        else:
            out_rows.append(_row_to_dict(row, df.columns))
    if not out_rows:
        return df.iloc[0:0]
    return pd.DataFrame(out_rows, columns=df.columns).astype(df.dtypes.to_dict())


def _row_to_dict(row, columns):
    return {c: getattr(row, c) for c in columns}


@register("hard_restart")
class HardRestartTransform(Transform):
    TIER = "bit_exact"
    OP_CODES = frozenset({HARD_RESTART_OP})
    OPERATES_ON_VOICE_REGS = True
    DECOMPOSES_TO_ATOMS = True
    LOSS_TIER = "structural"
    REQUIRES_ARGS = frozenset({"hard_restart_pass"})
    PROVIDES_OPS = frozenset({HARD_RESTART_OP})
    EMITS_NON_SET_REGS = frozenset({4})

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = HardRestartPass()
        self._decoder = HardRestartDecoder()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)

    def inverse(self, df, args=None):
        return _expand_op_rows(df, self.OP_CODES, self._expand_row)

    def expand_atom(self, row, state):
        return self._decoder.expand(row, state)

    @staticmethod
    def _expand_row(row):
        packed = int(getattr(row, "val")) & 0xFFFF
        a = (packed >> 8) & 0xFF
        b = packed & 0xFF
        base = {c: getattr(row, c) for c in row._fields}
        base["op"] = int(SET_OP)
        first = dict(base)
        first["val"] = int(a)
        second = dict(base)
        second["val"] = int(b)
        return [first, second]


@register("ctrl_bigram")
class CtrlBigramTransform(Transform):
    TIER = "bit_exact"
    OP_CODES = frozenset({CTRL_BIGRAM_OP})
    OPERATES_ON_VOICE_REGS = True
    DECOMPOSES_TO_ATOMS = True
    LOSS_TIER = "zero"
    REQUIRES_ARGS = frozenset({"ctrl_bigram_pass"})
    PROVIDES_OPS = frozenset({CTRL_BIGRAM_OP})
    EMITS_NON_SET_REGS = frozenset({4})

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = CtrlBigramPass()
        self._decoder = CtrlBigramDecoder()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)

    def inverse(self, df, args=None):
        return _expand_op_rows(df, self.OP_CODES, self._expand_row)

    def expand_atom(self, row, state):
        return self._decoder.expand(row, state)

    @staticmethod
    def _expand_row(row):
        idx = int(getattr(row, "val"))
        prev_byte, cur_byte = CTRL_BIGRAM_TABLE[idx]
        base = {c: getattr(row, c) for c in row._fields}
        base["op"] = int(SET_OP)
        first = dict(base)
        first["val"] = int(prev_byte)
        second = dict(base)
        second["val"] = int(cur_byte)
        return [first, second]


@register("gate_slope_shift")
class GateSlopeShiftTransform(Transform):
    TIER = "bit_exact"
    OP_CODES = frozenset()
    OPERATES_ON_VOICE_REGS = True

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = GateSlopeShiftPass()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)


@register("subreg_flush")
class SubregFlushTransform(Transform):
    TIER = "bit_exact"
    OP_CODES = frozenset({SUBREG_FLUSH_OP})
    OPERATES_ON_VOICE_REGS = True
    LOSS_TIER = "structural"
    PROVIDES_OPS = frozenset({SUBREG_FLUSH_OP})

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = SubregPass()
        self._decoder = SubregFlushDecoder()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)

    def expand_atom(self, row, state):
        return self._decoder.expand(row, state)


_LEGATO_OPS = frozenset(
    {
        LEGATO_OP_CLUSTER_2,
        LEGATO_OP_CLUSTER_3,
        LEGATO_OP_CLUSTER_4,
        LEGATO_OP_CLUSTER_7,
    }
)


@register("legato_per_cluster")
class LegatoPerClusterTransform(Transform):
    TIER = "bit_exact"
    OP_CODES = _LEGATO_OPS
    OPERATES_ON_VOICE_REGS = True
    DECOMPOSES_TO_ATOMS = True
    LOSS_TIER = "mid"
    PROVIDES_OPS = _LEGATO_OPS
    EMITS_NON_SET_REGS = frozenset({4})

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = LegatoPerClusterPass()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)

    def inverse(self, df, args=None):
        return _expand_op_rows(df, self.OP_CODES, self._expand_row)

    def expand_atom(self, row, state):
        decoder = _LEGATO_DECODERS[int(getattr(row, "op"))]
        return decoder.expand(row, state)

    @staticmethod
    def _expand_row(row):
        op = int(getattr(row, "op"))
        base = {c: getattr(row, c) for c in row._fields}
        base["op"] = int(SET_OP)
        if op == LEGATO_OP_CLUSTER_7:
            base["val"] = int(getattr(row, "val")) & 0xFF
        else:
            base["val"] = (int(getattr(row, "val")) & 0x0F) << 4
        return [base]


@register("voice_block_order")
class VoiceBlockOrderTransform(Transform):
    TIER = "bit_exact"
    OP_CODES = frozenset()
    OPERATES_ON_VOICE_REGS = False
    IDEMPOTENT = True
    REQUIRES_ARGS = frozenset({"voice_canonical_block_order"})

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = VoiceBlockOrderPass()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)
