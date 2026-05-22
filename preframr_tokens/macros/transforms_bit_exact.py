"""Bit-exact Transform wrappers: forward delegates to existing MacroPass, inverse decomposes per-row."""

from __future__ import annotations

from preframr_tokens.macros.decoders import (
    CtrlBigramDecoder,
    HardRestartDecoder,
    SubregFlushDecoder,
    _LegatoClusterByteDecoder,
    _LegatoClusterNibbleDecoder,
)
from preframr_tokens.macros.gate_slope_shift_pass import GateSlopeShiftPass
from preframr_tokens.macros.local_macros import CtrlBigramPass
from preframr_tokens.macros.passes import (
    HardRestartPass,
    LegatoPerClusterPass,
    SubregPass,
    VoiceBlockOrderPass,
)
from preframr_tokens.macros.transform import (
    PassBackedTransform,
    RowExpandingTransform,
    _row_to_dict,
    register,
)
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
    LEGATO_OP_CLUSTER_2: _LegatoClusterNibbleDecoder(LEGATO_OP_CLUSTER_2),
    LEGATO_OP_CLUSTER_3: _LegatoClusterNibbleDecoder(LEGATO_OP_CLUSTER_3),
    LEGATO_OP_CLUSTER_4: _LegatoClusterNibbleDecoder(LEGATO_OP_CLUSTER_4),
    LEGATO_OP_CLUSTER_7: _LegatoClusterByteDecoder(LEGATO_OP_CLUSTER_7),
}


@register("hard_restart")
class HardRestartTransform(RowExpandingTransform):
    TIER = "bit_exact"
    OP_CODES = frozenset({HARD_RESTART_OP})
    OPERATES_ON_VOICE_REGS = True
    DECOMPOSES_TO_ATOMS = True
    LOSS_TIER = "structural"
    REQUIRES_ARGS = frozenset({"hard_restart_pass"})
    PROVIDES_OPS = frozenset({HARD_RESTART_OP})
    EMITS_NON_SET_REGS = frozenset({4})
    PASS_CLASS = HardRestartPass
    DECODER_CLASS = HardRestartDecoder

    @staticmethod
    def _expand_row(row):
        packed = int(getattr(row, "val")) & 0xFFFF
        a = (packed >> 8) & 0xFF
        b = packed & 0xFF
        base = _row_to_dict(row, row._fields)
        base["op"] = int(SET_OP)
        first = dict(base)
        first["val"] = int(a)
        second = dict(base)
        second["val"] = int(b)
        return [first, second]


@register("ctrl_bigram")
class CtrlBigramTransform(RowExpandingTransform):
    TIER = "bit_exact"
    OP_CODES = frozenset({CTRL_BIGRAM_OP})
    OPERATES_ON_VOICE_REGS = True
    DECOMPOSES_TO_ATOMS = True
    LOSS_TIER = "zero"
    REQUIRES_ARGS = frozenset({"ctrl_bigram_pass"})
    PROVIDES_OPS = frozenset({CTRL_BIGRAM_OP})
    EMITS_NON_SET_REGS = frozenset({4})
    PASS_CLASS = CtrlBigramPass
    DECODER_CLASS = CtrlBigramDecoder

    @staticmethod
    def _expand_row(row):
        idx = int(getattr(row, "val"))
        prev_byte, cur_byte = CTRL_BIGRAM_TABLE[idx]
        base = _row_to_dict(row, row._fields)
        base["op"] = int(SET_OP)
        first = dict(base)
        first["val"] = int(prev_byte)
        second = dict(base)
        second["val"] = int(cur_byte)
        return [first, second]


@register("gate_slope_shift")
class GateSlopeShiftTransform(PassBackedTransform):
    TIER = "bit_exact"
    OP_CODES = frozenset()
    OPERATES_ON_VOICE_REGS = True
    PASS_CLASS = GateSlopeShiftPass


@register("subreg_flush")
class SubregFlushTransform(PassBackedTransform):
    TIER = "bit_exact"
    OP_CODES = frozenset({SUBREG_FLUSH_OP})
    OPERATES_ON_VOICE_REGS = True
    LOSS_TIER = "structural"
    PROVIDES_OPS = frozenset({SUBREG_FLUSH_OP})
    PASS_CLASS = SubregPass
    DECODER_CLASS = SubregFlushDecoder


_LEGATO_OPS = frozenset(
    {
        LEGATO_OP_CLUSTER_2,
        LEGATO_OP_CLUSTER_3,
        LEGATO_OP_CLUSTER_4,
        LEGATO_OP_CLUSTER_7,
    }
)


@register("legato_per_cluster")
class LegatoPerClusterTransform(RowExpandingTransform):
    TIER = "bit_exact"
    OP_CODES = _LEGATO_OPS
    OPERATES_ON_VOICE_REGS = True
    DECOMPOSES_TO_ATOMS = True
    LOSS_TIER = "mid"
    PROVIDES_OPS = _LEGATO_OPS
    EMITS_NON_SET_REGS = frozenset({4})
    PASS_CLASS = LegatoPerClusterPass

    def expand_atom(self, row, state):
        decoder = _LEGATO_DECODERS[int(getattr(row, "op"))]
        return decoder.expand(row, state)

    @staticmethod
    def _expand_row(row):
        op = int(getattr(row, "op"))
        base = _row_to_dict(row, row._fields)
        base["op"] = int(SET_OP)
        if op == LEGATO_OP_CLUSTER_7:
            base["val"] = int(getattr(row, "val")) & 0xFF
        else:
            base["val"] = (int(getattr(row, "val")) & 0x0F) << 4
        return [base]


@register("voice_block_order")
class VoiceBlockOrderTransform(PassBackedTransform):
    TIER = "bit_exact"
    OP_CODES = frozenset()
    OPERATES_ON_VOICE_REGS = False
    IDEMPOTENT = True
    REQUIRES_ARGS = frozenset({"voice_canonical_block_order"})
    PASS_CLASS = VoiceBlockOrderPass
