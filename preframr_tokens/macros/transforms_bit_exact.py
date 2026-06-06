"""Bit-exact Transform wrappers: forward delegates to existing MacroPass, inverse decomposes per-row."""

from __future__ import annotations

from preframr_tokens.macros.decoders import (
    HardRestartDecoder,
    SubregFlushDecoder,
    _LegatoClusterByteDecoder,
    _LegatoClusterNibbleDecoder,
)
from preframr_tokens.macros.passes import (
    HardRestartPass,
    LegatoPerClusterPass,
    SubregPass,
    VoiceBlockOrderPass,
)
from preframr_tokens.macros.transform import (
    PassBackedTransform,
    RowExpandingTransform,
    Transform,
    _row_to_dict,
    register,
)
from preframr_tokens.macros.voice_lane import forward_df, inverse_df
from preframr_tokens.stfconstants import (
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


@register("voice_lane")
class VoiceLaneTransform(Transform):
    """Layer-3 de-multiplex (AGENT_TASK_melody_skeleton.md §4B): reorder a frame-major block into
    voice-major lanes so a melody onset's own predecessor is positionally local. Default OFF; bit-exact
    (forward/inverse restore the canonical render order). MUST_FOLLOW voice_block_order so lanes map to a
    stable canonical voice order."""

    NAME = "voice_lane"
    TIER = "bit_exact"
    OP_CODES = frozenset()
    LOSS_TIER = "structural"
    REQUIRES_ARGS = frozenset({"voice_lane"})
    MUST_FOLLOW = frozenset({"voice_block_order"})

    def forward(self, df, args=None):
        if args is None or not getattr(args, "voice_lane", False):
            return df
        return forward_df(df)

    def inverse(self, df, args=None):
        if args is None or not getattr(args, "voice_lane", False):
            return df
        return inverse_df(df)
