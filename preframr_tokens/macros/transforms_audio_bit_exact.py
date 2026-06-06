"""Audio-bit-exact Transform wrappers: forward delegates to MacroPass, inverse via existing decoder."""

from __future__ import annotations

from preframr_tokens.macros.decoders import (
    DiffDecoder,
    FlipDecoder,
    SetDecoder,
    TransposeDecoder,
)
from preframr_tokens.macros.loop_pass import LoopPass
from preframr_tokens.macros.loops import expand_loops
from preframr_tokens.macros.passes import (
    DedupSetPass,
    TransposePass,
)
from preframr_tokens.macros.transform import (
    PassBackedTransform,
    Transform,
    register,
)
from preframr_tokens.stfconstants import (
    DIFF_OP,
    DO_LOOP_OP,
    FLIP_OP,
    PATTERN_OVERLAY_OP,
    PATTERN_REPLAY_OP,
    PATTERN_REPLAY_SUBREG_DIST_HI,
    PATTERN_REPLAY_SUBREG_DIST_LO,
    SET_OP,
    TRANSPOSE_OP,
)


@register("transpose")
class TransposeTransform(PassBackedTransform):
    TIER = "bit_exact"
    OP_CODES = frozenset({TRANSPOSE_OP})
    SUBSTITUTABLE_OPS = frozenset({TRANSPOSE_OP})
    OPERATES_ON_VOICE_REGS = True
    LOSS_TIER = "content"
    PROVIDES_OPS = frozenset({TRANSPOSE_OP})
    EMITS_NON_SET_REGS = frozenset({0})
    PASS_CLASS = TransposePass
    DECODER_CLASS = TransposeDecoder


@register("dedup_set")
class DedupSetTransform(PassBackedTransform):
    TIER = "audio_bit_exact"
    OP_CODES = frozenset()
    OPERATES_ON_VOICE_REGS = False
    IDEMPOTENT = True
    PASS_CLASS = DedupSetPass


@register("primitives")
class PrimitivesTransform(Transform):
    """Declares the primitive ops (SET/DIFF/FLIP) as snap-substitutable.
    No forward effect — primitive ops are emitted directly by the parser
    and don't go through a compression pass.
    """

    TIER = "bit_exact"
    OP_CODES = frozenset({SET_OP, DIFF_OP, FLIP_OP})
    SUBSTITUTABLE_OPS = frozenset({SET_OP, DIFF_OP, FLIP_OP})
    OPERATES_ON_VOICE_REGS = False
    LOSS_TIER = "content"
    IDEMPOTENT = True
    PROVIDES_OPS = frozenset({SET_OP, DIFF_OP, FLIP_OP})

    def __init__(self, **params):
        super().__init__(**params)
        self._decoders = {
            int(SET_OP): SetDecoder(),
            int(DIFF_OP): DiffDecoder(),
            int(FLIP_OP): FlipDecoder(),
        }

    def forward(self, df, args=None):
        return df

    def expand_atom(self, row, state):
        decoder = self._decoders[int(getattr(row, "op"))]
        return decoder.expand(row, state)


@register("loop")
class LoopTransform(PassBackedTransform):
    TIER = "bit_exact"
    LOSS_TIER = "structural"
    REQUIRES_ARGS = frozenset({"loop_pass"})
    OP_CODES = frozenset(
        {
            DO_LOOP_OP,
            PATTERN_REPLAY_OP,
            PATTERN_OVERLAY_OP,
        }
    )
    PROVIDES_OPS = frozenset(
        {
            DO_LOOP_OP,
            PATTERN_REPLAY_OP,
            PATTERN_OVERLAY_OP,
        }
    )
    SUBSTITUTABLE_OP_SUBREGS = frozenset(
        {
            (PATTERN_OVERLAY_OP, 2),
            (PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_DIST_HI),
            (PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_DIST_LO),
            (DO_LOOP_OP, 0),
        }
    )
    OPERATES_ON_VOICE_REGS = False
    DEFAULT_PARAMS = {"lookahead": 3}
    PARAM_VALIDATORS = {"lookahead": lambda v: isinstance(v, int) and v >= 1}
    DECODES_VIA_DF = True
    PASS_CLASS = LoopPass

    def inverse(self, df, args=None):
        return expand_loops(df.copy())
