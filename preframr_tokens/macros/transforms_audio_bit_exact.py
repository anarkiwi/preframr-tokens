"""Audio-bit-exact Transform wrappers: forward delegates to MacroPass, inverse via existing decoder."""

from __future__ import annotations

from preframr_tokens.macros.decoders import (
    DiffDecoder,
    Flip2Decoder,
    FlipDecoder,
    PresetDecoder,
    SetDecoder,
    ShiftedDecoder,
    SlopeDecoder,
    TransposeDecoder,
)
from preframr_tokens.macros.loop_pass import LoopPass
from preframr_tokens.macros.loops import expand_loops
from preframr_tokens.macros.passes import (
    DedupSetPass,
    Flip2Pass,
    TransposePass,
)
from preframr_tokens.macros.per_reg_burst import PerRegBurstPass
from preframr_tokens.macros.preset_pass import PresetPass
from preframr_tokens.macros.slope_pass import SlopePass
from preframr_tokens.macros.transform import Transform, register
from preframr_tokens.stfconstants import (
    BACK_REF_OP,
    BACK_REF_SUBREG_DIST_HI,
    BACK_REF_SUBREG_DIST_LO,
    DIFF_OP,
    DO_LOOP_OP,
    FC_PRESET_OP,
    FLIP_OP,
    FLIP2_OP,
    PATTERN_OVERLAY_OP,
    PATTERN_REPLAY_OP,
    PATTERN_REPLAY_SUBREG_DIST_HI,
    PATTERN_REPLAY_SUBREG_DIST_LO,
    PWM_PRESET_OP,
    PWM_PRESET_SHIFTED_OP,
    SET_OP,
    SLOPE_FC_LO_OP,
    SLOPE_FREQ_LO_OP,
    SLOPE_FREQ_LO_SHIFTED_OP,
    SLOPE_PW_LO_OP,
    SLOPE_PW_LO_SHIFTED_OP,
    TRANSPOSE_OP,
)

_SLOPE_OPS = (
    SLOPE_FREQ_LO_OP,
    SLOPE_PW_LO_OP,
    SLOPE_FC_LO_OP,
    SLOPE_FREQ_LO_SHIFTED_OP,
    SLOPE_PW_LO_SHIFTED_OP,
)
_PRESET_OPS = (PWM_PRESET_OP, FC_PRESET_OP, PWM_PRESET_SHIFTED_OP)


@register("slope")
class SlopeTransform(Transform):
    TIER = "audio_bit_exact"
    OP_CODES = frozenset(_SLOPE_OPS)
    SUBSTITUTABLE_OP_SUBREGS = frozenset(
        (int(op), int(sr)) for op in _SLOPE_OPS for sr in (0, 1, 2, 3, 4)
    )
    OPERATES_ON_VOICE_REGS = True
    LOSS_TIER = "content"
    REQUIRES_ARGS = frozenset({"slope_pass"})
    PROVIDES_OPS = frozenset(_SLOPE_OPS)
    EMITS_NON_SET_REGS = frozenset({0, 2, 21})

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = SlopePass()
        self._decoder = SlopeDecoder()
        self._shifted_decoder = ShiftedDecoder()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)

    def expand_atom(self, row, state):
        op = int(getattr(row, "op"))
        if op in (SLOPE_FREQ_LO_SHIFTED_OP, SLOPE_PW_LO_SHIFTED_OP):
            return self._shifted_decoder.expand(row, state)
        return self._decoder.expand(row, state)


@register("preset")
class PresetTransform(Transform):
    TIER = "audio_bit_exact"
    OP_CODES = frozenset(_PRESET_OPS)
    SUBSTITUTABLE_OP_SUBREGS = frozenset((int(op), -1) for op in _PRESET_OPS)
    OPERATES_ON_VOICE_REGS = True
    LOSS_TIER = "content"
    REQUIRES_ARGS = frozenset({"preset_pass"})
    PROVIDES_OPS = frozenset(_PRESET_OPS)
    EMITS_NON_SET_REGS = frozenset({2, 21})

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = PresetPass()
        self._decoder = PresetDecoder()
        self._shifted_decoder = ShiftedDecoder()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)

    def expand_atom(self, row, state):
        op = int(getattr(row, "op"))
        if op == PWM_PRESET_SHIFTED_OP:
            return self._shifted_decoder.expand(row, state)
        return self._decoder.expand(row, state)


@register("per_reg_burst")
class PerRegBurstTransform(Transform):
    TIER = "audio_bit_exact"
    OP_CODES = frozenset()
    OPERATES_ON_VOICE_REGS = True

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = PerRegBurstPass()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)


@register("transpose")
class TransposeTransform(Transform):
    TIER = "bit_exact"
    OP_CODES = frozenset({TRANSPOSE_OP})
    SUBSTITUTABLE_OPS = frozenset({TRANSPOSE_OP})
    OPERATES_ON_VOICE_REGS = True
    LOSS_TIER = "content"
    PROVIDES_OPS = frozenset({TRANSPOSE_OP})
    EMITS_NON_SET_REGS = frozenset({0})

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = TransposePass()
        self._decoder = TransposeDecoder()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)

    def expand_atom(self, row, state):
        return self._decoder.expand(row, state)


@register("flip2")
class Flip2Transform(Transform):
    TIER = "bit_exact"
    OP_CODES = frozenset({FLIP2_OP})
    SUBSTITUTABLE_OPS = frozenset({FLIP2_OP})
    OPERATES_ON_VOICE_REGS = True
    LOSS_TIER = "content"
    PROVIDES_OPS = frozenset({FLIP2_OP})
    EMITS_NON_SET_REGS = frozenset({0, 2})

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = Flip2Pass()
        self._decoder = Flip2Decoder()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)

    def expand_atom(self, row, state):
        return self._decoder.expand(row, state)


@register("dedup_set")
class DedupSetTransform(Transform):
    TIER = "audio_bit_exact"
    OP_CODES = frozenset()
    OPERATES_ON_VOICE_REGS = False
    IDEMPOTENT = True

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = DedupSetPass()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)


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
class LoopTransform(Transform):
    TIER = "bit_exact"
    LOSS_TIER = "structural"
    REQUIRES_ARGS = frozenset({"loop_pass"})
    OP_CODES = frozenset(
        {
            BACK_REF_OP,
            DO_LOOP_OP,
            PATTERN_REPLAY_OP,
            PATTERN_OVERLAY_OP,
        }
    )
    PROVIDES_OPS = frozenset(
        {
            BACK_REF_OP,
            DO_LOOP_OP,
            PATTERN_REPLAY_OP,
            PATTERN_OVERLAY_OP,
        }
    )
    SUBSTITUTABLE_OP_SUBREGS = frozenset(
        {
            (PATTERN_OVERLAY_OP, 2),
            (BACK_REF_OP, BACK_REF_SUBREG_DIST_HI),
            (BACK_REF_OP, BACK_REF_SUBREG_DIST_LO),
            (PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_DIST_HI),
            (PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_DIST_LO),
            (DO_LOOP_OP, 0),
        }
    )
    OPERATES_ON_VOICE_REGS = False
    DEFAULT_PARAMS = {"lookahead": 3}
    PARAM_VALIDATORS = {"lookahead": lambda v: isinstance(v, int) and v >= 1}
    DECODES_VIA_DF = True

    def __init__(self, **params):
        super().__init__(**params)
        self._impl = LoopPass()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)

    def inverse(self, df, args=None):
        return expand_loops(df.copy())
