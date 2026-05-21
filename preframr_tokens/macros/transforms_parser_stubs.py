"""No-op stub Transform registrations for the hardcoded parser-method steps. The actual implementation lives in reglogparser.py; these stubs exist so MUST_FOLLOW / MUST_PRECEDE / EMITS_NON_SET_REGS declarations can reference them by name and have the static checker enforce the contract."""

from __future__ import annotations

import pandas as pd

from preframr_tokens.macros.transform import Transform, register


class _ParserStub(Transform):
    """Base class for parser-method stubs. forward() is a no-op because the work is hardcoded in reglogparser.py; the stub only exists so other transforms can declare ordering constraints against it."""

    TIER = "bit_exact"

    def forward(self, df: pd.DataFrame, args=None) -> pd.DataFrame:
        return df


@register("squeeze_changes")
class SqueezeChangesStub(_ParserStub):
    IDEMPOTENT = True


@register("combine_regs")
class CombineRegsStub(_ParserStub):
    IDEMPOTENT = True


@register("quantize_freq_to_cents")
class QuantizeFreqToCentsStub(_ParserStub):
    pass


@register("simplify_ctrl")
class SimplifyCtrlStub(_ParserStub):
    pass


@register("simplify_pcm")
class SimplifyPcmStub(_ParserStub):
    pass


@register("add_frame_reg")
class AddFrameRegStub(_ParserStub):
    pass


@register("filter")
class FilterStub(_ParserStub):
    pass


@register("squeeze_frame_regs")
class SqueezeFrameRegsStub(_ParserStub):
    pass


@register("consolidate_frames")
class ConsolidateFramesStub(_ParserStub):
    pass


@register("cap_delay")
class CapDelayStub(_ParserStub):
    pass


@register("rotate_voice_augment")
class RotateVoiceAugmentStub(_ParserStub):
    pass


@register("norm_pr_order")
class NormPrOrderStub(_ParserStub):
    IDEMPOTENT = True


@register("add_voice_reg")
class AddVoiceRegStub(_ParserStub):
    pass
