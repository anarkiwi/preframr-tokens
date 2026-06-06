"""Default parser pipeline spec: the canonical transform ordering."""

from __future__ import annotations

DEFAULT_PIPELINE_SPEC = {
    "transforms": [
        {"name": "squeeze_changes"},
        {"name": "combine_regs"},
        {"name": "quantize_freq_to_cents"},
        {"name": "simplify_ctrl"},
        {"name": "simplify_pcm"},
        {"name": "squeeze_changes"},
        {"name": "add_frame_reg"},
        {"name": "filter"},
        {"name": "squeeze_frame_regs"},
        {"name": "preset"},
        {"name": "per_reg_burst"},
        {"name": "gate_slope_shift"},
        {"name": "consolidate_frames"},
        {"name": "cap_delay"},
        {"name": "rotate_voice_augment"},
        {"name": "norm_pr_order"},
        {"name": "hard_restart"},
        {"name": "legato_per_cluster"},
        {"name": "subreg_flush"},
        {"name": "norm_pr_order"},
        {"name": "voice_block_order"},
        {"name": "loop"},
        {"name": "add_voice_reg"},
    ]
}


def default_pipeline_spec():
    return {"transforms": [dict(t) for t in DEFAULT_PIPELINE_SPEC["transforms"]]}
