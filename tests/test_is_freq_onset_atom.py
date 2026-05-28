"""is_freq_onset_atom predicate: True only for op45 V0_HI/V0_LO on FREQ_TRAJ_REGS;
strict subset of is_melody_pitch_atom (which also accepts op48 FREQ_ONSET and op47 NUDGE
pitch). Imported from preframr_tokens so the preframr-side loss-weight buffer cannot
silently drift away from the encoder predicate."""

from preframr_tokens import is_freq_onset_atom, is_melody_pitch_atom
from preframr_tokens.stfconstants import (
    FREQ_NUDGE_OP,
    FREQ_NUDGE_SUBREG_HI,
    FREQ_NUDGE_SUBREG_LO,
    FREQ_ONSET_OP,
    FREQ_TRAJ_OP,
    FT_SUBREG_V0_HI,
    FT_SUBREG_V0_LO,
)


def test_freq_onset_atom_true_on_op45_v0_freq_regs():
    for reg in (0, 7, 14):
        for sr in (FT_SUBREG_V0_HI, FT_SUBREG_V0_LO):
            assert is_freq_onset_atom(FREQ_TRAJ_OP, reg, sr)


def test_freq_onset_atom_false_outside_op45_v0():
    assert not is_freq_onset_atom(FREQ_TRAJ_OP, 0, 6)
    assert not is_freq_onset_atom(FREQ_TRAJ_OP, 2, FT_SUBREG_V0_HI)
    assert not is_freq_onset_atom(FREQ_ONSET_OP, 0, -1)
    assert not is_freq_onset_atom(FREQ_NUDGE_OP, 0, FREQ_NUDGE_SUBREG_HI)
    assert not is_freq_onset_atom(0, 0, -1)


def test_freq_onset_is_strict_subset_of_melody_pitch():
    for op in (FREQ_TRAJ_OP, FREQ_ONSET_OP, FREQ_NUDGE_OP, 0):
        for reg in (0, 2, 7, 14, 21):
            for sr in (
                -1,
                FT_SUBREG_V0_HI,
                FT_SUBREG_V0_LO,
                6,
                FREQ_NUDGE_SUBREG_HI,
                FREQ_NUDGE_SUBREG_LO,
            ):
                if is_freq_onset_atom(op, reg, sr):
                    assert is_melody_pitch_atom(op, reg, sr)
