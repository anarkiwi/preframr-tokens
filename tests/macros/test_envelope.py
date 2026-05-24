"""Unit tests for the parametric envelope families (OSCILLATE_ENV item 0)."""

import pytest

from preframr_tokens.macros import envelope as env

ALL_FAMILIES = [
    env.ENV_CONSTANT,
    env.ENV_LINEAR,
    env.ENV_EXP_DECAY,
    env.ENV_EXP_GROWTH,
    env.ENV_STEP_2,
    env.ENV_STEP_3,
    env.ENV_STEP_4,
    env.ENV_TRIANGULAR,
]


def test_cycle_multipliers_length_and_baseline():
    for family in ALL_FAMILIES:
        for n in (1, 3, 5, 8):
            mult = env.cycle_multipliers(family, 0x80, n)
            assert len(mult) == n
            assert all(m >= 0.0 for m in mult)


def test_empty_and_degenerate():
    assert env.cycle_multipliers(env.ENV_CONSTANT, 0, 0) == []
    fam, _param, res = env.fit_family([])
    assert fam == env.ENV_CONSTANT
    assert res == float("inf")


def test_constant_fits_flat_chain():
    fam, _param, res = env.fit_family([1.0, 1.0, 1.0, 1.0])
    assert fam == env.ENV_CONSTANT
    assert res <= env.FIT_TOLERANCE


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_synthetic_chain_round_trips_within_tolerance(family):
    """A chain generated from a family must be re-fit and reconstructed
    within FIT_TOLERANCE (self-consistency of forward+inverse)."""
    n = 6
    param = {
        env.ENV_CONSTANT: 0,
        env.ENV_LINEAR: env._q_ratio_to_param(0.5),
        env.ENV_EXP_DECAY: env._q07_to_param(0.7),
        env.ENV_EXP_GROWTH: env._q_ratio_to_param(1.2),
        env.ENV_STEP_2: env._q07_to_param(0.5),
        env.ENV_STEP_3: env._pack_step_levels([3, 1, 2]),
        env.ENV_STEP_4: env._pack_step_levels([3, 0, 2, 1]),
        env.ENV_TRIANGULAR: (2 << 4) | 7,
    }[family]
    norm_amps = env.cycle_multipliers(family, param, n)
    base = norm_amps[0] if abs(norm_amps[0]) > 1e-9 else 1.0
    norm_amps = [a / base for a in norm_amps]
    fit_family, fit_param, res = env.fit_family(norm_amps)
    recon = env.cycle_multipliers(fit_family, fit_param, n)
    assert res <= env.FIT_TOLERANCE, (family, fit_family, res)
    for a, r in zip(norm_amps, recon):
        assert abs(a - r) <= 0.15 * max(abs(a), 1e-6) + 0.05


def test_decay_chain_prefers_decay_over_constant():
    norm_amps = [1.0, 0.7, 0.49, 0.343, 0.24]
    fam, _param, res = env.fit_family(norm_amps)
    assert fam == env.ENV_EXP_DECAY
    assert res <= env.FIT_TOLERANCE


def test_step2_alternation():
    norm_amps = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5]
    fam, _param, res = env.fit_family(norm_amps)
    assert fam in (env.ENV_STEP_2, env.ENV_STEP_3, env.ENV_STEP_4)
    assert res <= env.FIT_TOLERANCE
