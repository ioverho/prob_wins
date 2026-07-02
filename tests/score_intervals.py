import numpy as np
import pytest

from prob_wins.utils.score_intervals import (
    agresti_coull_score_interval,
    get_score_interval_func,
    wilson_score_interval,
)


# ---------------------------------------------------------------------------
# Shared parametrize helpers
# ---------------------------------------------------------------------------

# (p, n) pairs covering a range of proportions and sample sizes
_PN_CASES = pytest.mark.parametrize(
    "p, n",
    [
        (0.0,  10),
        (0.0,  100),
        (1.0,  10),
        (1.0,  100),
        (0.5,  10),
        (0.5,  100),
        (0.2,  30),
        (0.8,  30),
        (0.1,  5),
        (0.9,  5),
        (0.5,  1),
    ],
)

_METHODS = pytest.mark.parametrize(
    "interval_func",
    [wilson_score_interval, agresti_coull_score_interval],
    ids=["wilson", "agresti_coull"],
)


# ---------------------------------------------------------------------------
# get_score_interval_func dispatch
# ---------------------------------------------------------------------------

class TestGetScoreIntervalFunc:
    @pytest.mark.parametrize("name", ["wilson", "ac", "agresti-coull"])
    def test_known_names_return_callable(self, name):
        func = get_score_interval_func(name)
        assert callable(func)

    def test_wilson_alias_returns_wilson(self):
        assert get_score_interval_func("wilson") is wilson_score_interval

    @pytest.mark.parametrize("name", ["ac", "agresti-coull"])
    def test_ac_aliases_return_agresti_coull(self, name):
        assert get_score_interval_func(name) is agresti_coull_score_interval

    @pytest.mark.parametrize("name", ["", "normal", "clopper", "wilson ", "AC"])
    def test_unknown_names_raise_value_error(self, name):
        with pytest.raises(ValueError):
            get_score_interval_func(name)


# ---------------------------------------------------------------------------
# Shared structural properties (both methods)
# ---------------------------------------------------------------------------

class TestIntervalStructure:
    @_METHODS
    @_PN_CASES
    def test_lower_le_upper(self, interval_func, p, n):
        ci = interval_func(p, n)
        assert ci.lower <= ci.upper

    @_METHODS
    @_PN_CASES
    def test_no_nan_in_output(self, interval_func, p, n):
        ci = interval_func(p, n)
        assert not np.isnan(ci.lower)
        assert not np.isnan(ci.upper)

    @_METHODS
    @_PN_CASES
    def test_no_inf_in_output(self, interval_func, p, n):
        ci = interval_func(p, n)
        assert np.isfinite(ci.lower)
        assert np.isfinite(ci.upper)


# ---------------------------------------------------------------------------
# Bounds clipping: intervals may extend slightly outside [0, 1] by design
# (Agresti-Coull and Wilson can do this), but they should be close.
# ---------------------------------------------------------------------------

class TestIntervalBounds:
    @_METHODS
    @_PN_CASES
    def test_lower_not_far_below_zero(self, interval_func, p, n):
        ci = interval_func(p, n)
        assert ci.lower >= -0.5

    @_METHODS
    @_PN_CASES
    def test_upper_not_far_above_one(self, interval_func, p, n):
        ci = interval_func(p, n)
        assert ci.upper <= 1.5


# ---------------------------------------------------------------------------
# Interior proportions: point estimate inside the interval
# ---------------------------------------------------------------------------

class TestPointEstimateContained:
    @_METHODS
    @pytest.mark.parametrize("p, n", [
        (0.5,  10),
        (0.5,  100),
        (0.3,  20),
        (0.7,  20),
        (0.2,  50),
        (0.8,  50),
    ])
    def test_point_estimate_inside_interval(self, interval_func, p, n):
        ci = interval_func(p, n)
        assert ci.lower <= p <= ci.upper


# ---------------------------------------------------------------------------
# Symmetry: interval for p mirrors interval for 1-p around 0.5
# ---------------------------------------------------------------------------

class TestSymmetry:
    @_METHODS
    @pytest.mark.parametrize("p, n", [
        (0.2, 20),
        (0.3, 50),
        (0.4, 100),
    ])
    def test_interval_symmetric_around_half(self, interval_func, p, n):
        ci_p   = interval_func(p,     n)
        ci_q   = interval_func(1 - p, n)
        assert ci_p.lower == pytest.approx(1 - ci_q.upper)
        assert ci_p.upper == pytest.approx(1 - ci_q.lower)


# ---------------------------------------------------------------------------
# Monotonicity: larger n → narrower interval for fixed interior p
# ---------------------------------------------------------------------------

class TestMonotonicity:
    @_METHODS
    @pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
    def test_wider_interval_for_smaller_n(self, interval_func, p):
        ci_small = interval_func(p, n=10)
        ci_large = interval_func(p, n=200)
        width_small = ci_small.upper - ci_small.lower
        width_large = ci_large.upper - ci_large.lower
        assert width_small > width_large

    @_METHODS
    @pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
    def test_wider_interval_for_larger_z(self, interval_func, p):
        ci_narrow = interval_func(p, n=50, z=1.645)   # ~90 %
        ci_wide   = interval_func(p, n=50, z=2.576)   # ~99 %
        width_narrow = ci_narrow.upper - ci_narrow.lower
        width_wide   = ci_wide.upper   - ci_wide.lower
        assert width_wide > width_narrow


# ---------------------------------------------------------------------------
# Known values: Wilson at p=0.5, n=100, z=1.96 ≈ [0.402, 0.598]
# Reference: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
# ---------------------------------------------------------------------------

class TestKnownValues:
    def test_wilson_z_zero_collapses_to_point(self):
        # z=0 → no uncertainty, interval collapses to [p, p]
        ci = wilson_score_interval(0.3, 50, z=0.0)
        assert ci.lower == pytest.approx(0.3)
        assert ci.upper == pytest.approx(0.3)

    def test_agresti_coull_z_zero_collapses_to_point(self):
        ci = agresti_coull_score_interval(0.3, 50, z=0.0)
        assert ci.lower == pytest.approx(0.3)
        assert ci.upper == pytest.approx(0.3)

    def test_agresti_coull_half_n100(self):
        # AC at p=0.5, n=100 should be very close to Wilson
        ci_w = wilson_score_interval(0.5, 100, z=1.96)
        ci_a = agresti_coull_score_interval(0.5, 100, z=1.96)
        assert ci_a.lower == pytest.approx(ci_w.lower, abs=0.01)
        assert ci_a.upper == pytest.approx(ci_w.upper, abs=0.01)
