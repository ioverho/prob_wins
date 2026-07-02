import numpy as np
import pytest

from prob_wins.bayesian import BayesianComparisonResult, compare_paired_win_rates_bayesian


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def run(left, right, seed=0, **kwargs):
    return compare_paired_win_rates_bayesian(
        results=np.array(left, dtype=float),
        baseline_results=np.array(right, dtype=float),
        seed=seed,
        **kwargs,
    )


def _generate_mixed_cases():
    """Reproduce the same scenario generator used in the frequentist tests."""
    rng = np.random.default_rng(42)

    scenarios = [
        ("win_heavy_small",   10, 0.60, 0.20, 0.20),
        ("loss_heavy_small",  10, 0.20, 0.60, 0.20),
        ("tie_heavy_small",   10, 0.15, 0.15, 0.70),
        ("balanced_small",    10, 0.34, 0.33, 0.33),
        ("win_heavy_large",  100, 0.60, 0.20, 0.20),
        ("loss_heavy_large", 100, 0.20, 0.60, 0.20),
        ("tie_heavy_large",  100, 0.15, 0.15, 0.70),
        ("balanced_large",   100, 0.34, 0.33, 0.33),
    ]

    cases = []
    for label, n, pw, pl, pt in scenarios:
        outcomes = rng.choice(["win", "loss", "tie"], size=n, p=[pw, pl, pt])
        left  = np.where(outcomes == "win",  2.0,
                np.where(outcomes == "loss", 0.0, 1.0))
        right = np.ones(n)
        if not (np.any(left > right) and np.any(left < right) and np.any(left == right)):
            left[0], left[1], left[2] = 2.0, 0.0, 1.0
        cases.append(pytest.param((left.tolist(), right.tolist()), id=label))

    return cases


@pytest.fixture
def all_wins():
    return run([2] * 20, [1] * 20)


@pytest.fixture
def all_losses():
    return run([1] * 20, [2] * 20)


@pytest.fixture
def all_ties():
    return run([1] * 20, [1] * 20)


@pytest.fixture(params=_generate_mixed_cases())
def mixed(request):
    left, right = request.param
    return run(left, right)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_dataclass(self, mixed):
        assert isinstance(mixed, BayesianComparisonResult)

    def test_frozen(self, mixed):
        with pytest.raises(Exception):
            mixed.prob_win_median = 99.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self):
        left  = [2, 1, 3, 1, 2]
        right = [1, 2, 1, 1, 3]
        r1 = run(left, right, seed=7)
        r2 = run(left, right, seed=7)
        assert r1.prob_win_median == r2.prob_win_median
        assert r1.nb_median == r2.nb_median
        assert r1.p_direction == r2.p_direction

    def test_different_seeds_differ(self):
        left  = [2, 1, 3, 1, 2]
        right = [1, 2, 1, 1, 3]
        r1 = run(left, right, seed=1)
        r2 = run(left, right, seed=2)
        # With few samples the medians might coincidentally match, but
        # the direction probability (a mean) should differ.
        assert r1.p_direction != r2.p_direction


# ---------------------------------------------------------------------------
# Posterior probabilities sum to one (medians are point estimates of the
# Dirichlet marginals — they need not sum exactly, but should be close)
# ---------------------------------------------------------------------------

class TestProbabilityMassConservation:
    def test_median_probs_approx_sum_to_one(self, mixed):
        total = mixed.prob_win_median + mixed.prob_loss_median + mixed.prob_tie_median
        assert total == pytest.approx(1.0, abs=0.1)

    def test_all_wins_win_median_near_one(self, all_wins):
        assert all_wins.prob_win_median > 0.8

    def test_all_losses_loss_median_near_one(self, all_losses):
        assert all_losses.prob_loss_median > 0.8

    def test_all_ties_tie_median_near_one(self, all_ties):
        assert all_ties.prob_tie_median > 0.8


# ---------------------------------------------------------------------------
# HDIs contain the median
# ---------------------------------------------------------------------------

class TestHDIsContainMedian:
    def test_prob_win_hdi(self, mixed):
        assert mixed.prob_win_hdi.lower <= mixed.prob_win_median <= mixed.prob_win_hdi.upper

    def test_prob_loss_hdi(self, mixed):
        assert mixed.prob_loss_hdi.lower <= mixed.prob_loss_median <= mixed.prob_loss_hdi.upper

    def test_prob_tie_hdi(self, mixed):
        assert mixed.prob_tie_hdi.lower <= mixed.prob_tie_median <= mixed.prob_tie_hdi.upper

    def test_wr_hdi(self, mixed):
        assert mixed.wr_hdi.lower <= mixed.wr_median <= mixed.wr_hdi.upper

    def test_wo_hdi(self, mixed):
        assert mixed.wo_hdi.lower <= mixed.wo_median <= mixed.wo_hdi.upper

    def test_nb_hdi(self, mixed):
        assert mixed.nb_hdi.lower <= mixed.nb_median <= mixed.nb_hdi.upper

    def test_test_statistic_hdi(self, mixed):
        assert mixed.test_statistic_hdi.lower <= mixed.test_statistic_median <= mixed.test_statistic_hdi.upper


# ---------------------------------------------------------------------------
# HDIs are ordered (lower < upper)
# ---------------------------------------------------------------------------

class TestHDIOrdering:
    @pytest.mark.parametrize("attr", [
        "prob_win_hdi", "prob_loss_hdi", "prob_tie_hdi",
        "wr_hdi", "wo_hdi", "nb_hdi", "test_statistic_hdi",
    ])
    def test_hdi_lower_lt_upper(self, mixed, attr):
        hdi = getattr(mixed, attr)
        assert hdi.lower < hdi.upper


# ---------------------------------------------------------------------------
# Probability values are in valid ranges
# ---------------------------------------------------------------------------

class TestValueRanges:
    def test_p_direction_in_unit_interval(self, mixed):
        assert 0.0 <= mixed.p_direction <= 1.0

    def test_p_sig_neg_in_unit_interval(self, mixed):
        assert 0.0 <= mixed.p_sig_neg <= 1.0

    def test_p_sig_pos_in_unit_interval(self, mixed):
        assert 0.0 <= mixed.p_sig_pos <= 1.0

    def test_p_sig_bidirectional_in_unit_interval(self, mixed):
        assert 0.0 <= mixed.p_sig_bidirectional <= 1.0

    def test_p_sig_bidirectional_equals_neg_plus_pos(self, mixed):
        assert mixed.p_sig_bidirectional == pytest.approx(
            mixed.p_sig_neg + mixed.p_sig_pos, abs=1e-6
        )

    def test_prob_medians_in_unit_interval(self, mixed):
        assert 0.0 <= mixed.prob_win_median <= 1.0
        assert 0.0 <= mixed.prob_loss_median <= 1.0
        assert 0.0 <= mixed.prob_tie_median <= 1.0


# ---------------------------------------------------------------------------
# Symmetry: swapping results/baseline flips win/loss quantities
# ---------------------------------------------------------------------------

class TestSymmetry:
    @pytest.mark.parametrize("left, right", [
        ([3, 4, 5, 1, 2], [1, 2, 3, 2, 2]),
        ([0.1, 0.9, 0.5], [0.5, 0.5, 0.5]),
        (list(range(1, 11)), list(range(10, 0, -1))),
    ])
    def test_swap_flips_prob_win_and_loss_medians(self, left, right):
        fwd = run(left, right, seed=0)
        rev = run(right, left, seed=0)
        assert fwd.prob_win_median == pytest.approx(rev.prob_loss_median, abs=0.05)
        assert fwd.prob_loss_median == pytest.approx(rev.prob_win_median, abs=0.05)
        assert fwd.prob_tie_median == pytest.approx(rev.prob_tie_median, abs=0.05)

    @pytest.mark.parametrize("left, right", [
        ([3, 4, 5, 1, 2], [1, 2, 3, 2, 2]),
        ([0.1, 0.9, 0.5], [0.5, 0.5, 0.5]),
    ])
    def test_swap_negates_nb_median(self, left, right):
        fwd = run(left, right, seed=0)
        rev = run(right, left, seed=0)
        assert fwd.nb_median == pytest.approx(-rev.nb_median, abs=0.05)

    @pytest.mark.parametrize("left, right", [
        ([3, 4, 5, 1, 2], [1, 2, 3, 2, 2]),
        ([0.1, 0.9, 0.5], [0.5, 0.5, 0.5]),
    ])
    def test_swap_complements_p_direction(self, left, right):
        fwd = run(left, right, seed=0)
        rev = run(right, left, seed=0)
        assert fwd.p_direction == pytest.approx(1 - rev.p_direction, abs=0.05)


# ---------------------------------------------------------------------------
# Direction probability tracks the data
# ---------------------------------------------------------------------------

class TestDirectionProbability:
    def test_all_wins_p_direction_near_one(self, all_wins):
        assert all_wins.p_direction > 0.9

    def test_all_losses_p_direction_near_zero(self, all_losses):
        assert all_losses.p_direction < 0.1

    def test_all_ties_p_direction_near_half(self, all_ties):
        assert 0.3 <= all_ties.p_direction <= 0.7


# ---------------------------------------------------------------------------
# Confidence level: higher level → wider HDIs
# ---------------------------------------------------------------------------

class TestConfidenceLevel:
    @pytest.mark.parametrize("left, right", [
        ([3, 4, 5, 1, 2, 6, 7], [1, 2, 3, 2, 2, 4, 5]),
    ])
    def test_higher_confidence_gives_wider_nb_hdi(self, left, right):
        r90 = run(left, right, seed=0, confidence_level=0.90)
        r99 = run(left, right, seed=0, confidence_level=0.99)
        width_90 = r90.nb_hdi.upper - r90.nb_hdi.lower
        width_99 = r99.nb_hdi.upper - r99.nb_hdi.lower
        assert width_99 > width_90


# ---------------------------------------------------------------------------
# Prior strategies
# ---------------------------------------------------------------------------

class TestPriorStrategies:
    @pytest.mark.parametrize("strategy", [
        "ones", "bayes-laplace", "jeffreys", "haldane",
        0.5, 2.0,
        [0.5, 1.0, 1.5],
    ])
    def test_named_and_numeric_strategies_run(self, strategy):
        r = run([2, 3, 1, 2, 3], [1, 2, 2, 3, 1], prior_strategy=strategy)
        assert isinstance(r, BayesianComparisonResult)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            run([2, 3, 1], [1, 2, 2], prior_strategy="flat")

    def test_negative_scalar_prior_raises(self):
        with pytest.raises(ValueError):
            run([2, 3, 1], [1, 2, 2], prior_strategy=-1.0)

    def test_wrong_length_array_prior_raises(self):
        with pytest.raises(ValueError):
            run([2, 3, 1], [1, 2, 2], prior_strategy=[1.0, 1.0])


# ---------------------------------------------------------------------------
# num_samples propagates correctly
# ---------------------------------------------------------------------------

class TestNumSamples:
    @pytest.mark.parametrize("num_samples", [100, 1000, 5000])
    def test_various_num_samples_run(self, num_samples):
        r = run([2, 3, 1, 2], [1, 2, 2, 3], seed=0, num_samples=num_samples)
        assert isinstance(r, BayesianComparisonResult)

    def test_more_samples_reduces_variance_of_median(self):
        left  = [2, 1, 2, 1, 2] * 4
        right = [1, 2, 1, 2, 1] * 4
        medians_low  = [run(left, right, seed=s, num_samples=200).nb_median  for s in range(20)]
        medians_high = [run(left, right, seed=s, num_samples=5000).nb_median for s in range(20)]
        assert np.std(medians_high) < np.std(medians_low)
