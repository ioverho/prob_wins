import numpy as np
import pytest

from prob_wins.frequentist import FrequentistComparisonResult, compare_paired_win_rates_frequentist


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def run(left, right, **kwargs):
    return compare_paired_win_rates_frequentist(
        results=np.array(left, dtype=float),
        baseline_results=np.array(right, dtype=float),
        **kwargs,
    )


def _generate_mixed_cases():
    """Generate diverse (left, right, label) triples using a fixed RNG.

    Each case draws scores from a Dirichlet-multinomial so that wins, losses,
    and ties all appear, but in varying proportions.  The seed is fixed so the
    suite is fully reproducible.
    """
    rng = np.random.default_rng(42)

    # (label, n, win_prob, loss_prob, tie_prob)
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
        # Ensure at least one win, loss, and tie so degenerate edge cases stay
        # in the dedicated all_wins / all_losses / all_ties fixtures.
        if not (np.any(left > right) and np.any(left < right) and np.any(left == right)):
            left[0], left[1], left[2] = 2.0, 0.0, 1.0
        cases.append(pytest.param((left.tolist(), right.tolist()), id=label))

    return cases


@pytest.fixture
def all_wins():
    return run([2, 3, 4, 5], [1, 2, 3, 4])


@pytest.fixture
def all_losses():
    return run([1, 2, 3, 4], [2, 3, 4, 5])


@pytest.fixture
def all_ties():
    return run([1, 2, 3, 4], [1, 2, 3, 4])


@pytest.fixture(params=_generate_mixed_cases())
def mixed(request):
    left, right = request.param
    return run(left, right)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_dataclass(self, mixed):
        assert isinstance(mixed, FrequentistComparisonResult)

    def test_frozen(self, mixed):
        with pytest.raises(Exception):
            mixed.prob_win = 99.0


# ---------------------------------------------------------------------------
# Probabilities sum to one
# ---------------------------------------------------------------------------

class TestProbabilitiesSumToOne:
    def test_all_wins(self, all_wins):
        assert all_wins.prob_win + all_wins.prob_loss + all_wins.prob_tie == pytest.approx(1.0)

    def test_all_losses(self, all_losses):
        assert all_losses.prob_win + all_losses.prob_loss + all_losses.prob_tie == pytest.approx(1.0)

    def test_all_ties(self, all_ties):
        assert all_ties.prob_win + all_ties.prob_loss + all_ties.prob_tie == pytest.approx(1.0)

    def test_mixed(self, mixed):
        assert mixed.prob_win + mixed.prob_loss + mixed.prob_tie == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Point estimates match outcome counts
# ---------------------------------------------------------------------------

class TestPointEstimates:
    def test_all_wins_prob_win_is_one(self, all_wins):
        assert all_wins.prob_win == pytest.approx(1.0)
        assert all_wins.prob_loss == pytest.approx(0.0)
        assert all_wins.prob_tie == pytest.approx(0.0)

    def test_all_losses_prob_loss_is_one(self, all_losses):
        assert all_losses.prob_loss == pytest.approx(1.0)
        assert all_losses.prob_win == pytest.approx(0.0)
        assert all_losses.prob_tie == pytest.approx(0.0)

    def test_all_ties_prob_tie_is_one(self, all_ties):
        assert all_ties.prob_tie == pytest.approx(1.0)
        assert all_ties.prob_win == pytest.approx(0.0)
        assert all_ties.prob_loss == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CIs contain the point estimate
# ---------------------------------------------------------------------------

class TestCIsContainPointEstimate:
    def test_prob_win_ci(self, mixed):
        assert mixed.prob_win_ci.lower <= mixed.prob_win <= mixed.prob_win_ci.upper

    def test_prob_loss_ci(self, mixed):
        assert mixed.prob_loss_ci.lower <= mixed.prob_loss <= mixed.prob_loss_ci.upper

    def test_prob_tie_ci(self, mixed):
        assert mixed.prob_tie_ci.lower <= mixed.prob_tie <= mixed.prob_tie_ci.upper

    def test_nb_ci(self, mixed):
        assert mixed.nb_ci.lower <= mixed.nb <= mixed.nb_ci.upper

    def test_wr_ci(self, mixed):
        assert mixed.wr_ci.lower <= mixed.wr <= mixed.wr_ci.upper

    def test_wo_ci(self, mixed):
        assert mixed.wo_ci.lower <= mixed.wo <= mixed.wo_ci.upper

    @pytest.mark.parametrize("left, right", [
        ([1, 2, 3], [3, 2, 1]),
        ([0.1, 0.9], [0.5, 0.5]),
        ([10] * 20, [5] * 15 + [15] * 5),
    ])
    def test_all_cis_contain_point_estimate(self, left, right):
        r = run(left, right)
        assert r.prob_win_ci.lower <= r.prob_win <= r.prob_win_ci.upper
        assert r.prob_loss_ci.lower <= r.prob_loss <= r.prob_loss_ci.upper
        assert r.prob_tie_ci.lower <= r.prob_tie <= r.prob_tie_ci.upper
        assert r.nb_ci.lower <= r.nb <= r.nb_ci.upper


# ---------------------------------------------------------------------------
# Net benefit arithmetic identity: NB = prob_win - prob_loss
# ---------------------------------------------------------------------------

class TestNetBenefitIdentity:
    def test_mixed(self, mixed):
        assert mixed.nb == pytest.approx(mixed.prob_win - mixed.prob_loss)

    def test_all_wins(self, all_wins):
        assert all_wins.nb == pytest.approx(all_wins.prob_win - all_wins.prob_loss)

    def test_all_ties(self, all_ties):
        assert all_ties.nb == pytest.approx(0.0)

    @pytest.mark.parametrize("left, right", [
        ([1, 2, 3, 4], [4, 3, 2, 1]),
        ([0.0] * 10, [1.0] * 10),
        ([1, 1, 1, 2, 2], [2, 2, 1, 1, 1]),
    ])
    def test_nb_equals_prob_win_minus_prob_loss(self, left, right):
        r = run(left, right)
        assert r.nb == pytest.approx(r.prob_win - r.prob_loss)


# ---------------------------------------------------------------------------
# Win rate = prob_win / prob_loss
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_wr_ratio(self, mixed):
        assert mixed.wr == pytest.approx(mixed.prob_win / mixed.prob_loss)

    def test_wr_inf_when_no_losses(self, all_wins):
        # numpy float division returns inf, not nan, so the ZeroDivisionError guard is never hit
        assert np.isinf(all_wins.wr)

    def test_wr_zero_when_no_wins(self, all_losses):
        assert all_losses.wr == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Win odds = (prob_win + 0.5*prob_tie) / (prob_loss + 0.5*prob_tie)
# ---------------------------------------------------------------------------

class TestWinOdds:
    def test_wo_identity(self, mixed):
        expected = (mixed.prob_win + 0.5 * mixed.prob_tie) / (mixed.prob_loss + 0.5 * mixed.prob_tie)
        assert mixed.wo == pytest.approx(expected)

    def test_wo_one_when_all_ties(self, all_ties):
        assert all_ties.wo == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Symmetry: swapping results/baseline flips win/loss stats
# ---------------------------------------------------------------------------

class TestSymmetry:
    @pytest.mark.parametrize("left, right", [
        ([3, 4, 5, 1, 2], [1, 2, 3, 2, 2]),
        ([0.1, 0.9, 0.5], [0.5, 0.5, 0.5]),
        (list(range(1, 11)), list(range(10, 0, -1))),
    ])
    def test_swap_flips_prob_win_and_loss(self, left, right):
        fwd = run(left, right)
        rev = run(right, left)
        assert fwd.prob_win == pytest.approx(rev.prob_loss)
        assert fwd.prob_loss == pytest.approx(rev.prob_win)
        assert fwd.prob_tie == pytest.approx(rev.prob_tie)

    @pytest.mark.parametrize("left, right", [
        ([3, 4, 5, 1, 2], [1, 2, 3, 2, 2]),
        ([0.1, 0.9, 0.5], [0.5, 0.5, 0.5]),
    ])
    def test_swap_negates_nb(self, left, right):
        fwd = run(left, right)
        rev = run(right, left)
        assert fwd.nb == pytest.approx(-rev.nb)

    @pytest.mark.parametrize("left, right", [
        ([3, 4, 5, 1, 2], [1, 2, 3, 2, 2]),
        ([0.1, 0.9, 0.5], [0.5, 0.5, 0.5]),
    ])
    def test_swap_inverts_wr(self, left, right):
        fwd = run(left, right)
        rev = run(right, left)
        assert fwd.wr == pytest.approx(1 / rev.wr)


# ---------------------------------------------------------------------------
# Sign test
# ---------------------------------------------------------------------------

class TestSignTest:
    def test_all_wins_low_p_value(self, all_wins):
        assert all_wins.test_p_val_one_sided < 0.05

    def test_all_losses_low_p_value(self, all_losses):
        assert all_losses.test_p_val_one_sided > 0.05

    def test_all_ties_statistic_is_nan(self, all_ties):
        # wins=0, losses=0 → sqrt(0) in denominator
        assert np.isnan(all_ties.test_statistic)

    def test_two_sided_is_twice_min_tail(self, mixed):
        # two-sided = 2 * min(one_sided, 1 - one_sided)
        assert mixed.test_p_val_two_sided == pytest.approx(
            2 * min(mixed.test_p_val_one_sided, 1 - mixed.test_p_val_one_sided)
        )

    def test_p_values_in_unit_interval(self, mixed):
        assert 0.0 <= mixed.test_p_val_one_sided <= 1.0
        assert 0.0 <= mixed.test_p_val_two_sided <= 1.0

    def test_statistic_sign_matches_nb(self, mixed):
        assert np.sign(mixed.test_statistic) == np.sign(mixed.nb)


# ---------------------------------------------------------------------------
# Confidence level propagation
# ---------------------------------------------------------------------------

class TestConfidenceLevel:
    @pytest.mark.parametrize("confidence_level", [0.80, 0.90, 0.95, 0.99])
    def test_wider_ci_at_higher_confidence(self, confidence_level):
        left  = [3, 4, 5, 1, 2, 6, 7]
        right = [1, 2, 3, 2, 2, 4, 5]
        r = run(left, right, confidence_level=confidence_level)
        width = r.nb_ci.upper - r.nb_ci.lower
        assert width > 0

    def test_higher_confidence_gives_wider_nb_ci(self):
        left  = [3, 4, 5, 1, 2, 6, 7]
        right = [1, 2, 3, 2, 2, 4, 5]
        r90 = run(left, right, confidence_level=0.90)
        r99 = run(left, right, confidence_level=0.99)
        width_90 = r90.nb_ci.upper - r90.nb_ci.lower
        width_99 = r99.nb_ci.upper - r99.nb_ci.lower
        assert width_99 > width_90


# ---------------------------------------------------------------------------
# Score interval methods
# ---------------------------------------------------------------------------

class TestScoreIntervalMethods:
    @pytest.mark.parametrize("method", ["ac", "agresti-coull", "wilson"])
    def test_known_methods_run_without_error(self, method):
        r = run([2, 3, 4], [1, 2, 3], score_interval_method=method)
        assert r.prob_win_ci.lower <= r.prob_win <= r.prob_win_ci.upper

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            run([2, 3, 4], [1, 2, 3], score_interval_method="bad_method")
