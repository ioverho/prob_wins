import numpy as np
import pytest

from prob_wins.utils.outcome_counter import ComparisonOutcomes, results_to_comparison_outcomes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_outcomes(left, right):
    return results_to_comparison_outcomes(
        left_results=np.array(left, dtype=float),
        right_results=np.array(right, dtype=float),
    )


# ---------------------------------------------------------------------------
# ComparisonOutcomes dataclass
# ---------------------------------------------------------------------------

class TestComparisonOutcomes:
    def test_frozen(self):
        outcomes = ComparisonOutcomes(num_results=3, num_wins=1, num_losses=1, num_ties=1)
        with pytest.raises(Exception):
            outcomes.num_wins = 99

    def test_fields_accessible(self):
        outcomes = ComparisonOutcomes(num_results=5, num_wins=3, num_losses=1, num_ties=1)
        assert outcomes.num_results == 5
        assert outcomes.num_wins == 3
        assert outcomes.num_losses == 1
        assert outcomes.num_ties == 1


# ---------------------------------------------------------------------------
# All-wins / all-losses / all-ties
# ---------------------------------------------------------------------------

class TestUniformOutcomes:
    def test_all_wins(self):
        outcomes = make_outcomes([2, 3, 4], [1, 2, 3])
        assert outcomes.num_wins == 3
        assert outcomes.num_losses == 0
        assert outcomes.num_ties == 0
        assert outcomes.num_results == 3

    def test_all_losses(self):
        outcomes = make_outcomes([1, 2, 3], [2, 3, 4])
        assert outcomes.num_wins == 0
        assert outcomes.num_losses == 3
        assert outcomes.num_ties == 0
        assert outcomes.num_results == 3

    def test_all_ties(self):
        outcomes = make_outcomes([1, 2, 3], [1, 2, 3])
        assert outcomes.num_wins == 0
        assert outcomes.num_losses == 0
        assert outcomes.num_ties == 3
        assert outcomes.num_results == 3

    def test_single_win(self):
        outcomes = make_outcomes([1.0], [0.0])
        assert outcomes.num_wins == 1
        assert outcomes.num_losses == 0
        assert outcomes.num_ties == 0
        assert outcomes.num_results == 1

    def test_single_loss(self):
        outcomes = make_outcomes([0.0], [1.0])
        assert outcomes.num_wins == 0
        assert outcomes.num_losses == 1
        assert outcomes.num_ties == 0

    def test_single_tie(self):
        outcomes = make_outcomes([1.0], [1.0])
        assert outcomes.num_wins == 0
        assert outcomes.num_losses == 0
        assert outcomes.num_ties == 1


# ---------------------------------------------------------------------------
# Known mixed splits
# ---------------------------------------------------------------------------

class TestMixedOutcomes:
    def test_two_wins_one_loss(self):
        outcomes = make_outcomes([2, 3, 1], [1, 4, 2])
        assert outcomes.num_wins == 1
        assert outcomes.num_losses == 2
        assert outcomes.num_ties == 0
        assert outcomes.num_results == 3

    def test_wins_losses_ties(self):
        left  = [3, 1, 2, 4, 2]
        right = [1, 3, 2, 2, 5]
        outcomes = make_outcomes(left, right)
        assert outcomes.num_wins == 2
        assert outcomes.num_losses == 2
        assert outcomes.num_ties == 1
        assert outcomes.num_results == 5

    def test_counts_sum_to_num_results(self):
        left  = [1, 2, 3, 4, 5]
        right = [5, 4, 3, 2, 1]
        outcomes = make_outcomes(left, right)
        assert outcomes.num_wins + outcomes.num_losses + outcomes.num_ties == outcomes.num_results


# ---------------------------------------------------------------------------
# Duplicate pairs are counted correctly (np.unique collapses them)
# ---------------------------------------------------------------------------

class TestDuplicatePairs:
    def test_repeated_win_pairs(self):
        # (2, 1) appears three times — should count as 3 wins
        outcomes = make_outcomes([2, 2, 2], [1, 1, 1])
        assert outcomes.num_wins == 3
        assert outcomes.num_losses == 0
        assert outcomes.num_ties == 0

    def test_repeated_tie_pairs(self):
        outcomes = make_outcomes([1, 1, 1], [1, 1, 1])
        assert outcomes.num_ties == 3
        assert outcomes.num_results == 3

    def test_mixed_with_duplicates(self):
        # (1,2) loss × 2, (2,2) tie × 2, (3,3) tie × 1
        left  = [1, 1, 2, 2, 3]
        right = [2, 2, 2, 2, 3]
        outcomes = make_outcomes(left, right)
        assert outcomes.num_losses == 2
        assert outcomes.num_ties == 3
        assert outcomes.num_wins == 0


# ---------------------------------------------------------------------------
# Symmetry: swapping left/right flips wins and losses
# ---------------------------------------------------------------------------

class TestSymmetry:
    @pytest.mark.parametrize("left, right", [
        ([1, 2, 3], [3, 2, 1]),
        ([0.5, 1.5], [1.0, 1.0]),
        ([10, 20, 30, 40], [15, 15, 35, 35]),
    ])
    def test_swap_flips_wins_and_losses(self, left, right):
        fwd = make_outcomes(left, right)
        rev = make_outcomes(right, left)
        assert fwd.num_wins == rev.num_losses
        assert fwd.num_losses == rev.num_wins
        assert fwd.num_ties == rev.num_ties
        assert fwd.num_results == rev.num_results


# ---------------------------------------------------------------------------
# Numeric edge cases
# ---------------------------------------------------------------------------

class TestNumericEdgeCases:
    def test_negative_scores(self):
        outcomes = make_outcomes([-1, -2, -3], [-2, -3, -4])
        assert outcomes.num_wins == 3
        assert outcomes.num_losses == 0

    def test_zero_scores(self):
        outcomes = make_outcomes([0.0, 0.0], [0.0, 0.0])
        assert outcomes.num_ties == 2

    def test_large_array(self):
        n = 10_000
        left = np.ones(n)
        right = np.zeros(n)
        outcomes = results_to_comparison_outcomes(left, right)
        assert outcomes.num_wins == n
        assert outcomes.num_results == n
