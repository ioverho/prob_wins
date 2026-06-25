import dataclasses

import jaxtyping as jtyping
import numpy as np


@dataclasses.dataclass(frozen=True)
class ComparisonOutcomes:
    """The number of different outcomes between two separate result lists.

    Attributes:
        num_results (int): the total number of results
        num_wins (int): the total number of wins (left > right)
        num_losses (int): the total number of losses (right > left)
        num_ties (int): the total number of ties (left == right)
    """

    num_results: int
    num_wins: int
    num_losses: int
    num_ties: int


def results_to_comparison_outcomes(
    left_results: jtyping.Float[np.ndarray, "num_results"],  # noqa: F821
    right_results: jtyping.Float[np.ndarray, "num_results"],  # noqa: F821
):
    """Converts two arrays of results to comparison outcomes."""
    outcomes, counts = np.unique(
        np.stack([left_results, right_results], axis=1), return_counts=True, axis=0
    )

    num_results = outcomes.shape[0]

    wins = 0
    losses = 0
    ties = 0
    for outcome, count in zip(outcomes, counts, strict=True):
        if outcome[0] == outcome[1]:
            ties += count
        elif outcome[0] > outcome[1]:
            wins += count
        elif outcome[1] > outcome[0]:
            losses += count
        else:
            raise ValueError

    outcomes = ComparisonOutcomes(
        num_results=num_results,
        num_wins=wins,
        num_losses=losses,
        num_ties=ties,
    )

    return outcomes
