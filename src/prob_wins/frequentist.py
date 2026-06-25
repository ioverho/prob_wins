# Code written by me, documentation written with Claude
import dataclasses

import jaxtyping as jtyping
import numpy as np
import scipy
import scipy.stats

from prob_wins.utils.confidence_intervals import (
    ConfidenceInterval,
    confidence_to_critical_value,
)
from prob_wins.utils.outcome_counter import (
    ComparisonOutcomes,
    results_to_comparison_outcomes,
)
from prob_wins.utils.score_intervals import get_score_interval_func


@dataclasses.dataclass(frozen=True)
class FrequentistComparisonResult:
    """Results of a frequentist paired win-rate comparison.

    Attributes:
        outcomes (ComparisonOutcomes): Raw win/loss/tie counts from the paired comparison.
        prob_win (float): Estimated probability that the system outperforms the baseline.
        prob_win_ci (ConfidenceInterval): Confidence interval for `prob_win`.
        prob_loss (float): Estimated probability that the system underperforms the baseline.
        prob_loss_ci (ConfidenceInterval): Confidence interval for `prob_loss`.
        prob_tie (float): Estimated probability of a tie.
        prob_tie_ci (ConfidenceInterval): Confidence interval for `prob_tie`.
        wr (float): Win rate — ratio of wins to losses (prob_win / prob_loss).
        wr_ci (ConfidenceInterval): Confidence interval for `wr` computed via MOVER.
        wo (float): Win odds — (wins + 0.5*ties) / (losses + 0.5*ties).
        wo_ci (ConfidenceInterval): Confidence interval for `wo` derived from the NB interval.
        nb (float): Net benefit — difference of win and loss probabilities (prob_win - prob_loss).
        nb_ci (ConfidenceInterval): Confidence interval for `nb` computed via MOVER.
        test_statistic (float): Sign test statistic: (wins - losses) / sqrt(wins + losses).
        test_p_val_one_sided (float): One-sided p-value for the sign test (H1: system > baseline).
        test_p_val_two_sided (float): Two-sided p-value for the sign test.
    """

    outcomes: ComparisonOutcomes

    prob_win: float
    prob_win_ci: ConfidenceInterval

    prob_loss: float
    prob_loss_ci: ConfidenceInterval

    prob_tie: float
    prob_tie_ci: ConfidenceInterval

    wr: float
    wr_ci: ConfidenceInterval

    wo: float
    wo_ci: ConfidenceInterval

    nb: float
    nb_ci: ConfidenceInterval

    test_statistic: float
    test_p_val_one_sided: float
    test_p_val_two_sided: float


def compare_paired_win_rates_frequentist(
    results: jtyping.Float[np.ndarray, "num_results"],  # noqa: F821
    baseline_results: jtyping.Float[np.ndarray, "num_results"],  # noqa: F821
    confidence_level: float = 0.95,
    score_interval_method: str = "ac",
) -> FrequentistComparisonResult:
    """Compare a system against a baseline using a frequentist paired win-rate analysis.

    For each paired sample, the system either wins (result > baseline), loses
    (result < baseline), or ties (result == baseline). Point estimates and confidence
    intervals are computed for the win/loss/tie probabilities and three derived
    statistics: win rate (WR), win odds (WO), and net benefit (NB). Confidence
    intervals for WR and NB use the MOVER (Method of Variance Estimates Recovery)
    approach, which accounts for the negative correlation between win and loss
    probabilities. Statistical significance is assessed via the sign test.

    Args:
        results (jtyping.Float[np.ndarray, "num_results"]): Scores for the system being evaluated.
        baseline_results (jtyping.Float[np.ndarray, "num_results"]): Scores for the baseline system.
            Must have the same length as `results`.
        confidence_level (float): Desired confidence level for all intervals, e.g. 0.95 for 95%.
            Defaults to 0.95.
        score_interval_method (str): Method used to compute score (proportion) confidence
            intervals. Defaults to ``"ac"`` (Agresti–Coull).

    Returns:
        FrequentistComparisonResult: Dataclass containing point estimates, confidence intervals,
            and sign-test p-values for the comparison.
    """
    # Convert a confidence level to a standard Normal critical value
    critical_value = confidence_to_critical_value(confidence_level)

    # Convert results to comparison outcomes
    outcomes: ComparisonOutcomes = results_to_comparison_outcomes(
        left_results=results,
        right_results=baseline_results,
    )

    # Compute the win, loss and tie probabilities
    prob_win = outcomes.num_wins / outcomes.num_results
    prob_loss = outcomes.num_losses / outcomes.num_results
    prob_tie = outcomes.num_ties / outcomes.num_results

    # Get score interval function
    score_interval_func = get_score_interval_func(score_interval_method)

    # Compute score intervals for the probability estimates
    prob_win_ci = score_interval_func(
        prob_win,
        outcomes.num_results,
        z=critical_value,
    )

    prob_loss_ci = score_interval_func(
        prob_loss,
        outcomes.num_results,
        z=critical_value,
    )

    prob_tie_ci = score_interval_func(
        prob_tie,
        outcomes.num_results,
        z=critical_value,
    )

    # Compute win and loss probabilities correlation
    correlation = -(prob_win * prob_loss) / np.sqrt(
        prob_win * (1 - prob_win) * prob_loss * (1 - prob_loss)
    )

    # Compute win rate statistics
    try:
        wr = prob_win / prob_loss
    except ZeroDivisionError:
        wr = np.nan

    wo = (prob_win + 0.5 * prob_tie) / (prob_loss + 0.5 * prob_tie)

    nb = prob_win - prob_loss

    # Compute CI intervals for the different win rate statistics
    # MOVER - WR
    wr_mover_lb_numerator = prob_win * prob_loss - correlation * (
        prob_win - prob_win_ci.lower
    ) * (prob_loss_ci.upper - prob_loss)

    wr_mover_lb_denominator = prob_loss_ci.upper * (2 * prob_loss - prob_loss_ci.upper)

    wr_mover_lb = (
        wr_mover_lb_numerator / wr_mover_lb_denominator
        - np.sqrt(
            wr_mover_lb_numerator**2
            - prob_win_ci.lower
            * prob_loss_ci.upper
            * (2 * prob_win - prob_win_ci.lower)
            * (2 * prob_loss - prob_loss_ci.upper)
        )
        / wr_mover_lb_denominator
    )

    wr_mover_ub_numerator = prob_win * prob_loss - correlation * (
        prob_win_ci.upper - prob_win
    ) * (prob_loss - prob_loss_ci.lower)

    wr_mover_ub_denominator = prob_loss_ci.lower * (2 * prob_loss - prob_loss_ci.lower)

    wr_mover_ub = (
        wr_mover_ub_numerator / wr_mover_ub_denominator
        + np.sqrt(
            wr_mover_ub_numerator**2
            - prob_win_ci.upper
            * prob_loss_ci.lower
            * (2 * prob_win - prob_win_ci.upper)
            * (2 * prob_loss - prob_loss_ci.lower)
        )
        / wr_mover_ub_denominator
    )

    wr_ci = ConfidenceInterval(wr_mover_lb, wr_mover_ub)

    # MOVER - NB
    nb_mover_lb = nb - np.sqrt(
        (prob_win - prob_win_ci.lower) ** 2
        + (prob_loss_ci.upper - prob_loss) ** 2
        - 2
        * correlation
        * (prob_win - prob_win_ci.lower)
        * (prob_loss_ci.upper - prob_loss)
    )

    nb_mover_ub = nb + np.sqrt(
        (prob_win_ci.upper - prob_win) ** 2
        + (prob_loss - prob_loss_ci.lower) ** 2
        - 2
        * correlation
        * (prob_win_ci.upper - prob_win)
        * (prob_loss - prob_loss_ci.lower)
    )

    nb_ci = ConfidenceInterval(nb_mover_lb, nb_mover_ub)

    # MOVER - NB
    wo_mover_lb = (1 + nb_mover_lb) / (1 - nb_mover_lb)
    wo_mover_ub = (1 + nb_mover_ub) / (1 - nb_mover_ub)

    wo_ci = ConfidenceInterval(wo_mover_lb, wo_mover_ub)

    # Compute test statistic for WR
    test_statistic = (outcomes.num_wins - outcomes.num_losses) / np.sqrt(
        outcomes.num_wins + outcomes.num_losses
    )

    test_p_val_one_sided = scipy.stats.norm.cdf(
        test_statistic if test_statistic < 0 else -test_statistic
    )

    test_p_val_two_sided = 2 * test_p_val_one_sided

    result = FrequentistComparisonResult(
        outcomes=outcomes,
        prob_win=prob_win,
        prob_win_ci=prob_win_ci,
        prob_loss=prob_loss,
        prob_loss_ci=prob_loss_ci,
        prob_tie=prob_tie,
        prob_tie_ci=prob_tie_ci,
        wr=wr,
        wr_ci=wr_ci,
        wo=wo,
        wo_ci=wo_ci,
        nb=nb,
        nb_ci=nb_ci,
        test_statistic=test_statistic,
        test_p_val_one_sided=test_p_val_one_sided,
        test_p_val_two_sided=test_p_val_two_sided,
    )

    return result
