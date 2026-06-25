# Code written by me, documentation written with Claude
import dataclasses

import jaxtyping as jtyping
import numpy as np

from prob_wins.utils.confidence_intervals import ConfidenceInterval
from prob_wins.utils.dirichlet import dirichlet_prior
from prob_wins.utils.outcome_counter import (
    ComparisonOutcomes,
    results_to_comparison_outcomes,
)
from prob_wins.utils.summary import summarize_posterior
from prob_wins.utils.validation import validate_results


@dataclasses.dataclass(frozen=True)
class BayesianComparisonResult:
    """Results of a Bayesian paired win-rate comparison.

    All point estimates are posterior medians; all intervals are highest density
    intervals (HDIs) at the requested confidence level.

    Attributes:
        outcomes (ComparisonOutcomes): Raw win/loss/tie counts from the paired comparison.
        prob_win_median (float): Posterior median of the win probability.
        prob_win_hdi (ConfidenceInterval): HDI for the win probability.
        prob_loss_median (float): Posterior median of the loss probability.
        prob_loss_hdi (ConfidenceInterval): HDI for the loss probability.
        prob_tie_median (float): Posterior median of the tie probability.
        prob_tie_hdi (ConfidenceInterval): HDI for the tie probability.
        wr_median (float): Posterior median of the win rate (prob_win / prob_loss).
        wr_hdi (ConfidenceInterval): HDI for the win rate.
        wo_median (float): Posterior median of the win odds
            ((wins + 0.5*ties) / (losses + 0.5*ties)).
        wo_hdi (ConfidenceInterval): HDI for the win odds.
        nb_median (float): Posterior median of the net benefit (prob_win - prob_loss).
        nb_hdi (ConfidenceInterval): HDI for the net benefit.
        test_statistic_median (float): Posterior median of the Bayesian sign test statistic.
        test_statistic_hdi (ConfidenceInterval): HDI for the test statistic.
        p_direction (float): Probability of direction — posterior mass where test_statistic > 0.
        p_sig_neg (float): Posterior mass below the negative ROPE boundary (system is worse).
        p_sig_pos (float): Posterior mass above the positive ROPE boundary (system is better).
        p_sig_bidirectional (float): Total posterior mass outside the ROPE (p_sig_neg + p_sig_pos).
    """

    outcomes: ComparisonOutcomes

    prob_win_median: float
    prob_win_hdi: ConfidenceInterval

    prob_loss_median: float
    prob_loss_hdi: ConfidenceInterval

    prob_tie_median: float
    prob_tie_hdi: ConfidenceInterval

    wr_median: float
    wr_hdi: ConfidenceInterval

    wo_median: float
    wo_hdi: ConfidenceInterval

    nb_median: float
    nb_hdi: ConfidenceInterval

    test_statistic_median: float
    test_statistic_hdi: ConfidenceInterval

    p_direction: float

    p_sig_neg: float
    p_sig_pos: float
    p_sig_bidirectional: float


def compare_paired_win_rates_frequentist(
    results: jtyping.Float[np.ndarray, "num_results"],  # noqa: F821
    baseline_results: jtyping.Float[np.ndarray, "num_results"],  # noqa: F821
    seed: int,
    confidence_level: float = 0.95,
    prior_strategy: str | float | jtyping.Float[np.typing.ArrayLike, " ..."] = "ones",
    num_samples: int = 10000,
    min_sig_diff: float | None = None,
) -> BayesianComparisonResult:
    """Compare a system against a baseline using a Bayesian paired win-rate analysis.

    Paired outcomes (win/loss/tie) are modelled with a Dirichlet-multinomial conjugate
    model. The posterior is sampled via ``numpy.random.Generator.dirichlet``, and all
    derived statistics (WR, WO, NB, test statistic) are computed sample-wise so that
    their full posterior distributions are available. Point estimates are posterior
    medians; intervals are highest density intervals (HDIs). Significance is assessed
    via the probability of direction and a ROPE-based criterion.

    Args:
        results (jtyping.Float[np.ndarray, "num_results"]): Scores for the system being
            evaluated.
        baseline_results (jtyping.Float[np.ndarray, "num_results"]): Scores for the
            baseline system. Must have the same length as `results`.
        seed (int): Random seed passed to ``numpy.random.default_rng`` for
            reproducibility.
        confidence_level (float): Desired probability mass for all HDIs, e.g. 0.95 for
            95%. Defaults to 0.95.
        prior_strategy (str | float | array-like): Concentration parameter(s) for the
            Dirichlet prior. Named strategies: ``"ones"``/``"bayes-laplace"`` (1.0),
            ``"jeffreys"`` (0.5), ``"haldane"`` (0.0). A scalar float sets all three
            concentrations to that value; an array-like of length 3 sets them
            individually. Defaults to ``"ones"``.
        num_samples (int): Number of posterior samples to draw. Defaults to 10000.
        min_sig_diff (float | None): Half-width of the Region of Practical Equivalence
            (ROPE) around zero for the test statistic. If ``None``, defaults to 10% of
            the posterior standard deviation.

    Returns:
        BayesianComparisonResult: Dataclass containing posterior medians, HDIs, and
            ROPE-based significance probabilities for the comparison.
    """
    results, baseline_results = validate_results(results, baseline_results)

    # Convert results to comparison outcomes
    outcomes: ComparisonOutcomes = results_to_comparison_outcomes(
        left_results=results,
        right_results=baseline_results,
    )

    # Fetch a prior for the Dirichlet distribution
    prior = dirichlet_prior(strategy=prior_strategy, shape=3)

    # Construct likelihood
    observed = np.array([outcomes.num_wins, outcomes.num_losses, outcomes.num_ties])

    # Sample win, loss and tie probs
    rng = np.random.default_rng(seed)
    sampled_probs: jtyping.Float[np.ndarray, "num_samples 3"] = rng.dirichlet(
        alpha=prior + observed,
        size=num_samples,
    )

    # Compute win probabilities
    # Compute win rate statistics
    prob_win = sampled_probs[:, 0]
    prob_win_summary = summarize_posterior(prob_win, ci_probability=confidence_level)

    prob_loss = sampled_probs[:, 1]
    prob_loss_summary = summarize_posterior(prob_loss, ci_probability=confidence_level)

    prob_tie = sampled_probs[:, 2]
    prob_tie_summary = summarize_posterior(prob_tie, ci_probability=confidence_level)

    # Compute win rate statistics
    wr = sampled_probs[:, 0] / sampled_probs[:, 1]
    wr_summary = summarize_posterior(wr, ci_probability=confidence_level)

    wo = (sampled_probs[:, 0] + 0.5 * sampled_probs[:, 2]) / (
        sampled_probs[:, 1] + 0.5 * sampled_probs[:, 2]
    )
    wo_summary = summarize_posterior(wo, ci_probability=confidence_level)

    nb = sampled_probs[:, 0] - sampled_probs[:, 1]
    nb_summary = summarize_posterior(nb, ci_probability=confidence_level)

    # Compute test statistic
    test_statistic = (sampled_probs[:, 0] - sampled_probs[:, 1]) / np.sqrt(
        (sampled_probs[:, 0] + sampled_probs[:, 1]) / outcomes.num_results
    )
    test_statistic_summary = summarize_posterior(
        test_statistic, ci_probability=confidence_level
    )

    # Compute test statistic p direction
    p_direction = np.mean(test_statistic > 0)

    # Compute test statistic ROPE
    # Define a default ROPE
    if min_sig_diff is None:
        min_sig_diff: float = 0.1 * np.std(test_statistic_summary)  # pyright: ignore[reportAssignmentType]

    # Count the number of instances within each bin
    # Significantly negative, within ROPE, significantly positive
    counts, _ = np.histogram(
        test_statistic_summary,
        bins=[-float("inf"), -min_sig_diff, min_sig_diff + 1e-8, float("inf")],
    )

    p_sig_neg, p_rope, p_sig_pos = counts / test_statistic_summary.shape[0]

    p_sig_bidirectional = 1 - p_rope

    # Wrap in result class
    result = BayesianComparisonResult(
        outcomes=outcomes,
        prob_win_median=prob_win_summary.median,
        prob_win_hdi=prob_win_summary.hdi,
        prob_loss_median=prob_loss_summary.median,
        prob_loss_hdi=prob_loss_summary.hdi,
        prob_tie_median=prob_tie_summary.median,
        prob_tie_hdi=prob_tie_summary.hdi,
        wr_median=wr_summary.median,
        wr_hdi=wr_summary.hdi,
        wo_median=wo_summary.median,
        wo_hdi=wo_summary.hdi,
        nb_median=nb_summary.median,
        nb_hdi=nb_summary.hdi,
        test_statistic_median=test_statistic_summary.median,
        test_statistic_hdi=test_statistic_summary.hdi,
        p_direction=p_direction,
        p_sig_neg=p_sig_neg,
        p_sig_pos=p_sig_pos,
        p_sig_bidirectional=p_sig_bidirectional,
    )

    return result
