from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

from prob_wins.utils.hdi_estimation import hdi_estimator

if typing.TYPE_CHECKING:  # pragma: no cover
    import jaxtyping as jtyping

    from prob_wins.utils.confidence_intervals import ConfidenceInterval


@dataclass(frozen=True)
class PosteriorSummary:
    """Summary statistics of some probability distribution."""

    median: float
    ci_probability: float
    hdi: ConfidenceInterval

    @property
    def headers(self) -> list[str]:
        """The column headers."""
        return [
            "Median",
            f"{self.ci_probability * 100:.1f}% HDI",
        ]

    def as_dict(self) -> dict[str, float | ConfidenceInterval]:
        """Returns the dict representation of the statistics.

        Useful for converting to a table.

        Returns:
            dict[str, float | ConfidenceInterval]
        """
        d = {
            "Median": self.median,
            f"{self.ci_probability * 100:.1f}% HDI": self.hdi,
        }

        return d


def summarize_posterior(
    posterior_samples: jtyping.Float[np.ndarray, " num_samples"],
    ci_probability: float,
) -> PosteriorSummary:
    """Summarizes a distribution, assumed to be a posterior, based on samples from it.

    Args:
        posterior_samples (jtyping.Float[np.ndarray, " num_samples"]): samples from the posterior.
        ci_probability (float): the probability under the HDI.

    Returns:
        PosteriorSummary: the summary statistics.
    """
    summary = PosteriorSummary(
        median=typing.cast("float", np.median(posterior_samples)),
        ci_probability=ci_probability,
        hdi=hdi_estimator(posterior_samples, prob=ci_probability),
    )

    return summary
