import dataclasses

import scipy
import scipy.stats


@dataclasses.dataclass(frozen=True)
class ConfidenceInterval:
    """Confidence interval consisting of an upper and a lower bound."""

    lower: float
    upper: float


def confidence_to_critical_value(confidence_level: float = 0.95):
    """Converts a confidence level to a critical value according to standard Normal distribution."""
    critical_value = scipy.stats.norm.ppf(
        confidence_level + (1 - confidence_level) / 2,
        loc=0,
        scale=1,
    )

    return critical_value
