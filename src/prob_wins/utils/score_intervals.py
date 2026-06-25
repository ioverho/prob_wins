import math

from prob_wins.utils.confidence_intervals import ConfidenceInterval


def get_score_interval_func(method_name: str) -> callable:
    """Fetches a score interval method."""
    match method_name:
        case "ac" | "agresti-coull":
            score_interval_func = agresti_coull_score_interval

        case "wilson":
            score_interval_func = wilson_score_interval

        case _:
            raise ValueError("Score interval method not recognized.")

    return score_interval_func


def wilson_score_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Estimates Wilson score interval around a proportion.

    Adapted from [Stack Overflow](https://stackoverflow.com/a/74035575)

    Args:
        p (float): the relative number of wins
        n (int): the number of total trials
        z (float, optional): the critical value. Defaults to 1.96.

    Returns:
        tuple[float, float]: the interval
    """
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z * z / (2 * n)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)

    ub = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator

    lb = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator

    return ConfidenceInterval(lower=lb, upper=ub)


def agresti_coull_score_interval(
    p: float, n: int, z: float = 1.96
) -> tuple[float, float]:
    """Estimates Agresti-Coull score interval around a proportion.

    Args:
        p (float): the relative number of wins
        n (int): the number of total trials
        z (float, optional): the critical value. Defaults to 1.96.

    Returns:
        tuple[float, float]: the interval
    """
    z2 = z**2

    n_adj = n + z2

    p_adj = (p * n + 0.5 * z2) / n_adj

    delta = z * math.sqrt(p_adj / n_adj * (1 - p_adj))

    lb = p_adj - delta
    ub = p_adj + delta

    return ConfidenceInterval(lower=lb, upper=ub)
