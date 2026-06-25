# This module is written entirely by Claude
import numpy as np


def validate_results(
    results: object,
    baseline_results: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce paired result arrays for win-rate comparison.

    Ensures both arrays are 1-D, non-empty, the same length, finite, and have
    a numeric dtype so downstream numpy/scipy operations are safe.

    Args:
        results: Scores for the system being evaluated. Must be array-like of
            real numbers.
        baseline_results: Scores for the baseline system. Must be the same
            length as ``results``.

    Returns:
        Tuple of ``(results, baseline_results)`` as ``np.ndarray``.

    Raises:
        TypeError: If either argument cannot be converted to a numeric array.
        ValueError: If the arrays are empty, not 1-D, differ in length, or
            contain non-finite values (NaN / ±inf).
    """
    try:
        results = np.asarray(results)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"`results` could not be converted to an array: {exc}"
        ) from exc

    try:
        baseline_results = np.asarray(baseline_results)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"`baseline_results` could not be converted to an array: {exc}"
        ) from exc

    if not np.issubdtype(results.dtype, np.number):
        raise TypeError(
            f"`results` must have a numeric dtype, got {results.dtype}."
        )
    if not np.issubdtype(baseline_results.dtype, np.number):
        raise TypeError(
            f"`baseline_results` must have a numeric dtype, got {baseline_results.dtype}."
        )

    if results.ndim != 1:
        raise ValueError(
            f"`results` must be 1-D, got shape {results.shape}."
        )
    if baseline_results.ndim != 1:
        raise ValueError(
            f"`baseline_results` must be 1-D, got shape {baseline_results.shape}."
        )

    if results.size == 0:
        raise ValueError("`results` must not be empty.")

    if results.shape != baseline_results.shape:
        raise ValueError(
            f"`results` and `baseline_results` must have the same length, "
            f"got {results.shape[0]} and {baseline_results.shape[0]}."
        )

    if not np.issubdtype(results.dtype, np.integer) and not np.all(np.isfinite(results)):
        raise ValueError("`results` contains NaN or infinite values.")
    if not np.issubdtype(baseline_results.dtype, np.integer) and not np.all(
        np.isfinite(baseline_results)
    ):
        raise ValueError("`baseline_results` contains NaN or infinite values.")

    return results, baseline_results
