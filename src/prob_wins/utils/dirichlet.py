from __future__ import annotations

import typing

if typing.TYPE_CHECKING:  # pragma: no cover
    import jaxtyping as jtyping

import numpy as np

_DIRICHLET_PRIOR_STRATEGIES = {
    "bayes-laplace": 1.0,
    "bayes": 1.0,
    "laplace": 1.0,
    "ones": 1.0,
    "one": 1.0,
    "jeffrey": 0.5,
    "jeffreys": 0.5,
    "halves": 0.5,
    "half": 0.5,
    "haldane": 0.0,
    "zeros": 0.0,
    "zero": 0.0,
}


# TODO: add support for other dtypes
def dirichlet_prior(
    strategy: str | float | jtyping.Float[np.typing.ArrayLike, " ..."],
    shape: tuple[int, ...],
) -> jtyping.Float[np.ndarray, " ..."]:
    """Creates a prior array for a Dirichlet distribution.

    Returns:
        jtyping.Float[np.ndarray, " ..."]: the prior vector
    """
    if isinstance(strategy, float | int):
        if strategy < 0:
            raise ValueError(
                f"A Dirichlet prior must contain positive values. Received: {strategy}",
            )

        prior = np.full(shape, fill_value=strategy, dtype=np.float64)

    elif isinstance(strategy, str):
        if strategy not in _DIRICHLET_PRIOR_STRATEGIES:
            raise ValueError(
                (
                    f"Prior strategy `{strategy}` not recognized. "
                    f"Choose one of: {set(_DIRICHLET_PRIOR_STRATEGIES.keys())}"
                ),
            )

        strategy_fill_value = _DIRICHLET_PRIOR_STRATEGIES[strategy]
        prior = np.full(shape, fill_value=strategy_fill_value, dtype=np.float64)

    else:
        try:
            # TODO: control dtype
            prior = np.array(strategy, dtype=np.float64)
        except Exception as e:
            raise ValueError(
                (
                    f"While trying to convert {strategy} to a numpy array, "
                    f"received the following error:\n{e}"
                ),
            ) from e

        if prior.shape != shape:
            raise ValueError(
                (
                    f"Prior does not match required shape, {prior.shape} != {shape}. "
                    f"Parsed {prior} of type {type(prior)} "
                    f"from {strategy} of type {type(strategy)}."
                ),
            )

    return prior
