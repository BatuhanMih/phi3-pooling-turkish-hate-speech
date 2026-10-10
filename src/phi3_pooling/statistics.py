"""Small paired-analysis helpers used by the repository examples."""

from collections.abc import Sequence
from itertools import product

import numpy as np


def _paired_differences(first: Sequence[float], second: Sequence[float]) -> np.ndarray:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.size != right.size:
        raise ValueError("paired samples must be one-dimensional and equally sized")
    if left.size == 0:
        raise ValueError("paired samples cannot be empty")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("paired samples must contain finite values")
    return right - left


def exact_sign_flip_pvalue(
    first: Sequence[float],
    second: Sequence[float],
    *,
    two_sided: bool = True,
) -> float:
    """Compute an exact paired sign-flip p-value using the mean difference."""

    differences = _paired_differences(first, second)
    observed = float(differences.mean())
    null_means = np.asarray(
        [np.mean(differences * signs) for signs in product((-1.0, 1.0), repeat=len(differences))]
    )
    tolerance = np.finfo(float).eps * 16
    if two_sided:
        extreme = np.abs(null_means) >= abs(observed) - tolerance
    elif observed >= 0:
        extreme = null_means >= observed - tolerance
    else:
        extreme = null_means <= observed + tolerance
    return float(extreme.mean())


def paired_bootstrap_interval(
    first: Sequence[float],
    second: Sequence[float],
    *,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval for the paired mean difference ``second - first``."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    differences = _paired_differences(first, second)
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0, len(differences), size=(replicates, len(differences)))
    means = differences[indexes].mean(axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(means, (alpha, 1 - alpha))
    return float(lower), float(upper)

