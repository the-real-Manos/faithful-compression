"""Bootstrap confidence intervals and paired effect sizes.

All resampling is over items, never over generations, because items are the unit
of pairing. Every function takes an explicit seed so that reported intervals are
reproducible from the raw JSONL.
"""

import numpy as np

DEFAULT_N_RESAMPLES: int = 1000
DEFAULT_CI: float = 0.95
DEFAULT_SEED: int = 42


def _percentile_bounds(ci: float) -> tuple[float, float]:
    """Return the lower and upper percentile points for a central interval."""
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    tail = (1.0 - ci) / 2.0
    return 100.0 * tail, 100.0 * (1.0 - tail)


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) via percentile bootstrap over items.

    Args:
        values: Per-item metric values.
        n_resamples: Number of bootstrap resamples.
        ci: Central interval mass, e.g. 0.95.
        seed: RNG seed.

    Returns:
        Tuple of (observed mean, lower bound, upper bound).

    Raises:
        ValueError: If values is empty.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty array")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[idx].mean(axis=1)
    low_pct, high_pct = _percentile_bounds(ci)
    return (
        float(values.mean()),
        float(np.percentile(means, low_pct)),
        float(np.percentile(means, high_pct)),
    )


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Return (mean_delta, ci_lower, ci_upper) for A - B, resampling item pairs.

    Resampling the paired deltas rather than the two arrays independently is
    what makes this a paired test: it preserves each item's identity across the
    two precisions.

    Args:
        values_a: Per-item values for condition A, item-aligned with B.
        values_b: Per-item values for condition B.
        n_resamples: Number of bootstrap resamples.
        ci: Central interval mass.
        seed: RNG seed.

    Returns:
        Tuple of (observed mean delta, lower bound, upper bound).

    Raises:
        ValueError: If the arrays differ in length.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if values_a.shape != values_b.shape:
        raise ValueError(f"unpaired arrays: {values_a.shape} vs {values_b.shape}")
    return bootstrap_ci(values_a - values_b, n_resamples=n_resamples, ci=ci, seed=seed)


def disagreement_rate(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return the fraction of items where A != B.

    Args:
        values_a: Per-item values for condition A.
        values_b: Per-item values for condition B, item-aligned with A.

    Returns:
        Disagreement rate in [0, 1].

    Raises:
        ValueError: If the arrays differ in length or are empty.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    if values_a.shape != values_b.shape:
        raise ValueError(f"unpaired arrays: {values_a.shape} vs {values_b.shape}")
    if values_a.size == 0:
        raise ValueError("cannot compute disagreement over an empty array")
    return float(np.mean(values_a != values_b))


def cohens_d_paired(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return Cohen's d on paired differences (mean delta / SD of deltas).

    On binary exact-match scores this is a rescaled McNemar statistic and is
    reported alongside, not in place of, the disagreement and flip rates.

    Args:
        values_a: Per-item values for condition A.
        values_b: Per-item values for condition B, item-aligned with A.

    Returns:
        Effect size, or 0.0 when every delta is identical (zero variance),
        which includes the case where A equals B everywhere.

    Raises:
        ValueError: If the arrays differ in length or hold fewer than 2 items.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if values_a.shape != values_b.shape:
        raise ValueError(f"unpaired arrays: {values_a.shape} vs {values_b.shape}")
    if values_a.size < 2:
        raise ValueError("Cohen's d needs at least 2 items")
    deltas = values_a - values_b
    sd = float(deltas.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(deltas.mean() / sd)
