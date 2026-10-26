"""Tests for bootstrap confidence intervals and paired effect sizes."""

import numpy as np
import pytest

from faithful_compression.stats.bootstrap import (
    bootstrap_ci,
    cohens_d_paired,
    disagreement_rate,
    paired_bootstrap_ci,
)


class TestBootstrapCi:
    def test_bounds_within_unit_interval_for_binary_data(
        self, binary_scores_a: np.ndarray
    ) -> None:
        mean, low, high = bootstrap_ci(binary_scores_a)
        assert 0.0 <= low <= mean <= high <= 1.0

    def test_mean_is_the_observed_mean_not_a_resample_mean(
        self, binary_scores_a: np.ndarray
    ) -> None:
        mean, _, _ = bootstrap_ci(binary_scores_a)
        assert mean == pytest.approx(binary_scores_a.mean())

    def test_ci_narrows_with_larger_n(self) -> None:
        rng = np.random.default_rng(0)
        small = rng.binomial(1, 0.6, size=50)
        large = rng.binomial(1, 0.6, size=5000)
        _, small_low, small_high = bootstrap_ci(small)
        _, large_low, large_high = bootstrap_ci(large)
        assert (large_high - large_low) < (small_high - small_low)

    def test_constant_data_gives_degenerate_interval(self) -> None:
        mean, low, high = bootstrap_ci(np.ones(20))
        assert (mean, low, high) == (1.0, 1.0, 1.0)

    def test_is_reproducible_for_a_fixed_seed(self, binary_scores_a: np.ndarray) -> None:
        assert bootstrap_ci(binary_scores_a, seed=7) == bootstrap_ci(binary_scores_a, seed=7)

    def test_different_seeds_can_differ(self) -> None:
        # Continuous data, so the two intervals cannot coincide by landing on
        # the same coarse percentile point as they can with 10 binary items.
        values = np.random.default_rng(3).normal(size=200)
        assert bootstrap_ci(values, seed=1) != bootstrap_ci(values, seed=2)

    def test_wider_ci_level_gives_wider_interval(self, binary_scores_a: np.ndarray) -> None:
        _, low_95, high_95 = bootstrap_ci(binary_scores_a, ci=0.95)
        _, low_80, high_80 = bootstrap_ci(binary_scores_a, ci=0.80)
        assert (high_95 - low_95) >= (high_80 - low_80)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([]))

    def test_invalid_ci_level_raises(self, binary_scores_a: np.ndarray) -> None:
        with pytest.raises(ValueError):
            bootstrap_ci(binary_scores_a, ci=1.5)


class TestPairedBootstrapCi:
    def test_delta_ci_contains_zero_when_arrays_are_equal(
        self, binary_scores_a: np.ndarray
    ) -> None:
        mean, low, high = paired_bootstrap_ci(binary_scores_a, binary_scores_a)
        assert (mean, low, high) == (0.0, 0.0, 0.0)

    def test_mean_delta_matches_difference_of_means(
        self, binary_scores_a: np.ndarray, binary_scores_b: np.ndarray
    ) -> None:
        mean, _, _ = paired_bootstrap_ci(binary_scores_a, binary_scores_b)
        assert mean == pytest.approx(binary_scores_a.mean() - binary_scores_b.mean())

    def test_ci_brackets_the_mean_delta(
        self, binary_scores_a: np.ndarray, binary_scores_b: np.ndarray
    ) -> None:
        mean, low, high = paired_bootstrap_ci(binary_scores_a, binary_scores_b)
        assert low <= mean <= high

    def test_uniform_improvement_excludes_zero(self) -> None:
        a = np.ones(200)
        b = np.zeros(200)
        _, low, _ = paired_bootstrap_ci(a, b)
        assert low > 0.0

    def test_pairing_is_not_the_same_as_unpaired_difference(self) -> None:
        # Same marginal means, different item-level pairing: the paired interval
        # must respond to pairing, otherwise it is not a paired test.
        a = np.array([1, 1, 0, 0] * 25)
        b_aligned = np.array([1, 1, 0, 0] * 25)
        b_shuffled = np.array([0, 0, 1, 1] * 25)
        _, low_aligned, high_aligned = paired_bootstrap_ci(a, b_aligned)
        _, low_shuffled, high_shuffled = paired_bootstrap_ci(a, b_shuffled)
        assert (high_aligned - low_aligned) < (high_shuffled - low_shuffled)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            paired_bootstrap_ci(np.ones(5), np.ones(6))


class TestDisagreementRate:
    def test_zero_when_identical(self, binary_scores_a: np.ndarray) -> None:
        assert disagreement_rate(binary_scores_a, binary_scores_a) == 0.0

    def test_one_when_fully_opposed(self) -> None:
        assert disagreement_rate(np.zeros(10), np.ones(10)) == 1.0

    def test_counts_known_flips(
        self, binary_scores_a: np.ndarray, binary_scores_b: np.ndarray
    ) -> None:
        # Fixtures differ at indices 1, 4 and 6 out of 10 items.
        assert disagreement_rate(binary_scores_a, binary_scores_b) == pytest.approx(0.3)

    def test_is_symmetric(self, binary_scores_a: np.ndarray, binary_scores_b: np.ndarray) -> None:
        forward = disagreement_rate(binary_scores_a, binary_scores_b)
        backward = disagreement_rate(binary_scores_b, binary_scores_a)
        assert forward == backward

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            disagreement_rate(np.ones(5), np.ones(6))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            disagreement_rate(np.array([]), np.array([]))


class TestCohensDPaired:
    def test_zero_when_arrays_are_equal(self, binary_scores_a: np.ndarray) -> None:
        assert cohens_d_paired(binary_scores_a, binary_scores_a) == 0.0

    def test_zero_when_delta_has_no_variance(self) -> None:
        # Every item shifts by exactly 1: mean delta is non-zero but SD is zero.
        assert cohens_d_paired(np.ones(10), np.zeros(10)) == 0.0

    def test_sign_follows_direction_of_effect(
        self, binary_scores_a: np.ndarray, binary_scores_b: np.ndarray
    ) -> None:
        assert cohens_d_paired(binary_scores_a, binary_scores_b) > 0
        assert cohens_d_paired(binary_scores_b, binary_scores_a) < 0

    def test_matches_manual_computation(
        self, binary_scores_a: np.ndarray, binary_scores_b: np.ndarray
    ) -> None:
        deltas = binary_scores_a.astype(float) - binary_scores_b.astype(float)
        expected = deltas.mean() / deltas.std(ddof=1)
        assert cohens_d_paired(binary_scores_a, binary_scores_b) == pytest.approx(expected)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            cohens_d_paired(np.ones(5), np.ones(6))

    def test_single_item_raises(self) -> None:
        with pytest.raises(ValueError):
            cohens_d_paired(np.ones(1), np.zeros(1))
