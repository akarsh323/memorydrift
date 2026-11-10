"""Tests for the statistics behind the results tables."""

import math

import pytest

from memorydrift.analysis import (
    Estimate,
    paired_difference,
    pass_hat_k,
    pooled_rate,
)


def test_pooled_rate_is_a_ratio_of_sums():
    estimate = pooled_rate([(1, 1), (0, 9)], resamples=200)
    assert estimate.value == pytest.approx(0.1)
    assert (estimate.numerator, estimate.denominator) == (1.0, 10.0)
    assert estimate.low <= estimate.value <= estimate.high


def test_pooled_rate_without_variance_has_a_point_interval():
    estimate = pooled_rate([(3, 4)] * 10, resamples=200)
    assert (estimate.value, estimate.low, estimate.high) == (0.75, 0.75, 0.75)


def test_pooled_rate_edge_cases():
    assert not pooled_rate([]).defined
    assert not pooled_rate([(0, 0), (0, 0)]).defined
    single = pooled_rate([(1, 2)])
    assert single.value == 0.5
    assert math.isnan(single.low)


def test_pooled_rate_is_reproducible_and_seeded():
    clusters = [(i % 3, 4) for i in range(30)]
    first = pooled_rate(clusters, resamples=500, seed=1)
    assert first == pooled_rate(clusters, resamples=500, seed=1)
    assert first != pooled_rate(clusters, resamples=500, seed=2)


def test_interval_narrows_with_more_scenarios():
    def width(n):
        estimate = pooled_rate([(i % 2, 1) for i in range(n)], resamples=2000)
        return estimate.high - estimate.low

    assert width(400) < width(20)


def test_paired_difference_uses_shared_scenarios_only():
    treated = {"a": (4, 4), "b": (2, 4), "only-treated": (0, 4)}
    control = {"a": (2, 4), "b": (2, 4), "only-control": (4, 4)}
    estimate = paired_difference(treated, control, resamples=500)
    assert estimate.value == pytest.approx(0.25)
    assert estimate.denominator == 2
    assert estimate.low <= estimate.value <= estimate.high


def test_paired_difference_of_identical_arms_is_exactly_zero():
    arm = {str(i): (i % 3, 4) for i in range(20)}
    estimate = paired_difference(arm, arm, resamples=500)
    assert (estimate.value, estimate.low, estimate.high) == (0.0, 0.0, 0.0)


def test_paired_difference_without_overlap_is_undefined():
    assert not paired_difference({"a": (1, 1)}, {"b": (1, 1)}).defined


@pytest.mark.parametrize(
    ("successes", "trials", "k", "expected"),
    [
        (5, 5, 5, 1.0),
        (4, 5, 5, 0.0),
        (4, 5, 1, 0.8),
        (4, 5, 2, 0.6),
        (0, 5, 1, 0.0),
    ],
)
def test_pass_hat_k(successes, trials, k, expected):
    assert pass_hat_k(successes, trials, k) == pytest.approx(expected)


def test_pass_hat_k_needs_at_least_k_trials():
    assert math.isnan(pass_hat_k(2, 2, 3))


def test_estimate_formatting():
    assert (
        Estimate(0.6875, 0.6, 0.75).format(percent=True) == "68.8% [60.0, 75.0]"
    )
    assert Estimate(0.375, 0.3, 0.45).format(digits=3) == "0.375 [0.300, 0.450]"
    assert Estimate(0.5).format(percent=True) == "50.0%"
    assert Estimate(math.nan).format() == "n/a"
