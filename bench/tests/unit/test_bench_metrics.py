"""Tests for bench.metrics — percentile math and aggregation correctness."""
from __future__ import annotations

import math

import pytest

from bench.client import Outcome, RequestSample
from bench.metrics import _percentile, aggregate_samples


class TestPercentile:
    def test_single_value(self) -> None:
        assert _percentile([5.0], 50) == 5.0
        assert _percentile([5.0], 99) == 5.0

    def test_empty_returns_none(self) -> None:
        assert _percentile([], 50) is None

    def test_p50_of_sorted_evens(self) -> None:
        # [1, 2, 3, 4, 5] → median is 3
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0

    def test_p50_of_unsorted(self) -> None:
        # Same data, different order; median is still 3
        assert _percentile([5.0, 1.0, 4.0, 2.0, 3.0], 50) == 3.0

    def test_p99_close_to_max(self) -> None:
        values = [float(i) for i in range(1, 101)]  # 1..100
        # numpy.percentile([1..100], 99) = 99.01
        result = _percentile(values, 99)
        assert result is not None
        assert math.isclose(result, 99.01, abs_tol=1e-6)

    def test_p100_is_max(self) -> None:
        assert _percentile([1.0, 2.0, 3.0], 100) == 3.0

    def test_p0_is_min(self) -> None:
        assert _percentile([1.0, 2.0, 3.0], 0) == 1.0

    def test_invalid_percentile_raises(self) -> None:
        with pytest.raises(ValueError):
            _percentile([1.0, 2.0], 101)
        with pytest.raises(ValueError):
            _percentile([1.0, 2.0], -5)

    def test_linear_interpolation_between_points(self) -> None:
        # Two values [10, 20]; p50 = 15 (linear interp midpoint)
        assert _percentile([10.0, 20.0], 50) == 15.0

    def test_matches_numpy_for_known_input(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert _percentile(values, 25) == 3.25
        assert _percentile(values, 50) == 5.5
        assert _percentile(values, 75) == 7.75



def _success(
    *,
    ttft: float = 0.1,
    total: float = 1.0,
    completion_tokens: int = 100,
    itls: list[float] | None = None,
) -> RequestSample:
    return RequestSample(
        outcome=Outcome.SUCCESS,
        status_code=200,
        ttft_s=ttft,
        total_latency_s=total,
        completion_tokens=completion_tokens,
        prompt_tokens=20,
        inter_token_latencies_s=itls or [],
    )


def _failure(outcome: Outcome = Outcome.HTTP_ERROR) -> RequestSample:
    return RequestSample(
        outcome=outcome,
        status_code=500 if outcome is Outcome.HTTP_ERROR else None,
        ttft_s=None,
        total_latency_s=0.5,
        completion_tokens=None,
        prompt_tokens=None,
        error_message="boom",
    )


class TestAggregateBasics:
    def test_empty_samples(self) -> None:
        agg = aggregate_samples(
            [],
            target="t",
            pattern="single",
            concurrency=1,
            wall_clock_s=10.0,
        )
        assert agg.sample_count == 0
        assert agg.success_count == 0
        assert agg.success_rate == 0.0
        assert agg.requests_per_sec == 0.0
        assert agg.notes == ["No samples collected"]

    def test_all_successes_basic(self) -> None:
        samples = [_success(ttft=0.1, total=1.0) for _ in range(10)]
        agg = aggregate_samples(
            samples,
            target="t",
            pattern="single",
            concurrency=1,
            wall_clock_s=10.0,
        )
        assert agg.sample_count == 10
        assert agg.success_count == 10
        assert agg.failure_count == 0
        assert agg.success_rate == 1.0
        assert agg.ttft_p50 == pytest.approx(0.1)
        assert agg.total_p50 == pytest.approx(1.0)
        assert agg.requests_per_sec == pytest.approx(1.0)

    def test_mixed_success_and_failure(self) -> None:
        samples = (
            [_success() for _ in range(8)]
            + [_failure() for _ in range(2)]
        )
        agg = aggregate_samples(
            samples,
            target="t",
            pattern="single",
            concurrency=1,
            wall_clock_s=10.0,
        )
        assert agg.success_count == 8
        assert agg.failure_count == 2
        assert agg.success_rate == 0.8
        # Failures excluded from latency percentiles
        assert agg.ttft_p50 == pytest.approx(0.1)
        # RPS counts successes only
        assert agg.requests_per_sec == pytest.approx(0.8)

    def test_throughput_uses_wall_clock(self) -> None:
        samples = [_success(total=1.0) for _ in range(100)]
        agg = aggregate_samples(
            samples,
            target="t",
            pattern="steady",
            concurrency=10,
            wall_clock_s=10.0,
        )
        assert agg.requests_per_sec == pytest.approx(10.0)


class TestITLAggregation:
    def test_streaming_itls_flattened_across_requests(self) -> None:
        samples = [
            _success(itls=[0.01, 0.02, 0.03]),
            _success(itls=[0.05, 0.06]),
        ]
        agg = aggregate_samples(
            samples,
            target="t",
            pattern="single",
            concurrency=1,
            wall_clock_s=2.0,
        )
        assert agg.itl_p50 == pytest.approx(0.03)

    def test_non_streaming_itls_approximated(self) -> None:
        """Baseline (non-streaming) target: ITL approximated as
        total_latency / completion_tokens."""
        samples = [
            _success(total=2.0, completion_tokens=100, itls=[]),
            _success(total=4.0, completion_tokens=200, itls=[]),
        ]
        agg = aggregate_samples(
            samples,
            target="baseline",
            pattern="single",
            concurrency=1,
            wall_clock_s=6.0,
        )
        # Per-request ITL: 2.0/100 = 0.02, 4.0/200 = 0.02 → median 0.02
        assert agg.itl_p50 == pytest.approx(0.02)
        # Note must mention the approximation
        assert any(
            "approximated" in n.lower() for n in agg.notes
        )


class TestFailureCategorization:
    def test_failure_outcomes_recorded_in_notes(self) -> None:
        samples = (
            [_success() for _ in range(5)]
            + [_failure(Outcome.HTTP_ERROR), _failure(Outcome.TIMEOUT)]
        )
        agg = aggregate_samples(
            samples,
            target="t",
            pattern="single",
            concurrency=1,
            wall_clock_s=10.0,
        )
        # Notes should reference the outcomes
        notes_text = " ".join(agg.notes).lower()
        assert "http_error" in notes_text or "timeout" in notes_text


class TestTokenThroughput:
    def test_output_tokens_per_sec_computed(self) -> None:
        # 10 requests × 100 tokens each = 1000 tokens over 10s wall clock
        samples = [_success(completion_tokens=100) for _ in range(10)]
        agg = aggregate_samples(
            samples,
            target="t",
            pattern="single",
            concurrency=1,
            wall_clock_s=10.0,
        )
        assert agg.output_tokens_per_sec == pytest.approx(100.0)

    def test_missing_completion_tokens_handled(self) -> None:
        samples = [
            _success(completion_tokens=100),
            RequestSample(
                outcome=Outcome.SUCCESS,
                status_code=200,
                ttft_s=0.1,
                total_latency_s=1.0,
                completion_tokens=None,  # missing
                prompt_tokens=None,
            ),
        ]
        agg = aggregate_samples(
            samples,
            target="t",
            pattern="single",
            concurrency=1,
            wall_clock_s=2.0,
        )
        assert agg.output_tokens_per_sec == pytest.approx(50.0)
