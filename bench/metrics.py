"""Metric aggregation: percentiles, throughput, success rate.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from bench.client import Outcome, RequestSample


@dataclass(slots=True)
class Aggregate:
    """Aggregated metrics for one bench run."""

    target: str
    pattern: str
    concurrency: int

    sample_count: int
    success_count: int
    failure_count: int
    success_rate: float

    # Latency percentiles (seconds)
    ttft_p50: float | None
    ttft_p95: float | None
    ttft_p99: float | None
    total_p50: float | None
    total_p95: float | None
    total_p99: float | None

    # Inter-token latency percentiles (seconds)
    itl_p50: float | None
    itl_p95: float | None
    itl_p99: float | None

    # Throughput (whole-run averages)
    requests_per_sec: float
    output_tokens_per_sec: float
    """Sum of completion_tokens / total wall-clock seconds."""

    completed_fully: int

    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def aggregate_samples(
    samples: list[RequestSample],
    *,
    target: str,
    pattern: str,
    concurrency: int,
    wall_clock_s: float,
) -> Aggregate:
    """Compute the aggregate from a list of samples.
    """
    succ = [s for s in samples if s.outcome is Outcome.SUCCESS]
    fail = [s for s in samples if s.outcome is not Outcome.SUCCESS]

    notes: list[str] = []

    if not samples:
        return Aggregate(
            target=target,
            pattern=pattern,
            concurrency=concurrency,
            sample_count=0,
            success_count=0,
            failure_count=0,
            success_rate=0.0,
            ttft_p50=None,
            ttft_p95=None,
            ttft_p99=None,
            total_p50=None,
            total_p95=None,
            total_p99=None,
            itl_p50=None,
            itl_p95=None,
            itl_p99=None,
            requests_per_sec=0.0,
            output_tokens_per_sec=0.0,
            completed_fully=0,
            notes=["No samples collected"],
        )

    # Latency percentiles
    ttft_values = [s.ttft_s for s in succ if s.ttft_s is not None]
    total_values = [s.total_latency_s for s in succ]

    # ITL: flatten all per-chunk gaps across all streaming requests.
    itl_values: list[float] = []
    for s in succ:
        itl_values.extend(s.inter_token_latencies_s)

    if not itl_values and succ:
        approximations = [
            s.total_latency_s / s.completion_tokens
            for s in succ
            if s.completion_tokens and s.completion_tokens > 0
        ]
        itl_values = approximations
        if approximations:
            notes.append(
                "ITL approximated as total_latency / completion_tokens "
                "(non-streaming target)"
            )

    # Throughput
    rps = len(succ) / wall_clock_s if wall_clock_s > 0 else 0.0
    total_completion_tokens = sum(
        s.completion_tokens for s in succ if s.completion_tokens
    )
    tps = (
        total_completion_tokens / wall_clock_s if wall_clock_s > 0 else 0.0
    )

    if fail:
        outcomes = {o: 0 for o in Outcome if o is not Outcome.SUCCESS}
        for s in fail:
            outcomes[s.outcome] = outcomes.get(s.outcome, 0) + 1
        notes.append(f"Failures: {outcomes}")

    return Aggregate(
        target=target,
        pattern=pattern,
        concurrency=concurrency,
        sample_count=len(samples),
        success_count=len(succ),
        failure_count=len(fail),
        success_rate=len(succ) / len(samples),
        ttft_p50=_percentile(ttft_values, 50),
        ttft_p95=_percentile(ttft_values, 95),
        ttft_p99=_percentile(ttft_values, 99),
        total_p50=_percentile(total_values, 50),
        total_p95=_percentile(total_values, 95),
        total_p99=_percentile(total_values, 99),
        itl_p50=_percentile(itl_values, 50),
        itl_p95=_percentile(itl_values, 95),
        itl_p99=_percentile(itl_values, 99),
        requests_per_sec=rps,
        output_tokens_per_sec=tps,
        completed_fully=len(succ),
        notes=notes,
    )


def _percentile(values: list[float], pct: float) -> float | None:
    """Compute the pct-th percentile using linear interpolation"""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    if not (0 <= pct <= 100):
        raise ValueError(f"percentile must be in [0, 100], got {pct}")

    sorted_vals = sorted(values)
    rank = (pct / 100) * (len(sorted_vals) - 1)
    lower_idx = math.floor(rank)
    upper_idx = math.ceil(rank)

    if lower_idx == upper_idx:
        return sorted_vals[lower_idx]

    weight = rank - lower_idx
    return (
            sorted_vals[lower_idx] * (1 - weight)
            + sorted_vals[upper_idx] * weight
    )


def aggregate_to_json_line(agg: Aggregate) -> str:
    return json.dumps(asdict(agg), separators=(",", ":"))
