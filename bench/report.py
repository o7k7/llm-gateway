"""Comparison report generator — NDJSON in, Markdown out.

Reads a stream of Aggregate JSON lines (one per bench run) and produces
a comparison table with relative speedups. Designed for piping:

python -m bench.report --input bench/results/all.ndjson
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _Run:
    """One row of the NDJSON input."""

    target: str
    pattern: str
    concurrency: int
    success_rate: float
    ttft_p50: float | None
    ttft_p95: float | None
    ttft_p99: float | None
    total_p50: float | None
    total_p95: float | None
    total_p99: float | None
    itl_p50: float | None
    itl_p99: float | None
    requests_per_sec: float
    output_tokens_per_sec: float
    notes: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "_Run":
        return cls(
            target=d["target"],
            pattern=d["pattern"],
            concurrency=d["concurrency"],
            success_rate=d["success_rate"],
            ttft_p50=d.get("ttft_p50"),
            ttft_p95=d.get("ttft_p95"),
            ttft_p99=d.get("ttft_p99"),
            total_p50=d.get("total_p50"),
            total_p95=d.get("total_p95"),
            total_p99=d.get("total_p99"),
            itl_p50=d.get("itl_p50"),
            itl_p99=d.get("itl_p99"),
            requests_per_sec=d["requests_per_sec"],
            output_tokens_per_sec=d["output_tokens_per_sec"],
            notes=d.get("notes", []),
        )


def parse_ndjson(text: str) -> list[_Run]:
    """Parse NDJSON text into a list of runs.
    """
    runs: list[_Run] = []
    for line_num, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"line {line_num}: invalid JSON: {e}") from e
        runs.append(_Run.from_dict(obj))
    return runs




def pair_runs(
    runs: list[_Run],
    *,
    target_a: str = "gateway",
    target_b: str = "baseline",
) -> list[tuple[_Run, _Run | None]]:
    """Pair (target_a, target_b) runs sharing pattern + concurrency.

    Returns a list of (a, b) pairs where b may be None if no matching
    baseline run exists. Pairs are sorted by (pattern, concurrency)
    for deterministic output.
    """
    by_key: dict[tuple[str, str, int], _Run] = {}
    for r in runs:
        by_key[(r.target, r.pattern, r.concurrency)] = r

    a_runs = sorted(
        (r for r in runs if r.target == target_a),
        key=lambda r: (r.pattern, r.concurrency),
    )

    pairs: list[tuple[_Run, _Run | None]] = []
    for a in a_runs:
        b = by_key.get((target_b, a.pattern, a.concurrency))
        pairs.append((a, b))
    return pairs


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def _fmt_seconds(v: float | None) -> str:
    if v is None:
        return "—"
    if v < 0.01:
        return f"{v * 1000:.1f}ms"
    if v < 1.0:
        return f"{v * 1000:.0f}ms"
    return f"{v:.2f}s"


def _fmt_speedup(faster: float | None, slower: float | None) -> str:
    """Compute speedup of `slower / faster` (since lower latency is better).
    """
    if faster is None or slower is None or faster <= 0:
        return "—"
    ratio = slower / faster
    if ratio >= 1.0:
        return f"**{ratio:.2f}×**"
    return f"{ratio:.2f}×"


def _fmt_throughput(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


def render_comparison(
    pairs: list[tuple[_Run, _Run | None]],
) -> str:
    """Render the pairs as a Markdown report."""
    if not pairs:
        return "_No runs to compare._\n"

    lines: list[str] = []
    lines.append("## Benchmark Comparison: Gateway vs Naive HF Baseline\n")
    lines.append(
        "All times are wall-clock measurements from the bench client. "
        "Speedups are gateway-relative — **bold** means the gateway "
        "is faster.\n"
    )

    # Group by pattern for readability
    by_pattern: dict[str, list[tuple[_Run, _Run | None]]] = defaultdict(list)
    for pair in pairs:
        by_pattern[pair[0].pattern].append(pair)

    for pattern in sorted(by_pattern.keys()):
        lines.append(f"\n### Pattern: `{pattern}`\n")
        lines.append(_render_pattern_table(by_pattern[pattern]))
        lines.append(_render_throughput_table(by_pattern[pattern]))
        lines.append(_render_success_table(by_pattern[pattern]))

    lines.append("\n---\n")
    lines.append(
        "Notes from the bench harness are preserved in "
        "[`RAW_DATA.md`](RAW_DATA.md). Methodology is defended in "
        "[`PROTOCOL.md`](PROTOCOL.md).\n"
    )

    return "\n".join(lines)


def _render_pattern_table(
    pairs: list[tuple[_Run, _Run | None]],
) -> str:
    out: list[str] = []
    out.append(
        "| Conc | Metric | Gateway | Baseline | Speedup |"
    )
    out.append(
        "|------|--------|---------|----------|---------|"
    )
    for a, b in pairs:
        # TTFT row
        out.append(
            f"| {a.concurrency} | TTFT p50 | "
            f"{_fmt_seconds(a.ttft_p50)} | "
            f"{_fmt_seconds(b.ttft_p50) if b else '—'} | "
            f"{_fmt_speedup(a.ttft_p50, b.ttft_p50) if b else '—'} |"
        )
        out.append(
            f"| {a.concurrency} | TTFT p99 | "
            f"{_fmt_seconds(a.ttft_p99)} | "
            f"{_fmt_seconds(b.ttft_p99) if b else '—'} | "
            f"{_fmt_speedup(a.ttft_p99, b.ttft_p99) if b else '—'} |"
        )
        out.append(
            f"| {a.concurrency} | Total p50 | "
            f"{_fmt_seconds(a.total_p50)} | "
            f"{_fmt_seconds(b.total_p50) if b else '—'} | "
            f"{_fmt_speedup(a.total_p50, b.total_p50) if b else '—'} |"
        )
        out.append(
            f"| {a.concurrency} | Total p99 | "
            f"{_fmt_seconds(a.total_p99)} | "
            f"{_fmt_seconds(b.total_p99) if b else '—'} | "
            f"{_fmt_speedup(a.total_p99, b.total_p99) if b else '—'} |"
        )
    return "\n".join(out) + "\n"


def _render_throughput_table(
    pairs: list[tuple[_Run, _Run | None]],
) -> str:
    out: list[str] = []
    out.append("\n**Throughput**\n")
    out.append(
        "| Conc | Metric | Gateway | Baseline | Speedup |"
    )
    out.append(
        "|------|--------|---------|----------|---------|"
    )
    for a, b in pairs:
        out.append(
            f"| {a.concurrency} | req/s | "
            f"{_fmt_throughput(a.requests_per_sec)} | "
            f"{_fmt_throughput(b.requests_per_sec) if b else '—'} | "
            f"{_fmt_speedup(b.requests_per_sec, a.requests_per_sec) if b else '—'} |"
        )
        out.append(
            f"| {a.concurrency} | tokens/s | "
            f"{_fmt_throughput(a.output_tokens_per_sec)} | "
            f"{_fmt_throughput(b.output_tokens_per_sec) if b else '—'} | "
            f"{_fmt_speedup(b.output_tokens_per_sec, a.output_tokens_per_sec) if b else '—'} |"
        )
    return "\n".join(out) + "\n"


def _render_success_table(
    pairs: list[tuple[_Run, _Run | None]],
) -> str:
    out: list[str] = []
    out.append("\n**Success rate**\n")
    out.append(
        "| Conc | Gateway | Baseline |"
    )
    out.append(
        "|------|---------|----------|"
    )
    for a, b in pairs:
        out.append(
            f"| {a.concurrency} | "
            f"{a.success_rate * 100:.1f}% | "
            f"{(b.success_rate * 100):.1f}% |"
            if b
            else f"| {a.concurrency} | "
                 f"{a.success_rate * 100:.1f}% | — |"
        )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bench-report",
        description="Render benchmark NDJSON as a Markdown comparison",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="NDJSON input file. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--gateway-name",
        default="gateway",
        help="Target name to compare as 'fast' system",
    )
    parser.add_argument(
        "--baseline-name",
        default="baseline",
        help="Target name to compare as 'slow' system",
    )
    args = parser.parse_args(argv)

    text = (
        args.input.read_text() if args.input else sys.stdin.read()
    )

    runs = parse_ndjson(text)
    pairs = pair_runs(
        runs,
        target_a=args.gateway_name,
        target_b=args.baseline_name,
    )
    print(render_comparison(pairs))


if __name__ == "__main__":
    main()
