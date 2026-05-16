"""Benchmark CLI: run one (target × pattern × concurrency) combination.

Usage examples
--------------
# Single-stream against the gateway
python -m bench.runner \\
    --target gateway \\
    --pattern single \\
    --concurrency 1 \\
    --duration 30

# Burst against the baseline
python -m bench.runner \\
    --target baseline \\
    --pattern burst \\
    --concurrency 16

# Steady load with custom URL (e.g. remote gateway)
python -m bench.runner \\
    --target-url http://staging.example.com:8000/v1/chat/completions \\
    --target-name staging \\
    --pattern steady \\
    --concurrency 8 \\
    --duration 60
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass

from bench.client import RequestSample, blocking_request, stream_request
from bench.load import LoadConfig, run_burst, run_single_stream, run_steady
from bench.metrics import aggregate_samples, aggregate_to_json_line
from bench.prompts import build_chat_request

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TargetSpec:
    name: str
    url: str
    streaming: bool

    model: str


# Default targets. URLs match the docker-compose.bench.yml setup.
TARGETS: dict[str, TargetSpec] = {
    "gateway": TargetSpec(
        name="gateway",
        url="http://localhost:8000/v1/chat/completions",
        streaming=True,
        model="auto",
    ),
    "baseline": TargetSpec(
        name="baseline",
        url="http://localhost:8100/v1/chat/completions",
        streaming=False,
        model="baseline",
    ),
}


PATTERNS = {
    "single": run_single_stream,
    "steady": run_steady,
    "burst": run_burst,
}


async def run_one(
    *,
    target: TargetSpec,
    pattern_name: str,
    concurrency: int,
    duration_s: float,
    warmup_s: float,
    request_timeout_s: float,
) -> None:
    """Run one (target × pattern × concurrency) combination and emit
    one NDJSON line to stdout."""

    payload = build_chat_request(
        model=target.model, stream=target.streaming
    )
    request_fn = (
        stream_request if target.streaming else blocking_request
    )

    config = LoadConfig(
        target_url=target.url,
        payload=payload,
        request_fn=request_fn,
        concurrency=concurrency,
        duration_s=duration_s,
        warmup_s=warmup_s,
        request_timeout_s=request_timeout_s,
    )

    pattern_fn = PATTERNS[pattern_name]

    samples: list[RequestSample] = []
    wall_start = time.monotonic()

    async for sample in pattern_fn(config):
        samples.append(sample)
        if len(samples) % 20 == 0:
            print(
                f"  ... {len(samples)} samples "
                f"({sample.outcome.value})",
                file=sys.stderr,
                flush=True,
            )

    wall_clock_s = time.monotonic() - wall_start

    aggregate = aggregate_samples(
        samples,
        target=target.name,
        pattern=pattern_name,
        concurrency=concurrency,
        wall_clock_s=wall_clock_s,
    )

    # Emit the aggregate as one NDJSON line on stdout
    print(aggregate_to_json_line(aggregate), flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Benchmark a chat-completion endpoint",
    )
    parser.add_argument(
        "--target",
        choices=list(TARGETS.keys()),
        help="Predefined target (gateway or baseline)",
    )
    parser.add_argument(
        "--target-url",
        help="Custom target URL (overrides --target)",
    )
    parser.add_argument(
        "--target-name",
        default="custom",
        help="Name for custom target in the JSON output",
    )
    parser.add_argument(
        "--target-streaming",
        action="store_true",
        help="Treat custom target as a streaming endpoint",
    )
    parser.add_argument(
        "--target-model",
        default="auto",
        help="Model name to send in the request body",
    )
    parser.add_argument(
        "--pattern",
        choices=list(PATTERNS.keys()),
        required=True,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Measurement duration in seconds (single/steady only)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
    )
    return parser


def _resolve_target(args: argparse.Namespace) -> TargetSpec:
    if args.target_url:
        return TargetSpec(
            name=args.target_name,
            url=args.target_url,
            streaming=args.target_streaming,
            model=args.target_model,
        )
    if args.target:
        return TARGETS[args.target]
    raise SystemExit("Either --target or --target-url is required")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    target = _resolve_target(args)
    print(
        f"# Bench: target={target.name} pattern={args.pattern} "
        f"concurrency={args.concurrency} duration={args.duration}s",
        file=sys.stderr,
    )

    asyncio.run(
        run_one(
            target=target,
            pattern_name=args.pattern,
            concurrency=args.concurrency,
            duration_s=args.duration,
            warmup_s=args.warmup,
            request_timeout_s=args.request_timeout,
        )
    )


if __name__ == "__main__":
    main()
