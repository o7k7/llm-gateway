"""Naive HuggingFace FastAPI baseline server.

This package is intentionally a strawman — but an honest one.

Benchmarks the main gateway against this to quantify the value of
the production choices (vLLM + AWQ + continuous batching + streaming +
cache + guardrails + rate limiting).

See baseline/README.md for the rationale.
"""