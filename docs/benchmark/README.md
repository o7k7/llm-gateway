# Benchmark Protocol

This document defines the methodology used to produce the numbers in
[RESULTS.md](../../bench/results/RESULTS.md). It captures the hardware, software, request
patterns, measurement conventions, known limitations, and reproduction
steps. The goal is enough detail that another engineer could rerun the
benchmarks and arrive at comparable numbers, or independently audit any
specific claim from RESULTS.md.

## Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 SXM5 (single GPU) |
| Provider | RunPod Community Cloud |
| CPU | Provided by RunPod template (typical: AMD EPYC, 16 vCPU) |
| RAM | Provided by RunPod template (typical: 128 GB) |
| Container disk | 100 GB |
| Network | Localhost-only (all services co-located in single pod) |

All services ran inside one RunPod pod with localhost networking. No
network hops between gateway and backends. This means the bench
measures the **gateway pipeline + inference**, not network or
load-balancer overhead.

## Software stack

| Component | Version | Purpose |
|---|---|---|
| Python | 3.12 | Gateway and bench runtime |
| FastAPI | latest stable | Gateway HTTP framework |
| vLLM | latest stable | Inference backends |
| Redis Stack | 7.x | State, cache, rate limiting |
| HuggingFace Transformers | latest stable | Naive baseline |
| `uv` | 0.4+ | Dependency management |

OS: Ubuntu 22.04 (RunPod PyTorch 2.4 template).

## Models

Both targets serve **Qwen 2.5 7B** as the primary model, ensuring
fair comparison.

| Target | Model | Format | GPU memory |
|---|---|---|---|
| Gateway (small) | Qwen 2.5 7B Instruct AWQ | 4-bit AWQ | ~6 GB |
| Gateway (large) | Qwen 2.5 14B Instruct AWQ | 4-bit AWQ | ~14 GB |
| Baseline | Qwen 2.5 7B Instruct | fp16 | ~14 GB |

The gateway can route between the 7B and 14B models depending on
prompt complexity; the bench primarily exercises the 7B path to
match the baseline. The 14B path is verified at smoke-test level.

## Request patterns

The bench harness implements three load patterns:

### `single`
One request issued, response measured, then exit. Used for "no
contention" baseline measurement of the request pipeline.

### `steady`
Sustained N concurrent in-flight requests for a fixed duration.
As soon as a request completes, a new one is issued. Measures
sustained throughput and latency under continuous load.

### `burst`
N requests fired simultaneously at start, then no new requests.
All N complete (or fail) before the run ends. Measures behavior
under spike load and rate-limiter response.

## Concurrency levels tested

| Pattern | Concurrencies tested |
|---|---|
| single | 1 |
| steady | 1, 4, 8, 16 |
| burst | 8, 16, 32 |

Higher concurrencies for steady (e.g., c=32) and burst (e.g., c=64)
were not tested because c=16 sustained already exceeds the gateway's
configured rate limit, producing 92% rate-limit rejection. Higher
concurrency would not produce additional information beyond
"rate limiter still rejects."

The baseline was tested only at concurrencies ≤ 4. Higher concurrency
on the baseline produces deadlock-like queueing via `asyncio.Lock`
that does not yield meaningful measurements — see [Baseline
measurement limits](#baseline-measurement-limits) below.

## Measurement conventions

### Time-to-first-token (TTFT)

Defined as **wall-clock time from request submission to first byte
received by client**.

For streaming targets (gateway): this measures genuine first-token
latency — the SSE channel delivers the first chunk as soon as the
first token is generated.

For non-streaming targets (baseline): This measures the full
response time, because no bytes arrive at the client until the
entire JSON response is ready. This is documented as **TTFT
measurement asymmetry** and is the most important caveat in
RESULTS.md.

This asymmetry inflates the apparent TTFT speedup of streaming
targets. The honest interpretation:

- **TTFT speedup (60-67×)** = streaming delivery + inference combined
- **Total speedup (1.31× single, 4.91× at c=4)** = inference + parallelism

Both are real architectural advantages. RESULTS.md leads with the
c=4 Total speedup because it's not affected by streaming differences.

### Total latency

Wall-clock time from request submission to last byte received.
Equivalent across both targets — no asymmetry concern.

### Inter-token latency (ITL)

For streaming targets: time between consecutive SSE chunks during
generation. Measured per token.

For non-streaming targets: approximated as `total_latency /
completion_tokens`. This approximation is noted in the bench
output's `notes` field for any non-streaming run.

### Throughput (req/s and tokens/s)

Total successful requests (or output tokens) divided by total run
duration. Failed requests are excluded from the numerator. Run
duration includes warmup if any.

### Success rate

Fraction of requests that returned HTTP 200 with parseable response.
HTTP 429 (rate limited), 5xx (backend error), and client-side
timeouts (> bench timeout) all count as failures.

### Percentiles

`p50`, `p95`, `p99` computed from per-request samples. Reported only
when sample count ≥ 30 for the percentile to be meaningful — the
bench harness emits a `notes` field flagging low-sample runs.

For samples < 30, p99 is reported but should be read as p50≈p95≈p99
(not actual variance). This affects burst c=8/16/32 specifically.

## Baseline measurement limits

The naive baseline serves requests via FastAPI with an `asyncio.Lock`
around inference (`_state.generate_lock`). This means **all inference
runs strictly sequentially** regardless of how many concurrent client
requests arrive.

Implications for benchmarking:

| Concurrency | Baseline behavior | Measured |
|---|---|---|
| c=1 | Single inference at a time | ✅ Yes — clean numbers |
| c=2-4 | Queue depth grows, latency = depth × inference time | ✅ Yes — shows lock impact |
| c=8+ | Queue depth exceeds bench client timeout window | ❌ No — bench client times out |
| burst | First request runs, all others queue and time out | ❌ No — meaningless |

## Bench harness implementation

The harness lives in `bench/runner.py`. Key implementation choices:

1. **httpx.AsyncClient** for HTTP requests with connection pooling
2. **Per-request timing** captured with `time.perf_counter()` for
   sub-microsecond precision
3. **NDJSON output** to `bench/results/{target}_{pattern}_c{N}.ndjson`
   with one summary line per scenario
4. **Streaming detection** automatic from response Content-Type header
   (`text/event-stream` → SSE parsing; otherwise treated as JSON)
5. **Warmup** disabled by default — the first request in each scenario
   may have higher latency due to cold caches
6. **No retry on failure** — a 429 or timeout counts as one failed
   sample, not retried within the same scenario

## Reproducibility

Full reproduction requires a RunPod pod with H100 80GB and the
following sequence:

1. Provision RunPod pod (template: PyTorch 2.4, GPU: H100 80GB,
   container disk 100GB, expose port 8000)

2. Bootstrap services using the script in `deploy/runpod-bootstrap.sh`

3. Verify all services healthy:
   ```bash
   curl http://localhost:8000/health   # gateway
   curl http://localhost:8001/v1/models  # vllm-small
   curl http://localhost:8002/v1/models  # vllm-large
   curl http://localhost:8100/health   # baseline (round 2 only)
   ```

4. Run the gateway scenario matrix:
   ```bash
   make bench ARGS="--target gateway --pattern single --concurrency 1 --duration 30"
   make bench ARGS="--target gateway --pattern steady --concurrency 1 --duration 60"
   make bench ARGS="--target gateway --pattern steady --concurrency 4 --duration 60"
   make bench ARGS="--target gateway --pattern steady --concurrency 8 --duration 60"
   make bench ARGS="--target gateway --pattern steady --concurrency 16 --duration 60"
   make bench ARGS="--target gateway --pattern burst --concurrency 8"
   make bench ARGS="--target gateway --pattern burst --concurrency 16"
   make bench ARGS="--target gateway --pattern burst --concurrency 32"
   ```

5. Stop vLLM services to free GPU memory for baseline:
   ```bash
   tmux kill-session -t vllm-small
   tmux kill-session -t vllm-large
   ```

6. Start baseline and run the constrained matrix:
   ```bash
   # Start baseline (separate process, in baseline/ directory)
   make bench ARGS="--target baseline --pattern single --concurrency 1 --duration 30"
   make bench ARGS="--target baseline --pattern steady --concurrency 1 --duration 60"
   make bench ARGS="--target baseline --pattern steady --concurrency 4 --duration 30"
   ```

7. Combine and generate report:
   ```bash
   cat bench/results/gateway_*.ndjson bench/results/baseline_*.ndjson \
     > bench/results/combined.ndjson
   make bench-report
   ```

Expected total cloud cost: ~$5-7 (H100 at ~$2.50/hr for ~2-3 hours
including troubleshooting). Expected wall-clock time: ~90 minutes
of focused work.

## Known anomalies and bugs found during this bench session

Two bugs were discovered and fixed during the bench session that
produced the data in RESULTS.md:

### Ledger Lua script SHA mismatch

The gateway's `app/accounting/ledger.py` computed SHA-256 of the
Lua script for use as Redis's `EVALSHA` cache key. Redis's script
cache uses SHA-1, not SHA-256. Every request triggered `NoScriptError`
and fell back to `EVAL` (sending the full ~1KB script body each call).

Impact: ~0.3-1ms per-request overhead pre-fix.

Fix: Changed `hashlib.sha256` → `hashlib.sha1` in
`app/accounting/ledger.py:_sha()`. Added explicit `script_load()`
on app startup via `Ledger.initialize()` called from FastAPI lifespan.

The numbers in RESULTS.md are **post-fix**.

### Baseline running on CPU instead of GPU

The naive baseline's `from_pretrained()` call did not specify
`device_map`. HuggingFace Transformers defaults to CPU when no
device is specified. Initial bench runs measured CPU inference
(~0.75 tok/s), which would have made comparisons meaningless.

Fix: Added `device_map="auto"` parameter in
`baseline/server.py:_lifespan()`. Verified GPU placement via
`nvidia-smi` showing ~14GB allocated post-load (vs 825MB before).

The baseline numbers in RESULTS.md are **post-fix**, measured on
GPU as intended.

## Limitations and caveats

The numbers in RESULTS.md are honest measurements of the system as
configured, but the configuration has limits worth flagging:

1. **Single-region, single-replica.** No cross-region latency,
   no replica fan-out, no graceful failover between regions
   tested. Real production deployments would add latency.

2. **No background load.** The bench is the only client. Real
   production has telemetry sidecar load, log shippers, health
   checks, ingress traffic. These add small constant overhead.

3. **Localhost networking.** All services on same pod. Real
   K8s deployment adds ~0.5-2ms east-west per hop.

4. **Warm caches assumed.** The bench doesn't randomize prompts
   to defeat the gateway's semantic cache. Cache hit rates in
   the bench are higher than typical production traffic.

5. **No PII or jailbreak triggers in bench prompts.** The
   guardrail modules run but don't reject any test traffic.
   Production traffic would have some rejection rate.

6. **Tenant config not stress-tested.** The bench uses a single
   tenant. Multi-tenant fairness, tenant-config Redis hot keys,
   and tenant rate-limit interaction were verified in unit
   tests but not benchmarked.

7. **Bench harness itself adds overhead.** The Python httpx client
   adds ~0.5-1ms per request to all measurements. This affects
   both targets equally so doesn't bias the comparison, but
   means absolute TTFT numbers slightly overstate actual
   gateway latency.

## See also

- [RESULTS.md](../../bench/results/RESULTS.md) — full bench data and comparison tables
