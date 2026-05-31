"""Benchmark harness for comparing the gateway against the naive baseline.

The harness measures TTFT (time-to-first-token), ITL (inter-token
latency), end-to-end latency, throughput, and success rate across three
load patterns (single-stream, steady, burst) using a fixed prompt
repeated at each concurrency level.

Why a fixed prompt
------------------
Isolates concurrency effects from prompt-variance noise. With varied
prompts, p99 latency moves with the random distribution of long inputs;
with a fixed prompt, p99 movement reflects only the system's tail
behavior under load.
"""