## Benchmark Comparison: Gateway vs Naive HF Baseline

All times are wall-clock measurements from the bench client. Speedups are gateway-relative — **bold** means the gateway is faster.


### Pattern: `burst`

| Conc | Metric | Gateway | Baseline | Speedup |
|------|--------|---------|----------|---------|
| 8 | TTFT p50 | 278ms | — | — |
| 8 | TTFT p99 | 279ms | — | — |
| 8 | Total p50 | 3.22s | — | — |
| 8 | Total p99 | 3.22s | — | — |
| 16 | TTFT p50 | 476ms | — | — |
| 16 | TTFT p99 | 769ms | — | — |
| 16 | Total p50 | 3.60s | — | — |
| 16 | Total p99 | 3.65s | — | — |
| 32 | TTFT p50 | 1.27s | — | — |
| 32 | TTFT p99 | 1.27s | — | — |
| 32 | Total p50 | 4.25s | — | — |
| 32 | Total p99 | 4.25s | — | — |


**Throughput**

| Conc | Metric | Gateway | Baseline | Speedup |
|------|--------|---------|----------|---------|
| 8 | req/s | 0.6 | — | — |
| 8 | tokens/s | 149 | — | — |
| 16 | req/s | 0.9 | — | — |
| 16 | tokens/s | 229 | — | — |
| 32 | req/s | 1.0 | — | — |
| 32 | tokens/s | 246 | — | — |


**Success rate**

| Conc | Gateway | Baseline |
|------|---------|----------|
| 8 | 50.0% | — |
| 16 | 37.5% | — |
| 32 | 21.9% | — |


### Pattern: `single`

| Conc | Metric | Gateway | Baseline | Speedup |
|------|--------|---------|----------|---------|
| 1 | TTFT p50 | 65ms | 3.92s | **60.16×** |
| 1 | TTFT p99 | 82ms | 3.95s | **48.09×** |
| 1 | Total p50 | 3.00s | 3.92s | **1.31×** |
| 1 | Total p99 | 3.02s | 3.95s | **1.31×** |


**Throughput**

| Conc | Metric | Gateway | Baseline | Speedup |
|------|--------|---------|----------|---------|
| 1 | req/s | 0.3 | 0.2 | **1.35×** |
| 1 | tokens/s | 70.4 | 52.2 | **1.35×** |


**Success rate**

| Conc | Gateway | Baseline |
|------|---------|----------|
| 1 | 100.0% | 100.0% |


### Pattern: `steady`

| Conc | Metric | Gateway | Baseline | Speedup |
|------|--------|---------|----------|---------|
| 1 | TTFT p50 | 59ms | 3.92s | **66.69×** |
| 1 | TTFT p99 | 141ms | 3.95s | **28.07×** |
| 1 | Total p50 | 3.00s | 3.92s | **1.31×** |
| 1 | Total p99 | 3.06s | 3.95s | **1.29×** |
| 4 | TTFT p50 | 233ms | 15.66s | **67.20×** |
| 4 | TTFT p99 | 496ms | 15.71s | **31.65×** |
| 4 | Total p50 | 3.19s | 15.66s | **4.91×** |
| 4 | Total p99 | 4.19s | 15.71s | **3.75×** |
| 8 | TTFT p50 | 472ms | — | — |
| 8 | TTFT p99 | 1.70s | — | — |
| 8 | Total p50 | 4.48s | — | — |
| 8 | Total p99 | 6.35s | — | — |
| 16 | TTFT p50 | 1.09s | — | — |
| 16 | TTFT p99 | 2.69s | — | — |
| 16 | Total p50 | 5.69s | — | — |
| 16 | Total p99 | 7.29s | — | — |


**Throughput**

| Conc | Metric | Gateway | Baseline | Speedup |
|------|--------|---------|----------|---------|
| 1 | req/s | 0.3 | 0.2 | **1.30×** |
| 1 | tokens/s | 73.6 | 56.6 | **1.30×** |
| 4 | req/s | 1.0 | 0.2 | **5.66×** |
| 4 | tokens/s | 267 | 47.2 | **5.66×** |
| 8 | req/s | 1.6 | — | — |
| 8 | tokens/s | 405 | — | — |
| 16 | req/s | 0.9 | — | — |
| 16 | tokens/s | 231 | — | — |


**Success rate**

| Conc | Gateway | Baseline |
|------|---------|----------|
| 1 | 100.0% | 100.0% |
| 4 | 100.0% | 100.0% |
| 8 | 99.1% | — |
| 16 | 8.0% | — |

