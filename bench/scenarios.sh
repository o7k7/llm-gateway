#!/usr/bin/env bash
# Run the full benchmark matrix and emit NDJSON to bench/results/.
#
# Usage:
#   bench/scenarios.sh                 # all targets, all patterns
#   bench/scenarios.sh gateway         # gateway only
#   bench/scenarios.sh baseline        # baseline only

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p bench/results

TARGET="${1:-both}"

run_target() {
    local target="$1"
    local out_dir="bench/results"

    echo "=== Target: $target ===" >&2

    # Single-stream — isolates per-request latency
    echo "  -> single, c=1" >&2
    uv run --no-project python -m bench.runner \
        --target "$target" \
        --pattern single \
        --concurrency 1 \
        --duration 30 \
        >> "$out_dir/${target}_single_c1.ndjson"

    # Steady — production-representative sustained load
    for conc in 1 4 8 16; do
        echo "  -> steady, c=$conc" >&2
        uv run --no-project python -m bench.runner \
            --target "$target" \
            --pattern steady \
            --concurrency "$conc" \
            --duration 60 \
            >> "$out_dir/${target}_steady_c${conc}.ndjson"
    done

    # Burst — stress test queueing/cache-under-pressure
    for conc in 8 16 32; do
        echo "  -> burst, c=$conc" >&2
        uv run --no-project python -m bench.runner \
            --target "$target" \
            --pattern burst \
            --concurrency "$conc" \
            >> "$out_dir/${target}_burst_c${conc}.ndjson"
    done
}

case "$TARGET" in
    gateway)
        run_target gateway
        ;;
    baseline)
        run_target baseline
        ;;
    both)
        run_target gateway
        run_target baseline
        ;;
    *)
        echo "Usage: $0 [gateway|baseline|both]" >&2
        exit 1
        ;;
esac

cat bench/results/*.ndjson > bench/results/all.ndjson

echo "" >&2
echo "Done. Generate report with:" >&2
echo "  uv run --package llm-gateway-bench python -m bench.report --input bench/results/all.ndjson" >&2
