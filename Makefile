.PHONY: up up-redis-stack lint-format run-linter up-vllm-small down-vllm-small install-spacy-model

up:
	docker compose up -d

up-redis-stack:
	docker compose up redis-stack -d

lint-format:
	uv run ruff format app tests

run-linter:
	uv run ruff check app

install-spacy-model:
	uv run python -m spacy download en_core_web_sm

up-vllm-small:
	docker run --rm -it \
    --name vllm-test \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    -e HIP_VISIBLE_DEVICES=0 \
    --device /dev/kfd --device /dev/dri \
    --group-add video --security-opt seccomp=unconfined \
    --ipc=host \
    -p 8001:8000 \
    rocm/vllm:latest \
    python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --gpu-memory-utilization 0.50 \
    --max-model-len 2048 \
    --enforce-eager \
    --disable-log-stats \
    --kv-cache-dtype auto \
    --distributed-executor-backend mp

down-vllm-small:
	docker stop vllm-small || true