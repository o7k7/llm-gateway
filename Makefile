.PHONY: up up-redis-stack lint-format run-linter up-vllm-small down-vllm-small

up:
	docker compose up -d

up-redis-stack:
	docker compose up redis-stack -d

lint-format:
	uv run ruff format app tests

run-linter:
	uv run ruff check app

up-vllm-small:
	docker run --rm -d \
		--name vllm-small \
		--device /dev/kfd --device /dev/dri \
		--group-add video --security-opt seccomp=unconfined \
		--ipc=host \
		-v $$HOME/.cache/huggingface:/root/.cache/huggingface \
		-p 8001:8000 \
		rocm/vllm:latest \
		--model Qwen/Qwen2.5-7B-Instruct-AWQ \
		--quantization awq \
		--enable-prefix-caching \
		--max-model-len 8192 \
		--gpu-memory-utilization 0.90

down-vllm-small:
	docker stop vllm-small || true