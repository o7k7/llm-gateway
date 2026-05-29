FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG BASELINE_MODEL=Qwen/Qwen2.5-7B-Instruct
ENV BASELINE_MODEL=${BASELINE_MODEL}
ENV UV_PYTHON=python3.12
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg \
        curl \
        git \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/.hf_cache && \
    chown -R appuser:appuser /app

WORKDIR /app

USER appuser

COPY --chown=appuser:appuser baseline/pyproject.toml baseline/uv.lock /app/
RUN uv sync --frozen --no-install-project --no-dev
ENV PATH="/app/.venv/bin:$PATH"

ENV HF_HOME=/app/.hf_cache
ENV PYTHONUNBUFFERED=1
ENV HF_XET_HIGH_PERFORMANCE=1

RUN python -c "\
import os; \
model = os.environ['BASELINE_MODEL']; \
print(f'Pre-downloading {model}...'); \
from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained(model); \
AutoModelForCausalLM.from_pretrained(model, low_cpu_mem_usage=True); \
print('Pre-download complete.')"

ENV HF_TOKEN=""

COPY --chown=appuser:appuser baseline/ /app/baseline/

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8100/health || exit 1

EXPOSE 8100

CMD ["fastapi", "run", "baseline/server.py", "--host", "0.0.0.0", "--port", "8100"]
