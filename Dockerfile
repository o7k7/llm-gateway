# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache \
    PATH="/app/.venv/bin:$PATH"

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/model_cache && \
    chown -R appuser:appuser /app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

USER appuser

COPY --chown=appuser:appuser uv.lock pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace --no-dev && \
    uv pip install pip

COPY --chown=appuser:appuser ./app ./app

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uv", "run", "--no-project", "fastapi", "run", "app/main.py", "--port", "8000", "--host", "0.0.0.0"]
