FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache

WORKDIR /app

RUN useradd -m -u 1000 appuser

COPY uv.lock pyproject.toml ./

RUN uv sync --frozen --no-install-project --no-dev

ENV PATH="/app/.venv/bin:$PATH"

RUN uv pip install pip
COPY scripts/download_models.py ./scripts/

RUN mkdir -p /app/model_cache && chown -R appuser:appuser /app/model_cache

RUN python scripts/download_models.py

COPY ./app ./app

RUN chown -R appuser:appuser /app

USER appuser

CMD ["fastapi", "run", "app/main.py", "--port", "80"]