FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache

WORKDIR /app

COPY uv.lock pyproject.toml ./

RUN uv sync --frozen --no-install-project --no-dev

ENV PATH="/app/.venv/bin:$PATH"

COPY scripts/download_models.py ./scripts/

RUN python scripts/download_models.py

COPY ./app ./app

RUN chown -R appuser:appuser /app

USER appuser

CMD ["fastapi", "run", "app/main.py", "--port", "80"]