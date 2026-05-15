FROM ubuntu:22.04

# Prevent interactive configuration prompts during package updates
# Prevent interactive configuration prompts during package updates
ENV DEBIAN_FRONTEND=noninteractive

# Install system prerequisites, INCLUDING gnupg for repository key management
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    curl \
    git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install uv via the official pre-compiled binary method for optimal container speeds
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy the environment file into the workspace root
COPY baseline/pyproject.toml /app/pyproject.toml

# Install dependencies into the global system python path using the tool choices from your project configuration
RUN uv pip install --system --no-cache -r pyproject.toml

# Set up caching directory for Hugging Face
ENV HF_HOME=/app/.hf_cache

# Warm up the cache by pre-downloading the Qwen-7B weights
RUN python3.11 -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', low_cpu_mem_usage=True)" \
    || echo "Pre-download failed; will download at runtime"

# Copy the application source code
COPY baseline/ /app/baseline/

ENV PYTHONUNBUFFERED=1
EXPOSE 8100

# Execute using system python, calling fastapi directly as resolved by the installation layer
CMD ["fastapi", "run", "baseline/server.py", "--host", "0.0.0.0", "--port", "8100"]