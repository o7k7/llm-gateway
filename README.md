# Local Development Prerequisites

- Use makefile to run redis-stack with `make up-redis-stack`
- Install pip via uv `uv pip install pip`
- Preload models only for once `python scripts/download_models.py` this will load models into `~/.cache/huggingface/hub`

# Architecture

```mermaid
graph TD
    Client[Client / Frontend] -->|POST /chat/completions/stream| Limiter{Rate Limiter}
    
    Limiter -- Limit Exceeded --> 429[429 Too Many Requests]
    Limiter -- Allowed --> API[FastAPI Gateway]
    
    subgraph Security Layer [1. Security Dependencies]
        API -->|1. Check| PII[PII Guard]
        API -->|2. Check| Jail[Jailbreak Detection]
    end
    
    subgraph Caching Layer [2. Caching Logic]
        API -->|Calculate Hash & Check| Redis[(Redis Cache)]
    end
    
    %% Fast Path: Cache Hit skips the LLM entirely
    Redis -- Hit JSON --> API
    
    %% Slow Path: Cache Miss goes to LLM
    Redis -.->|Cache Miss| API
    API -- If Miss --> LLM[LLM Provider]
    
    subgraph Observability
        API -.->|Async Telemetry\n Tokens/Latency| Langfuse[Langfuse]
    end
    
    LLM -->|Stream Response| API
    
    %% Response Paths
    API -->|Stream Response| Client
    API -->|Save to cache| Redis
```