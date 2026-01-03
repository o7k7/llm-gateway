import logging
import os

from app.config import config

logger = logging.getLogger("LLMEnvironment")

def configure_llm_environment():
    """
    Sets the correct API Key env var.
    """
    try:
        provider = config.LLM_PROVIDER.lower()
        model_name = config.LLM_MODEL.lower()
    except ValueError:
        raise Exception("Provide a valid LLM model name")
    env_var_mapping = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
    }

    target_env_var = env_var_mapping.get(provider)

    if target_env_var:
        os.environ[target_env_var] = config.LLM_API_KEY
        logger.info(f"Set {target_env_var} for provider '{provider}'")

    return provider, model_name