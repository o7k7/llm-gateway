import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("MiniLMSentenceTransformer")

MODEL_TENSOR_DIM = 384

@lru_cache(maxsize=1)
def get_model_instance() -> SentenceTransformer:
    logger.info("Initializing Sentence Transformer model...")
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_model_instance_tensor_dim() -> int:
    return MODEL_TENSOR_DIM