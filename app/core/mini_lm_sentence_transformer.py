import logging

from sentence_transformers import SentenceTransformer

logger = logging.Logger("MiniLMSentenceTransformer")

model_instance: SentenceTransformer | None = None

model_instance_tensor_dim = 384

def get_model_instance_tensor_dim():
    global model_instance_tensor_dim
    return model_instance_tensor_dim

def get_model_instance() -> SentenceTransformer:
    global model_instance
    if model_instance is None:
        init_model_instance()

    return model_instance

def init_model_instance():
    global model_instance

    model_instance = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info('Initializing Sentence Transformer model...')
