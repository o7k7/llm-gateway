import logging
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

logger = logging.getLogger("AnalyzerEngine")


@lru_cache(maxsize=1)
def get_analyzer() -> AnalyzerEngine:
    """
    Initializes and returns the Presidio AnalyzerEngine as a singleton.
    If initialization fails, the error is logged and the exception is re-raised.
    """
    try:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "language": "en",
                "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
            }
        )
        nlp_engine = provider.create_engine()
        engine = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

        logger.info("Analyzer initialized")
        return engine

    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        raise  # Re-raise so the error isn't silently swallowed
