import logging

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

logger = logging.getLogger("AnalyzerEngine")

analyzer: AnalyzerEngine | None = None

def init_analyzer():
    global analyzer

    provider = NlpEngineProvider(nlp_configuration={
        'nlp_engine_name' : "spacy",
        "language": "en",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
    })
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    print("Analyzer initialized")

def get_analyzer() -> AnalyzerEngine:
    global analyzer
    if analyzer is None:
        try:
            init_analyzer()
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {e}")

    return analyzer