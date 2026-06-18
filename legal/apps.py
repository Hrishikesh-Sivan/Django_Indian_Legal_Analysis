import logging
import threading

from django.apps import AppConfig
from django.conf import settings


class LegalConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "legal"

    def ready(self):
        # Pre-load the NLP pipeline once at startup
        from .nlp import get_nlp
        get_nlp(settings.SPACY_MODEL, settings.FALLBACK_SPACY_MODEL)

        # Warm up the lexical retriever in a background thread so the first
        # request to /retrieve doesn't pay the full BM25 load latency.
        def _warmup():
            try:
                from .retrieval.lexical import get_lexical_retriever
                get_lexical_retriever()._ensure_loaded()
                
                from .retrieval.semantic import get_semantic_retriever
                get_semantic_retriever()._try_init()
            except Exception as exc:
                logging.getLogger(__name__).warning("Retrieval warmup failed: %s", exc)

        threading.Thread(target=_warmup, daemon=True, name="retrieval-warmup").start()
        