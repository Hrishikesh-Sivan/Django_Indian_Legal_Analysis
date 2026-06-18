from django.db import models
from pgvector.django import VectorField


class CaseEmbedding(models.Model):
    """
    Stores the 768-dimensional InLegalBERT embedding for each judgment.

    Populated once by the management command:
        python manage.py import_faiss_to_pgvector

    Used by SemanticRetriever (retrieval/semantic.py) to perform
    cosine-similarity ANN search via pgvector's <=> operator.
    """
    case_id   = models.CharField(max_length=512, unique=True, db_index=True)
    embedding = VectorField(dimensions=768)

    class Meta:
        app_label = "legal"

    def __str__(self):
        return self.case_id
