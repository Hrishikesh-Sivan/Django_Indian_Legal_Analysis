from django.db import models
from pgvector.django import VectorField


class CaseEmbedding(models.Model):
    """
    Stores the 768-dimensional InLegalBERT embedding for each judgment,
    along with key metadata sourced directly from the pkl files.

    Populated once by the management command:
        python manage.py import_pkl_to_pgvector --pkl-dir "C:\\path\\to\\pkl files"

    Used by SemanticRetriever (retrieval/semantic.py) to perform
    cosine-similarity ANN search via pgvector's <=> operator.
    """
    case_id    = models.CharField(max_length=512, unique=True, db_index=True)
    year       = models.IntegerField(null=True, blank=True, db_index=True)
    clean_text = models.TextField(null=True, blank=True)
    embedding  = VectorField(dimensions=768)

    class Meta:
        app_label = "legal"

    def __str__(self):
        return f"{self.case_id} ({self.year})"
