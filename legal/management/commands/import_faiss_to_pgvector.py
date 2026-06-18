"""
management/commands/import_faiss_to_pgvector.py
===============================================

One-time migration: reads all vectors from the existing FAISS flat index
and bulk-inserts them into the `legal_caseembedding` PostgreSQL table.

Usage:
    python manage.py import_faiss_to_pgvector

Options:
    --batch-size    Number of rows to insert per transaction (default: 500)
    --clear         Drop all existing rows before importing (default: False)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

log = logging.getLogger(__name__)

_FAISS_IDX = Path(__file__).resolve().parent.parent.parent / "retrieval" / "faiss.index"
_CASE_IDS  = Path(__file__).resolve().parent.parent.parent / "retrieval" / "case_ids.json"


class Command(BaseCommand):
    help = "Import InLegalBERT vectors from faiss.index into PostgreSQL/pgvector."

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size", type=int, default=500,
            help="Rows per bulk_create transaction (default: 500)"
        )
        parser.add_argument(
            "--clear", action="store_true", default=False,
            help="Delete all existing CaseEmbedding rows before importing."
        )

    def handle(self, *args, **options):
        batch_size = options["batch_size"]
        do_clear   = options["clear"]

        # ── Validate source files ─────────────────────────────────────
        if not _FAISS_IDX.exists():
            raise CommandError(
                f"FAISS index not found at {_FAISS_IDX}.\n"
                "Make sure the file exists before running this command."
            )
        if not _CASE_IDS.exists():
            raise CommandError(
                f"case_ids.json not found at {_CASE_IDS}.\n"
                "Make sure the file exists before running this command."
            )

        # ── Load FAISS ────────────────────────────────────────────────
        self.stdout.write("Loading FAISS index…")
        try:
            import faiss
        except ImportError:
            raise CommandError(
                "faiss-cpu is required for this one-time import.\n"
                "Run: pip install faiss-cpu\n"
                "You can uninstall it afterwards — pgvector takes over at runtime."
            )

        index = faiss.read_index(str(_FAISS_IDX))
        with open(_CASE_IDS) as f:
            case_ids: list[str] = json.load(f)

        total = index.ntotal
        if total != len(case_ids):
            raise CommandError(
                f"Mismatch: FAISS has {total} vectors but case_ids.json has "
                f"{len(case_ids)} entries."
            )

        self.stdout.write(f"FAISS index loaded: {total} vectors, dim={index.d}")

        # ── Optionally clear existing rows ────────────────────────────
        from legal.models import CaseEmbedding

        if do_clear:
            deleted, _ = CaseEmbedding.objects.all().delete()
            self.stdout.write(f"Cleared {deleted} existing rows.")

        existing = set(CaseEmbedding.objects.values_list("case_id", flat=True))
        self.stdout.write(f"Existing rows in DB: {len(existing)}. Skipping already-imported vectors.")

        # ── Bulk insert ───────────────────────────────────────────────
        self.stdout.write(f"Importing {total - len(existing)} new vectors (batch_size={batch_size})…")

        batch: list[CaseEmbedding] = []
        inserted = 0
        skipped  = 0

        for i in range(total):
            cid = case_ids[i]

            if cid in existing:
                skipped += 1
                continue

            vec = index.reconstruct(i).tolist()  # list[float], len=768
            batch.append(CaseEmbedding(case_id=cid, embedding=vec))

            if len(batch) >= batch_size:
                CaseEmbedding.objects.bulk_create(batch, ignore_conflicts=True)
                inserted += len(batch)
                batch = []
                self.stdout.write(
                    f"  … {inserted}/{total - len(existing)} inserted", ending="\r"
                )
                self.stdout.flush()

        # Final partial batch
        if batch:
            CaseEmbedding.objects.bulk_create(batch, ignore_conflicts=True)
            inserted += len(batch)

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Done. Inserted {inserted} vectors, skipped {skipped} "
                f"(already in DB). Total in table: "
                f"{CaseEmbedding.objects.count()}"
            )
        )

        # ── Optional: create IVFFlat index ───────────────────────────
        self.stdout.write(
            "\nTip: for faster ANN search, run this SQL once in psql:\n"
            "  CREATE INDEX ON legal_caseembedding\n"
            "  USING ivfflat (embedding vector_cosine_ops)\n"
            "  WITH (lists = 100);\n"
        )
