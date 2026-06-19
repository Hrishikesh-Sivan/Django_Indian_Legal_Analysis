"""
management/commands/import_pkl_to_pgvector.py
=============================================

Imports InLegalBERT vectors + metadata from the pkl files directly
into the `legal_caseembedding` PostgreSQL table.

Each pkl file is a pandas DataFrame with columns:
    file        - case filename / case_id  (e.g. "Abdulla_Ahmed_vs_...1950_1.PDF")
    year        - judgment year            (e.g. "1950")
    clean_text  - cleaned judgment text
    embedding   - numpy ndarray of shape (768,)
    text, path  - ignored

Usage:
    python manage.py import_pkl_to_pgvector --pkl-dir "C:\\Users\\hrishikesh\\Desktop\\pkl files"

Options:
    --pkl-dir    Path to folder containing the .pkl files  (required)
    --batch-size Number of rows per bulk_create transaction (default: 200)
    --clear      Delete ALL existing CaseEmbedding rows first (default: False)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

log = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Import InLegalBERT vectors + metadata from pkl files into PostgreSQL/pgvector."

    def add_arguments(self, parser):
        parser.add_argument(
            "--pkl-dir", type=str, required=True,
            help="Path to the folder containing the .pkl files."
        )
        parser.add_argument(
            "--batch-size", type=int, default=200,
            help="Rows per bulk_create transaction (default: 200)."
        )
        parser.add_argument(
            "--clear", action="store_true", default=False,
            help="Delete ALL existing CaseEmbedding rows before importing."
        )

    def handle(self, *args, **options):
        pkl_dir    = Path(options["pkl_dir"])
        batch_size = options["batch_size"]
        do_clear   = options["clear"]

        # ── Validate directory ────────────────────────────────────────
        if not pkl_dir.exists() or not pkl_dir.is_dir():
            raise CommandError(f"Directory not found: {pkl_dir}")

        pkl_files = sorted(pkl_dir.glob("*.pkl"))
        if not pkl_files:
            raise CommandError(f"No .pkl files found in: {pkl_dir}")

        self.stdout.write(f"Found {len(pkl_files)} pkl file(s) in: {pkl_dir}")
        for p in pkl_files:
            self.stdout.write(f"  - {p.name}")

        # ── Optionally clear existing rows ────────────────────────────
        from legal.models import CaseEmbedding

        if do_clear:
            deleted, _ = CaseEmbedding.objects.all().delete()
            self.stdout.write(self.style.WARNING(f"Cleared {deleted} existing rows."))

        existing = set(CaseEmbedding.objects.values_list("case_id", flat=True))
        self.stdout.write(f"Rows already in DB: {len(existing)} (will be skipped).\n")

        # ── Process each pkl file ─────────────────────────────────────
        total_inserted = 0
        total_skipped  = 0
        total_errors   = 0

        for pkl_path in pkl_files:
            self.stdout.write(f"[{pkl_path.name}] Loading…")

            try:
                import pandas as pd
                with open(pkl_path, "rb") as f:
                    df = pickle.load(f)
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f"  Failed to load: {exc}"))
                total_errors += 1
                continue

            if not isinstance(df, pd.DataFrame):
                self.stdout.write(self.style.ERROR(f"  Unexpected type: {type(df)}. Skipping."))
                total_errors += 1
                continue

            # Validate required columns
            required = {"file", "embedding"}
            missing = required - set(df.columns)
            if missing:
                self.stdout.write(self.style.ERROR(f"  Missing columns {missing}. Skipping."))
                total_errors += 1
                continue

            self.stdout.write(f"  Loaded {len(df)} rows. Processing…")

            batch: list[CaseEmbedding] = []
            file_inserted = 0
            file_skipped  = 0

            for _, row in df.iterrows():
                case_id = str(row["file"]).strip()
                if not case_id:
                    file_skipped += 1
                    continue

                if case_id in existing:
                    file_skipped += 1
                    continue

                # Parse year — handle int, str, or missing
                raw_year = row.get("year", None)
                try:
                    year = int(raw_year) if raw_year is not None else None
                except (ValueError, TypeError):
                    year = None

                clean_text = str(row["clean_text"]).strip() if "clean_text" in row and row["clean_text"] else None

                # Embedding must be a 768-dim numpy array
                emb = row["embedding"]
                try:
                    emb_list = emb.tolist()
                    if len(emb_list) != 768:
                        log.warning("Skipping %s — embedding dim=%d (expected 768)", case_id, len(emb_list))
                        file_skipped += 1
                        continue
                except Exception:
                    log.warning("Skipping %s — could not convert embedding", case_id)
                    file_skipped += 1
                    continue

                batch.append(CaseEmbedding(
                    case_id    = case_id,
                    year       = year,
                    clean_text = clean_text,
                    embedding  = emb_list,
                ))
                existing.add(case_id)  # prevent duplicates across files

                if len(batch) >= batch_size:
                    CaseEmbedding.objects.bulk_create(batch, ignore_conflicts=True)
                    file_inserted += len(batch)
                    batch = []
                    self.stdout.write(
                        f"  … {file_inserted} inserted so far", ending="\r"
                    )
                    self.stdout.flush()

            # Final partial batch
            if batch:
                CaseEmbedding.objects.bulk_create(batch, ignore_conflicts=True)
                file_inserted += len(batch)

            self.stdout.write(
                f"  Done -> inserted={file_inserted}, skipped={file_skipped}"
            )
            total_inserted += file_inserted
            total_skipped  += file_skipped

        # ── Final summary ─────────────────────────────────────────────
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(
            f"Import complete!\n"
            f"  Total inserted  : {total_inserted}\n"
            f"  Total skipped   : {total_skipped} (already in DB or duplicate)\n"
            f"  Files with errors: {total_errors}\n"
            f"  Total rows in DB: {CaseEmbedding.objects.count()}"
        ))
        self.stdout.write(
            "\nTip: rebuild the HNSW index after a full re-import:\n"
            "  DROP INDEX IF EXISTS legal_caseembedding_embedding_idx;\n"
            "  CREATE INDEX ON legal_caseembedding USING hnsw (embedding vector_cosine_ops);\n"
        )
