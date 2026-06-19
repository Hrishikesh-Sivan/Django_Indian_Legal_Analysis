# Indian Legal Analysis

A Django (Python 3.11) web application that performs NLP analysis on Indian legal text — judgments, statute extracts, and case notes. It uses [spaCy](https://spacy.io) and the [Blackstone](https://github.com/ICLRandD/Blackstone) legal NLP model, augmented with custom regex patterns tuned for Indian citations (AIR, SCC, SCC OnLine, ILR) and statutes (IPC, CrPC, CPC, Evidence Act, Constitution Articles).

## Features

- **Web UI & JSON API**: Browse, search, and analyze legal judgments.
- **Hybrid Semantic Search**: Uses `rank-bm25` for lexical keyword search, and a PostgreSQL `pgvector` database storing 768-dimensional `InLegalBERT` embeddings for semantic context search.
- **NLP Extraction**:
  - Named entities (`CASE`, `COURT`, `JUDGE`, `PROVISION`).
  - Indian case citations (AIR, SCC, SCC OnLine, ILR).
  - Statutory references and Constitution Articles.
  - Sentence segmentation and POS summary.
- **Interactive Dashboards**: Citation networks, judgment trends, and statute analysis.
- **Offline Trend Pipeline**: Aggregates 75 years (1950–2025) of Supreme Court judgments.

---

## 🚀 Setup & Installation (Windows)

This project requires **Python 3.11** and **PostgreSQL 16**. 

### 1. Database Setup (PostgreSQL + pgvector)

Because semantic search runs on native cosine similarity, you must install PostgreSQL and the `pgvector` extension.

1. **Install PostgreSQL 16**: Download the [EnterpriseDB Windows Installer](https://www.postgresql.org/download/windows/). 
   - Use the default port `5432` (or `5433` if occupied) and set a password for the `postgres` user.
2. **Install pgvector**: Download the pre-built Windows binary for pgvector (or compile it). Place `vector.dll` into `C:\Program Files\PostgreSQL\16\lib` and the `.sql`/`.control` files into `C:\Program Files\PostgreSQL\16\share\extension`.
3. **Create the Database**: Open the SQL shell (`psql`) and run:
   ```sql
   CREATE DATABASE legal_db;
   \c legal_db
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

### 2. Python Environment

Open your PowerShell terminal in the project folder:

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies (Includes Torch, Transformers, psycopg2, pgvector)
pip install -r requirements.txt

# 3. Install the baseline spaCy model
python -m spacy download en_core_web_sm

# 4. (Optional) Install the Blackstone legal NLP model
#    This may fail on newer spaCy — the app falls back to en_core_web_sm automatically
pip install https://blackstone-model.s3-eu-west-1.amazonaws.com/en_blackstone_proto-0.0.1.tar.gz
```

> **Notes on Blackstone + Python 3.11:** Blackstone's published wheel was trained on an older spaCy. If the direct install fails on spaCy 3.7+, the app will automatically fall back to `en_core_web_sm` (general English NER) — the Indian-specific regex extractors for citations and statutes will still run perfectly.

### 3. Environment Variables

Create your `.env` file from the example:

```powershell
copy .env.example .env
```
Open `.env` and fill in your PostgreSQL credentials (`DB_PASSWORD`, `DB_PORT`, etc.).

### 4. Database Migrations & Data Import

```powershell
# 1. Create Django tables (including the CaseEmbedding vector table)
python manage.py migrate

# 2. Import the Semantic Vectors
# NOTE: This requires the original `faiss.index` and `case_ids.json` files to be present in `legal/retrieval/`.
# It reads all 26,688 InLegalBERT vectors and bulk-inserts them into PostgreSQL.
python manage.py import_faiss_to_pgvector

# 3. Create a superuser (optional, for accessing the Django Admin)
python manage.py createsuperuser
```

*(Highly Recommended)*: To make semantic search lightning fast, open `psql` and create an HNSW index:
```sql
CREATE INDEX ON legal_caseembedding USING hnsw (embedding vector_cosine_ops);
```

### 5. Run the Application

```powershell
# Start the Django development server
python manage.py runserver
```

Open **http://127.0.0.1:8000**. The first time you perform a search, the `InLegalBERT` model weights (~400 MB) will be downloaded from HuggingFace and cached automatically.

---

## 🛠️ API Endpoints

```bash
# Search for cases (Hybrid: Lexical + Semantic)
curl -X GET "http://127.0.0.1:8000/api/cases/?query=kesavananda&limit=10"

# Analyze raw text
curl -X POST http://127.0.0.1:8000/api/analyze/ \
  -H "Content-Type: application/json" \
  -d '{"text": "In AIR 1973 SC 1461 (Kesavananda Bharati), the Court examined Article 368."}'

# Get judgment details
curl -X GET "http://127.0.0.1:8000/api/judgment/<judgment_id>/"

# Health check
curl -X GET "http://127.0.0.1:8000/api/health"
```

---

## 📊 Data Ingestion Pipeline

The corpus ingestion is a three-step process. Steps 1–2 convert raw PDFs into analyst-friendly Parquet tables; the trend scripts (Step 3) aggregate those into charts and summary tables.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 0 (utility) | `app/data/supreme_court_judgments/script.py` | Raw PDFs with messy filenames | Sanitized PDF filenames |
| 1 | `app/data/convert_pdfs_to_text.py` | PDFs under `supreme_court_judgments/<year>/` | Per-year `combined_txt_file.txt` |
| 2 | `app/data/parse_combined_txt.py` | Per-year `combined_txt_file.txt` | `judgments.parquet` (26,688 rows) + `citations.parquet` (19,211 rows) |
| 3 | `app/data/trends/` (12 scripts) | Parquet tables from Step 2 | 27 trend parquets + 26 PNGs + `dashboard.html` |

### Regenerate parquets from text

```powershell
python app\data\parse_combined_txt.py
# Or for specific years:
python app\data\parse_combined_txt.py --years 2024 2025
```

---

## 📈 Trend Analysis Pipeline

This is an offline analytical layer over the 75-year Supreme Court corpus (26,688 judgments, 1950–2025). 

The pipeline processes Supreme Court judgments through multiple phases:
1. **Phase 1**: Coverage & volume EDA
2. **Phase 1.5**: Rule-based case categorization
3. **Phase 2a–2c**: Volume, length, and bench composition trends
4. **Phase 3**: Judge assignment patterns
5. **Phase 4a–4b**: Statute and section hotspot analysis
6. **Phase 4.5**: IPC/CrPC to BNS/BNSS evolution tracking
7. **Phase 5**: Citation network analysis
8. **Phase 6**: Keyword trend extraction
9. **Phase 7**: Report generation & dashboard HTML

### Regenerate everything

Run each phase in order:

```powershell
python app\data\trends\eda_overview.py          # Phase 1
python app\data\trends\categorize.py            # Phase 1.5
python app\data\trends\volume_trends.py         # Phase 2a
python app\data\trends\length_trends.py         # Phase 2b
python app\data\trends\bench_trends.py          # Phase 2c
python app\data\trends\judge_trends.py          # Phase 3
python app\data\trends\statute_trends.py        # Phase 4a
python app\data\trends\section_hotspots.py      # Phase 4b
python app\data\trends\ipc_crpc_evolution.py    # Phase 4.5
python app\data\trends\citation_network.py      # Phase 5
python app\data\trends\keyword_trends.py        # Phase 6
python app\data\trends\build_report.py          # Phase 7 (rollup)
```

Outputs land under:
- `app/data/processed/trends/` — Parquet tables
- `app/data/processed/charts/` — PNG charts + `dashboard.html`

---

## 📌 Key Artefacts

| File | Contents |
|------|----------|
| `trends_summary.parquet` | Per-year headline metrics (volume, length, bench, marquee sections) |
| `ipc_crpc_summary.parquet` | Per-section totals, OLS slope, constitution-bench share, BNS/BNSS successor |
| `citations_pagerank.parquet` | Top-50 most-influential judgments by corpus PageRank |
| `dashboard.html` | Interactive Plotly rollup (volume, length, bench, marquee sections) |

---

## ⚠️ Data Quality Notes

- **Volume Caps:** Indian Kanoon caps downloads at ~400 judgments/year from ~1970 onwards; absolute volumes reflect the scraped sample, not actual court output.
- **Headnotes:** Headnote coverage is dense pre-2000 and sparse after; keyword TF-IDF therefore limits to the 9,530 judgments with a substantive headnote.
- **Bench Sizes:** Bench-size values above 15 are parser errors (upstream "Bench:" line ate trailing text); `bench_trends.py` clips to ≤15 (largest legitimate SC bench is 13: Kesavananda Bharati, 1973).
- **Statute Disambiguation:** IPC/CrPC disambiguation prefers the curated `bns_bnss_mapping.yaml`; sections not in the mapping fall back to act-text presence (credited to both when both IPC and CrPC are named).

### Reference YAMLs
- **`amendments.yaml`**: Landmark amendments (498A insertion, CrPC 1973 commencement, Navtej, Nirbhaya, BNS/BNSS commencement, 44th Amendment) used as vertical-line overlays in timelines.
- **`bns_bnss_mapping.yaml`**: Curated ~60 IPC and ~40 CrPC section → BNS/BNSS successor mappings.

---

## 📂 Project Layout

```text
django_project/
├── manage.py                          # Django management script
├── compose.yaml                       # Docker Compose configuration
├── dockerfile                         # Docker image definition
├── requirements.txt                   # pip dependency file
├── Readme.md                          # Project documentation
│
├── django_project/                    # Django configuration
│   ├── settings.py                    # Django settings & PostgreSQL config
│   ├── urls.py                        # URL routing
│   └── wsgi.py / asgi.py              # Application entry points
│
├── legal/                             # Django app for legal analysis
│   ├── models.py                      # Database models (CaseEmbedding VectorField)
│   ├── views.py                       # Request handlers & API endpoints
│   ├── nlp.py                         # NLP analysis logic
│   │
│   ├── management/commands/           # Custom commands
│   │   └── import_faiss_to_pgvector.py # FAISS to PostgreSQL migration tool
│   │
│   ├── retrieval/                     # Document retrieval system
│   │   ├── semantic.py                # pgvector semantic search
│   │   ├── lexical.py                 # BM25 full-text search
│   │   └── fusion.py                  # Hybrid search fusion
│   │
│   └── data/                          # Data processing & storage
│       ├── processed/                 # Processed judgment data
│       └── trends/                    # Trend analysis scripts
│
└── cache/                             # Model cache directory
    └── hub/                           # Hugging Face model cache
```
