import json
import logging
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .nlp import analyze_text, get_nlp

log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
TRENDS_DIR = PROCESSED_DIR / "trends"


def _read_parquet(name: str) -> pd.DataFrame | None:
    p = TRENDS_DIR / name
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _load_dashboard_data() -> dict:
    """Load all trend data needed for the dashboard."""
    data: dict = {}

    summary = _read_parquet("trends_summary.parquet")
    if summary is not None:
        summary = summary.fillna(0)
        data["summary"] = summary.to_dict(orient="records")
        data["total_judgments"] = int(summary["judgments"].sum())
        data["year_range"] = [int(summary["year"].min()), int(summary["year"].max())]
        latest = summary.iloc[-1]
        data["latest_year"] = int(latest["year"])
        data["latest_judgments"] = int(latest["judgments"])
    else:
        data["summary"] = []
        data["total_judgments"] = 0

    cats = _read_parquet("case_categories.parquet")
    if cats is not None:
        dist = cats["case_category"].value_counts().to_dict()
        data["category_distribution"] = dist
    else:
        data["category_distribution"] = {}

    vol_cat = _read_parquet("volume_by_year_category.parquet")
    if vol_cat is not None:
        data["volume_by_category"] = vol_cat.fillna(0).to_dict(orient="records")
    else:
        data["volume_by_category"] = []

    judges = _read_parquet("judges_top.parquet")
    if judges is not None:
        data["top_judges"] = judges.head(10).to_dict(orient="records")
    else:
        data["top_judges"] = []

    pagerank = _read_parquet("citations_pagerank.parquet")
    if pagerank is not None:
        top = pagerank.head(10).copy()
        top["case_id"] = top["case_id"].str.replace("_", " ").str[:60]
        data["top_cases"] = top.to_dict(orient="records")
    else:
        data["top_cases"] = []

    statutes = _read_parquet("statute_top.parquet")
    if statutes is not None:
        data["top_statutes"] = statutes.head(10).fillna(0).to_dict(orient="records")
    else:
        data["top_statutes"] = []

    bench = _read_parquet("bench_by_year.parquet")
    if bench is not None:
        data["bench_trend"] = bench.fillna(0).to_dict(orient="records")

    return data


# ── UI Routes ──────────────────────

def index(request):
    data = _load_dashboard_data()
    for judge in data.get("top_judges", []):
        judge["const_bench_pct"] = round(
            judge.get("const_bench_share", 0) * 100, 1
        )
    data["distinct_judges_estimate"] = len(data.get("top_judges", [])) * 33
    return render(request, "dashboard.html", {
        "data": data,
        "data_json": json.dumps(data, default=str),
    })


def analyze(request):
    if request.method == "POST":
        text = (request.POST.get("text") or "").strip()
        result = None
        error = None
        if not text:
            error = "Please paste some legal text to analyze."
        else:
            max_len = settings.MAX_TEXT_LENGTH
            if len(text) > max_len:
                error = f"Text too long ({len(text)} chars). Max is {max_len}."
            else:
                nlp = get_nlp(settings.SPACY_MODEL, settings.FALLBACK_SPACY_MODEL)
                result = analyze_text(nlp, text)
        return render(request, "result.html", {"text": text, "result": result, "error": error})

    return render(request, "index.html")


def trends(request):
    data = _load_dashboard_data()
    return render(request, "trends.html", {
        "data": data,
        "data_json": json.dumps(data, default=str),
    })


def retrieval(request):
    from .retrieval import DEFAULT_WEIGHTS
    from .retrieval.semantic import get_semantic_retriever

    CATEGORIES = [
        "Criminal", "Civil", "Constitutional", "Tax", "Service",
        "Arbitration", "Contempt", "Election", "Family", "MotorAccident",
        "Consumer", "Labour", "Reference", "Other",
    ]
    CATEGORY_COLORS = {
        "Criminal": "#c0392b", "Civil": "#2980b9", "Constitutional": "#8e44ad",
        "Tax": "#16a085", "Service": "#d35400", "Arbitration": "#7f8c8d",
        "Contempt": "#2c3e50", "Election": "#27ae60", "Family": "#e67e22",
        "MotorAccident": "#34495e", "Consumer": "#1abc9c", "Labour": "#f39c12",
        "Reference": "#95a5a6", "Other": "#bdc3c7",
    }
    weight_labels = [
        ("bm25",     "BM25 / Keyword",  DEFAULT_WEIGHTS["bm25"]),
        ("semantic", "Semantic (BERT)", DEFAULT_WEIGHTS["semantic"]),
        ("sections", "Shared Sections", DEFAULT_WEIGHTS["sections"]),
        ("statutes", "Shared Statutes", DEFAULT_WEIGHTS["statutes"]),
        ("citation", "Citation Link",   DEFAULT_WEIGHTS["citation"]),
        ("articles", "Const. Articles", DEFAULT_WEIGHTS["articles"]),
        ("category", "Same Category",   DEFAULT_WEIGHTS["category"]),
    ]
    sem = get_semantic_retriever()
    return render(request, "retrieval.html", {
        "categories": CATEGORIES,
        "category_colors": json.dumps(CATEGORY_COLORS),  # ← pre-serialized
        "weights": weight_labels,
        "default_weights": json.dumps(DEFAULT_WEIGHTS),  # ← pre-serialized
        "semantic_available": sem.is_available(),
    })


# ── API Routes ──────────────────────

def health(request):
    nlp = get_nlp(settings.SPACY_MODEL, settings.FALLBACK_SPACY_MODEL)
    return JsonResponse({
        "status": "ok",
        "model": nlp.meta.get("name", "unknown"),
        "pipeline": nlp.pipe_names,
    })


@csrf_exempt
def analyze_json(request):
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        payload = {}

    text = (payload.get("text") or "").strip()
    if not text:
        return JsonResponse({"error": "Field 'text' is required."}, status=400)

    max_len = settings.MAX_TEXT_LENGTH
    if len(text) > max_len:
        return JsonResponse({"error": f"Text exceeds {max_len} characters."}, status=413)

    nlp = get_nlp(settings.SPACY_MODEL, settings.FALLBACK_SPACY_MODEL)
    return JsonResponse(analyze_text(nlp, text))


def dashboard_api(request):
    return JsonResponse(_load_dashboard_data())


# ── Retrieval API ──────────────────────


@csrf_exempt
def retrieve_api(request):
    from .retrieval import retrieve

    try:
        body = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        body = {}

    query   = str(body.get("query",   "") or "").strip()
    case_id = str(body.get("case_id", "") or "").strip() or None
    filters = body.get("filters") or {}
    weights = body.get("weights") or {}
    top_k   = int(body.get("top_k", 20))

    if not query and not case_id:
        return JsonResponse({"status": "error", "error": "Provide 'query' or 'case_id'."}, status=400)

    top_k = max(1, min(top_k, 100))

    try:
        result = retrieve(
            query=query,
            case_id=case_id,
            filters=filters,
            weights=weights or None,
            top_k=top_k,
        )
        return JsonResponse(result)
    except Exception as exc:
        log.exception("Retrieval error")
        return JsonResponse({"status": "error", "error": str(exc)}, status=500)
    

def retrieve_similar(request, case_id: str):
    from .retrieval import retrieve

    top_k = int(request.GET.get("top_k", 20))
    try:
        result = retrieve(query="", case_id=case_id, top_k=top_k)
        return JsonResponse(result)
    except Exception as exc:
        log.exception("Retrieval error")
        return JsonResponse({"status": "error", "error": str(exc)}, status=500)
    

def search_cases(request):
    from .retrieval import get_lexical_retriever

    q = (request.GET.get("q") or "").strip()
    if len(q) < 2:
        return JsonResponse({"results": []})
    lex = get_lexical_retriever()
    results = lex.search_titles(q, top_k=10)
    return JsonResponse({"results": results})


def retrieval_status(request):
    from .retrieval.semantic import get_semantic_retriever

    sem = get_semantic_retriever()
    return JsonResponse({
        "semantic_available": sem.is_available(),
        "bm25_cache_exists": (
            Path(__file__).parent / "data" / "processed" / "retrieval" / "bm25_corpus.pkl"
        ).exists(),
    })
