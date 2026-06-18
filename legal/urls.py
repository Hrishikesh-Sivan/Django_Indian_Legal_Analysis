from django.urls import path
from . import views

urlpatterns = [
    # UI pages
    path("", views.index, name="index"),
    path("analyze", views.analyze, name="analyze"),
    path("trends", views.trends, name="trends"),
    path("retrieve", views.retrieval, name="retrieve_page"),

    # JSON API
    path("api/health", views.health, name="api_health"),
    path("api/analyze", views.analyze_json, name="api_analyze"),
    path("api/dashboard", views.dashboard_api, name="api_dashboard"),
    path("api/retrieve", views.retrieve_api, name="api_retrieve"),
    path("api/retrieve/case/<path:case_id>", views.retrieve_similar, name="api_retrieve_case"),
    path("api/retrieve/search-cases", views.search_cases, name="api_search_cases"),
    path("api/retrieve/status", views.retrieval_status, name="api_retrieve_status"),
]