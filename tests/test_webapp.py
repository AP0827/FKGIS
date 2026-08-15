"""Tests for the FKGIS web app (FastAPI, session-based, no database).

These tests exercise the REST API without spaCy: the sample case is seeded
from the bundled pre-generated outputs, so graph viewing, statistics and
figure export all work in a plain environment.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def isolated_sessions_dir(tmp_path_factory) -> None:
    """Point the app at a throwaway sessions dir before it is imported."""
    sessions = tmp_path_factory.mktemp("fkgis_sessions")
    os.environ["FKGIS_SESSIONS_DIR"] = str(sessions)
    os.environ["GEMINI_API_KEY"] = ""
    yield


@pytest.fixture(scope="session")
def client() -> TestClient:
    from FKGIS.webapp.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_case(client: TestClient) -> str:
    response = client.post("/api/cases/sample")
    assert response.status_code == 200
    return response.json()["id"]


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is False
    assert body["llm_model"] == "gemini-2.0-flash"


def test_create_and_delete_case(client: TestClient) -> None:
    created = client.post("/api/cases", json={"name": "Empty Case"})
    assert created.status_code == 200
    case = created.json()
    assert case["name"] == "Empty Case"
    assert case["status"] == "idle"
    assert case["documents"] == []

    listed = client.get("/api/cases")
    assert listed.status_code == 200
    assert any(c["id"] == case["id"] for c in listed.json())

    deleted = client.delete(f"/api/cases/{case['id']}")
    assert deleted.status_code == 200
    assert not any(c["id"] == case["id"] for c in client.get("/api/cases").json())


def test_sample_case_is_ready(client: TestClient, sample_case: str) -> None:
    case = client.get(f"/api/cases/{sample_case}").json()
    assert case["status"] == "ready"
    assert {d["doc_type"] for d in case["documents"]} == {"biography", "narrative", "interview"}


def test_document_upload_and_remove(client: TestClient, sample_case: str) -> None:
    response = client.post(
        f"/api/cases/{sample_case}/documents",
        files={"file": ("extra_statement.txt", b"An extra witness statement.", "text/plain")},
        data={"doc_type": "interview"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "extra_statement.txt"

    case = client.get(f"/api/cases/{sample_case}").json()
    assert "extra_statement.txt" in [d["name"] for d in case["documents"]]

    content = client.get(f"/api/cases/{sample_case}/documents/extra_statement.txt/content")
    assert content.status_code == 200
    assert "witness" in content.text

    removed = client.delete(f"/api/cases/{sample_case}/documents/extra_statement.txt")
    assert removed.status_code == 200
    assert removed.json()["removed"] == "extra_statement.txt"
    case = client.get(f"/api/cases/{sample_case}").json()
    assert "extra_statement.txt" not in [d["name"] for d in case["documents"]]


def test_graph_payload_has_significant_flag(client: TestClient, sample_case: str) -> None:
    for variant in ("raw", "refined"):
        response = client.get(f"/api/cases/{sample_case}/graph?variant={variant}")
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["nodes"]) > 100
        assert len(payload["edges"]) > 0
        assert all("significant" in node for node in payload["nodes"])
        assert payload["pagerank"]["significant_nodes_count"] > 0
        assert payload["variant"] == variant


def test_graph_figure_exports(client: TestClient, sample_case: str) -> None:
    for fmt, content_type in (("png", "image/png"), ("svg", "image/svg+xml"), ("pdf", "application/pdf")):
        response = client.get(f"/api/cases/{sample_case}/graph.png?variant=raw&fmt={fmt}")
        assert response.status_code == 200
        assert response.headers["content-type"] == content_type
        assert len(response.content) > 1000


def test_results_reports_graph_stats(client: TestClient, sample_case: str) -> None:
    response = client.get(f"/api/cases/{sample_case}/results")
    assert response.status_code == 200
    body = response.json()
    assert "graph_nodes.json" in body["graph_stats"]
    assert body["graph_stats"]["graph_nodes.json"]["nodes"] > 100


def test_output_artifact_access(client: TestClient, sample_case: str) -> None:
    response = client.get(f"/api/cases/{sample_case}/output/pagerank_results.json")
    assert response.status_code == 200
    assert "nodes" in response.json()


def test_run_pipeline_reports_status(client: TestClient, sample_case: str) -> None:
    response = client.post(f"/api/cases/{sample_case}/run")
    assert response.status_code == 200
    assert response.json() == {"started": True, "case_id": sample_case}
    status = client.get(f"/api/cases/{sample_case}/run/status")
    assert status.status_code == 200
    assert status.json()["state"] in {"running", "error"}


def test_frontend_served(client: TestClient) -> None:
    index = client.get("/")
    assert index.status_code == 200
    assert "vis-network" in index.text

    app_js = client.get("/static/app.js")
    assert app_js.status_code == 200
    assert "vis.Network" in app_js.text