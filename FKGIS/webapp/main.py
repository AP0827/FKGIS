"""FKGIS web application.

A self-contained FastAPI app for running FKGIS case sessions without any
database. Each case owns a directory under the sessions folder; documents are
uploaded as plain text files and the knowledge graph is generated, refined
(optionally with the Gemini LLM) and visualized through a browser frontend.

Run with::

    python -m FKGIS.webapp.main
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .case_manager import CaseError, CaseManager
from .config import get_settings
from .pipeline_runner import PipelineRunner

settings = get_settings()

manager = CaseManager(
    settings.sessions_dir,
    bundled_case_docs=settings.bundled_case_docs,
    bundled_outputs=settings.bundled_outputs,
)

runner = PipelineRunner(
    manager,
    gemini_api_key=settings.gemini_api_key,
    gemini_model=settings.gemini_model,
    use_llm=settings.use_llm,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="FKGIS — Forensic Knowledge Graph Intelligence",
    description="Session-based knowledge graph generation for forensic case documents.",
    version="0.2.0",
)

_run_lock = threading.Lock()
_run_threads: Dict[str, threading.Thread] = {}


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# Case sessions
# --------------------------------------------------------------------------- #
@app.get("/api/cases")
def list_cases() -> List[Dict[str, Any]]:
    return manager.list_cases()


@app.post("/api/cases")
def create_case(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = (payload or {}).get("name", "Untitled Case")
    case_id = (payload or {}).get("id")
    try:
        return manager.create_case(name, case_id=case_id)
    except CaseError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/cases/sample")
def create_sample_case() -> Dict[str, Any]:
    try:
        return manager.create_sample_case()
    except CaseError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> Dict[str, Any]:
    try:
        return manager.get_case(case_id)
    except CaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/api/cases/{case_id}")
def delete_case(case_id: str) -> Dict[str, str]:
    try:
        manager.delete_case(case_id)
        return {"deleted": case_id}
    except CaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
@app.get("/api/cases/{case_id}/documents")
def list_documents(case_id: str) -> List[Dict[str, Any]]:
    try:
        return manager.list_documents(case_id)
    except CaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/cases/{case_id}/documents")
async def upload_document(
    case_id: str,
    file: UploadFile = File(...),
    doc_type: str = Form("narrative"),
) -> Dict[str, Any]:
    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 5 MB).")
    try:
        return manager.add_document(case_id, file.filename or "document.txt", content, doc_type)
    except CaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/cases/{case_id}/documents/{doc_name}/content")
def document_content(case_id: str, doc_name: str) -> Response:
    try:
        path = manager.case_docs_dir(case_id) / doc_name
        text = path.read_text(encoding="utf-8")
    except (CaseError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return Response(text, media_type="text/plain")


@app.delete("/api/cases/{case_id}/documents/{doc_name}")
def remove_document(case_id: str, doc_name: str) -> Dict[str, str]:
    try:
        manager.remove_document(case_id, doc_name)
        return {"removed": doc_name}
    except CaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# --------------------------------------------------------------------------- #
# Pipeline run
# --------------------------------------------------------------------------- #
@app.post("/api/cases/{case_id}/run")
def run_pipeline(case_id: str) -> Dict[str, Any]:
    try:
        manager.get_case(case_id)
    except CaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not manager.list_documents(case_id):
        raise HTTPException(status_code=400, detail="Add at least one document before running.")

    with _run_lock:
        existing = _run_threads.get(case_id)
        if existing and existing.is_alive():
            raise HTTPException(status_code=409, detail="A pipeline run is already in progress.")

        def _work() -> None:
            try:
                runner.run(case_id)
            finally:
                with _run_lock:
                    _run_threads.pop(case_id, None)

        thread = threading.Thread(target=_work, name=f"fkgis-run-{case_id}", daemon=True)
        _run_threads[case_id] = thread
        thread.start()

    return {"started": True, "case_id": case_id}


@app.get("/api/cases/{case_id}/run/status")
def run_status(case_id: str) -> Dict[str, Any]:
    return runner.read_status(case_id)


# --------------------------------------------------------------------------- #
# Results / graph
# --------------------------------------------------------------------------- #
def _resolve_artifact(case_id: str, variant: str, artifact: str) -> Path:
    """Return the path for an artifact, honoring raw/refined variant."""
    if variant == "refined":
        refined = artifact.replace("graph_nodes.json", "graph_nodes_refined.json")
        path = manager.output_dir(case_id) / refined
        if path.exists():
            return path
    path = manager.output_dir(case_id) / artifact
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run the pipeline first — missing {artifact}.")
    return path


@app.get("/api/cases/{case_id}/results")
def case_results(case_id: str) -> Dict[str, Any]:
    try:
        info = manager.get_case(case_id)
    except CaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    outputs = info.get("outputs", {})
    timing = info.get("timing", {})

    graph_stats: Dict[str, Any] = {}
    for nodes_name, edges_name in (("graph_nodes.json", "graph_edges.json"),
                                   ("graph_nodes_refined.json", "graph_edges_refined.json")):
        nodes_path = manager.output_dir(case_id) / nodes_name
        edges_path = manager.output_dir(case_id) / edges_name
        if not nodes_path.exists() or not edges_path.exists():
            continue
        from ..nlp_pipeline.evaluation import graph_summary

        with nodes_path.open("r", encoding="utf-8") as f:
            nodes = json.load(f)
        with edges_path.open("r", encoding="utf-8") as f:
            edges = json.load(f)
        graph_stats[nodes_name] = graph_summary(nodes, edges)

    return {
        "case": info,
        "outputs": outputs,
        "timing": timing,
        "graph_stats": graph_stats,
        "llm_configured": bool(settings.gemini_api_key and settings.use_llm),
        "llm_model": settings.gemini_model,
    }


@app.get("/api/cases/{case_id}/graph")
def graph_payload(
    case_id: str,
    variant: str = Query("raw", pattern="^(raw|refined)$"),
) -> Dict[str, Any]:
    nodes_path = _resolve_artifact(case_id, variant, "graph_nodes.json")
    edges_path = _resolve_artifact(case_id, variant, "graph_edges.json")

    from ..nlp_pipeline.graph_render import build_vis_payload
    from ..nlp_pipeline.kg.pagerank_analysis import compute_pagerank

    with nodes_path.open("r", encoding="utf-8") as f:
        nodes = json.load(f)
    with edges_path.open("r", encoding="utf-8") as f:
        edges = json.load(f)

    pagerank = compute_pagerank(nodes, edges)
    scores = {entry["id"]: entry["score"] for entry in pagerank["nodes"]}
    threshold = pagerank["metadata"].get("threshold_used", 0.0)
    payload = build_vis_payload(nodes, edges, scores)
    for node in payload["nodes"]:
        node["significant"] = scores.get(node["id"], 0.0) >= threshold
    payload["variant"] = variant
    payload["pagerank"] = pagerank["metadata"]
    return payload


@app.get("/api/cases/{case_id}/graph.png")
def graph_figure(
    case_id: str,
    variant: str = Query("raw", pattern="^(raw|refined)$"),
    fmt: str = Query("png", pattern="^(png|svg|pdf)$"),
) -> FileResponse:
    import tempfile

    from ..nlp_pipeline.graph_render import render_static

    nodes_path = _resolve_artifact(case_id, variant, "graph_nodes.json")
    edges_path = _resolve_artifact(case_id, variant, "graph_edges.json")
    with nodes_path.open("r", encoding="utf-8") as f:
        nodes = json.load(f)
    with edges_path.open("r", encoding="utf-8") as f:
        edges = json.load(f)

    suffix = f".{fmt}"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix=f"fkgis_{case_id}_")
    import os

    os.close(fd)
    media_types = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}
    try:
        render_static(nodes, edges, output_path=Path(tmp_path))
        return FileResponse(tmp_path, media_type=media_types[fmt])
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


@app.get("/api/cases/{case_id}/graph.json")
def graph_download(
    case_id: str,
    variant: str = Query("raw", pattern="^(raw|refined)$"),
) -> Response:
    nodes_path = _resolve_artifact(case_id, variant, "graph_nodes.json")
    edges_path = _resolve_artifact(case_id, variant, "graph_edges.json")
    with nodes_path.open("r", encoding="utf-8") as f:
        nodes = json.load(f)
    with edges_path.open("r", encoding="utf-8") as f:
        edges = json.load(f)
    payload = {"variant": variant, "nodes": nodes, "edges": edges}
    return JSONResponse(payload)


@app.get("/api/cases/{case_id}/output/{artifact:path}")
def output_file(case_id: str, artifact: str) -> FileResponse:
    path = manager.output_dir(case_id) / artifact
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact}' not found.")
    return FileResponse(path)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "llm_configured": bool(settings.gemini_api_key),
        "llm_model": settings.gemini_model,
        "use_llm": settings.use_llm,
        "sessions_dir": str(settings.sessions_dir),
    }


def main() -> None:
    import uvicorn

    uvicorn.run("FKGIS.webapp.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()