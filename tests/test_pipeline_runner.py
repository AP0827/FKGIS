"""Regression tests for the web app pipeline runner and case workspace prep.

These run without spaCy and without an LLM key: the refine step uses the
deterministic fallback path that previously raised a NameError.
"""

from __future__ import annotations

import json

import pytest

from FKGIS.webapp.case_manager import CaseManager
from FKGIS.webapp.pipeline_runner import PipelineRunner


@pytest.fixture()
def manager(tmp_path) -> CaseManager:
    return CaseManager(sessions_dir=tmp_path / "sessions")


def _write_graph(output_dir, n: int = 5) -> None:
    nodes = [
        {"id": f"e{i}", "label": f"Entity {i}", "type": "Entity", "mention_ids": [f"m{i}"]}
        for i in range(n)
    ]
    edges = [
        {"source": "e0", "target": f"e{i}", "relation_type": "related_to", "source_id": "e0", "target_id": f"e{i}", "edge_type": "related_to"}
        for i in range(1, n)
    ]
    (output_dir / "graph_nodes.json").write_text(json.dumps(nodes), encoding="utf-8")
    (output_dir / "graph_edges.json").write_text(json.dumps(edges), encoding="utf-8")


def test_maybe_refine_fallback_produces_refined_artifacts(tmp_path) -> None:
    manager = CaseManager(sessions_dir=tmp_path / "sessions")
    case = manager.create_case("Refine Test")
    output_dir = manager.output_dir(case["id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_graph(output_dir)

    runner = PipelineRunner(manager, use_llm=False)
    runner._maybe_refine(case["id"], output_dir)

    assert (output_dir / "graph_nodes_refined.json").exists()
    assert (output_dir / "graph_edges_refined.json").exists()
    refined_nodes = json.loads((output_dir / "graph_nodes_refined.json").read_text(encoding="utf-8"))
    assert len(refined_nodes) == 5


def test_prepare_work_dir_does_not_double_prefix(manager: CaseManager) -> None:
    case = manager.create_case("Prefix Test")
    manager.add_document(case["id"], "Interview_Kimberley_Friend.txt", b"hello", "interview")
    manager.add_document(case["id"], "statement.txt", b"hello", "narrative")

    work_dir = manager.prepare_work_dir(case["id"])
    names = sorted(p.name for p in work_dir.iterdir())
    assert names == ["IncidentReport_statement.txt", "Interview_Kimberley_Friend.txt"]
    assert "Interview_Interview_Kimberley_Friend.txt" not in names


def test_status_write_is_atomic(manager: CaseManager) -> None:
    case = manager.create_case("Status Test")
    runner = PipelineRunner(manager, use_llm=False)
    runner._write_status(case["id"], "running", "working...")
    status = runner.read_status(case["id"])
    assert status["state"] == "running"
    assert not (manager.status_path(case["id"]).with_suffix(".json.tmp")).exists()