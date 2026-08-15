"""Tests for graph rendering utilities (vis payload + paper-quality figures).

These run without spaCy: they load the bundled pre-generated outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from FKGIS.nlp_pipeline.graph_render import (
    build_vis_payload,
    load_nodes_edges,
    render_static,
)
from FKGIS.nlp_pipeline.kg.pagerank_analysis import compute_pagerank


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "FKGIS" / "nlp_pipeline" / "output"
NODES_PATH = OUTPUT_DIR / "graph_nodes.json"
EDGES_PATH = OUTPUT_DIR / "graph_edges.json"
PAGERANK_PATH = OUTPUT_DIR / "pagerank_results.json"


def _load() -> tuple:
    return load_nodes_edges(NODES_PATH, EDGES_PATH, PAGERANK_PATH)


def test_load_nodes_edges_loads_bundled_outputs() -> None:
    nodes, edges, pagerank = _load()
    assert len(nodes) > 100
    assert len(edges) > 0
    assert len(pagerank) == len(nodes)


def test_build_vis_payload_is_small_and_typed() -> None:
    nodes, edges, pagerank = _load()

    payload = build_vis_payload(nodes, edges, pagerank)
    assert len(payload["nodes"]) == len(nodes)
    assert len(payload["edges"]) == len(edges)
    assert payload["stats"]["nodes"] == len(nodes)
    assert all(set(node) >= {"id", "label", "group", "value"} for node in payload["nodes"])
    assert all(set(edge) >= {"from", "to", "label"} for edge in payload["edges"])


def test_pagerank_metadata_reported() -> None:
    nodes, edges, _ = _load()
    result = compute_pagerank(nodes, edges)
    assert result["metadata"]["significant_nodes_count"] > 0
    assert len(result["nodes"]) == len(nodes)
    assert all("id" in entry and "score" in entry for entry in result["nodes"])


def test_render_static_png(tmp_path) -> None:
    nodes, edges, pagerank = _load()
    out = Path(render_static(nodes, edges, pagerank, output_path=tmp_path / "kg.png"))
    assert out.exists()
    assert out.stat().st_size > 50_000


def test_render_static_svg(tmp_path) -> None:
    nodes, edges, pagerank = _load()
    out = Path(render_static(nodes, edges, pagerank, output_path=tmp_path / "kg.svg"))
    assert out.exists()
    assert out.stat().st_size > 10_000