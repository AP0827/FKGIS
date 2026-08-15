from __future__ import annotations

import json
from pathlib import Path

from FKGIS.nlp_pipeline.evaluation import compare_graph_variants, graph_summary


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "FKGIS" / "nlp_pipeline" / "output"


def _load(name: str):
    return json.loads((OUTPUT_DIR / name).read_text(encoding="utf-8"))


def test_spacy_only_baseline_graph_summary() -> None:
    """Treat the raw graph outputs as the spaCy-only baseline."""
    nodes = _load("graph_nodes.json")
    edges = _load("graph_edges.json")
    summary = graph_summary(nodes, edges)

    print("spaCy-only baseline summary:")
    print(json.dumps({
        "nodes": summary["nodes"],
        "edges": summary["edges"],
        "components": summary["components"],
        "largest_component": summary["largest_component"],
        "density": summary["density"],
        "avg_degree": summary["avg_degree"],
        "avg_clustering": summary["avg_clustering"],
        "isolates": summary["isolates"],
        "nullish_entities": summary["nullish_entities"],
        "unique_relation_types": summary["unique_relation_types"],
        "top_relation_types": summary["top_relation_types"],
    }, indent=2, default=str))

    assert summary["node_records"] == 270
    assert summary["edge_records"] == 149
    assert summary["nodes"] == 270
    assert summary["edges"] == 138
    assert summary["nullish_entities"] == 111
    assert summary["unique_relation_types"] == 26


def test_llm_only_baseline_graph_summary() -> None:
    """Treat the refined graph outputs as the LLM-based baseline."""
    nodes = _load("graph_nodes_refined.json")
    edges = _load("graph_edges_refined.json")
    summary = graph_summary(nodes, edges)

    print("LLM-based baseline summary:")
    print(json.dumps({
        "nodes": summary["nodes"],
        "edges": summary["edges"],
        "components": summary["components"],
        "largest_component": summary["largest_component"],
        "density": summary["density"],
        "avg_degree": summary["avg_degree"],
        "avg_clustering": summary["avg_clustering"],
        "isolates": summary["isolates"],
        "nullish_entities": summary["nullish_entities"],
        "unique_relation_types": summary["unique_relation_types"],
        "top_relation_types": summary["top_relation_types"],
    }, indent=2, default=str))

    assert summary["node_records"] == 287
    assert summary["edge_records"] == 143
    assert summary["nodes"] == 291
    assert summary["edges"] == 143
    assert summary["nullish_entities"] == 93
    assert summary["unique_relation_types"] == 14


def test_baseline_delta_report() -> None:
    raw_nodes = _load("graph_nodes.json")
    raw_edges = _load("graph_edges.json")
    refined_nodes = _load("graph_nodes_refined.json")
    refined_edges = _load("graph_edges_refined.json")

    comparison = compare_graph_variants(raw_nodes, raw_edges, refined_nodes, refined_edges)

    print("Baseline delta report:")
    print(json.dumps({
        "delta_node_records": comparison["delta_node_records"],
        "delta_edge_records": comparison["delta_edge_records"],
        "delta_nodes": comparison["delta_nodes"],
        "delta_edges": comparison["delta_edges"],
        "delta_nullish_entities": comparison["delta_nullish_entities"],
        "delta_unique_relation_types": comparison["delta_unique_relation_types"],
    }, indent=2))

    assert comparison["delta_node_records"] == 17
    assert comparison["delta_edge_records"] == -6
    assert comparison["delta_nodes"] == 21
    assert comparison["delta_edges"] == 5
    assert comparison["delta_nullish_entities"] == -18
    assert comparison["delta_unique_relation_types"] == -12
