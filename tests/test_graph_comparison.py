from __future__ import annotations

import json
from pathlib import Path

from FKGIS.nlp_pipeline.evaluation import compare_graph_variants


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "FKGIS" / "nlp_pipeline" / "output"


def test_saved_raw_and_refined_graphs_have_expected_comparison() -> None:
    raw_nodes = json.loads((OUTPUT_DIR / "graph_nodes.json").read_text(encoding="utf-8"))
    raw_edges = json.loads((OUTPUT_DIR / "graph_edges.json").read_text(encoding="utf-8"))
    refined_nodes = json.loads((OUTPUT_DIR / "graph_nodes_refined.json").read_text(encoding="utf-8"))
    refined_edges = json.loads((OUTPUT_DIR / "graph_edges_refined.json").read_text(encoding="utf-8"))

    comparison = compare_graph_variants(raw_nodes, raw_edges, refined_nodes, refined_edges)

    assert comparison["raw"]["node_records"] == 270
    assert comparison["raw"]["edge_records"] == 149
    assert comparison["raw"]["nodes"] == 270
    assert comparison["raw"]["edges"] == 138
    assert comparison["raw"]["unique_relation_types"] == 26
    assert comparison["raw"]["nullish_entities"] == 111

    assert comparison["refined"]["node_records"] == 287
    assert comparison["refined"]["edge_records"] == 143
    assert comparison["refined"]["nodes"] == 291
    assert comparison["refined"]["edges"] == 143
    assert comparison["refined"]["unique_relation_types"] == 14
    assert comparison["refined"]["nullish_entities"] == 93

    assert comparison["delta_node_records"] == 17
    assert comparison["delta_edge_records"] == -6
    assert comparison["delta_nodes"] == 21
    assert comparison["delta_edges"] == 5
    assert comparison["delta_unique_relation_types"] == -12
