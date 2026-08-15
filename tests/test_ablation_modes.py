from __future__ import annotations

import json
from pathlib import Path

from FKGIS.nlp_pipeline.evaluation import compare_graph_variants, graph_summary
from FKGIS.nlp_pipeline.timeline.timeline_builder import build_before_after_edges


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "FKGIS" / "nlp_pipeline" / "output"


def _load(name: str):
    return json.loads((OUTPUT_DIR / name).read_text(encoding="utf-8"))


def test_llm_refinement_off_ablation_summary() -> None:
    """Model the LLM-refinement-off ablation by comparing raw and refined graphs."""
    raw_nodes = _load("graph_nodes.json")
    raw_edges = _load("graph_edges.json")
    refined_nodes = _load("graph_nodes_refined.json")
    refined_edges = _load("graph_edges_refined.json")

    raw_summary = graph_summary(raw_nodes, raw_edges)
    refined_summary = graph_summary(refined_nodes, refined_edges)
    comparison = compare_graph_variants(raw_nodes, raw_edges, refined_nodes, refined_edges)

    print("LLM-refinement-off ablation (raw graph):")
    print(json.dumps({
        "node_records": raw_summary["node_records"],
        "edge_records": raw_summary["edge_records"],
        "nodes": raw_summary["nodes"],
        "edges": raw_summary["edges"],
        "nullish_entities": raw_summary["nullish_entities"],
        "unique_relation_types": raw_summary["unique_relation_types"],
        "components": raw_summary["components"],
        "largest_component": raw_summary["largest_component"],
        "avg_degree": raw_summary["avg_degree"],
        "density": raw_summary["density"],
    }, indent=2, default=str))

    print("LLM-refinement-on summary:")
    print(json.dumps({
        "node_records": refined_summary["node_records"],
        "edge_records": refined_summary["edge_records"],
        "nodes": refined_summary["nodes"],
        "edges": refined_summary["edges"],
        "nullish_entities": refined_summary["nullish_entities"],
        "unique_relation_types": refined_summary["unique_relation_types"],
        "components": refined_summary["components"],
        "largest_component": refined_summary["largest_component"],
        "avg_degree": refined_summary["avg_degree"],
        "density": refined_summary["density"],
    }, indent=2, default=str))

    print("LLM-refinement delta:")
    print(json.dumps({
        "delta_node_records": comparison["delta_node_records"],
        "delta_edge_records": comparison["delta_edge_records"],
        "delta_nodes": comparison["delta_nodes"],
        "delta_edges": comparison["delta_edges"],
        "delta_nullish_entities": comparison["delta_nullish_entities"],
        "delta_unique_relation_types": comparison["delta_unique_relation_types"],
    }, indent=2))

    assert raw_summary["node_records"] == 270
    assert raw_summary["edge_records"] == 149
    assert refined_summary["node_records"] == 287
    assert refined_summary["edge_records"] == 143
    assert comparison["delta_node_records"] == 17
    assert comparison["delta_edge_records"] == -6
    assert comparison["delta_nodes"] == 21
    assert comparison["delta_edges"] == 5
    assert comparison["delta_nullish_entities"] == -18
    assert comparison["delta_unique_relation_types"] == -12


def test_temporal_reasoning_ablation_on_and_off() -> None:
    """Show that timeline edges appear only when temporal information exists."""
    case_id = "CASE_TEMPORAL_TEST"
    timed_case_docs = [
        {
            "doc_name": "doc_a",
            "doc_type": "narrative",
            "processed_doc": {
                "events": [
                    {
                        "event_id": "doc_a_EV1",
                        "description": "Event one",
                        "time_resolved": "2022-10-16T11:06:00",
                    },
                    {
                        "event_id": "doc_a_EV2",
                        "description": "Event two",
                        "time_resolved": "2022-10-16T11:42:00",
                    },
                ]
            },
        },
        {
            "doc_name": "doc_b",
            "doc_type": "interview",
            "processed_doc": {
                "events": [
                    {
                        "event_id": "doc_b_EV1",
                        "description": "Event three",
                        "time_resolved": "2022-10-16T12:53:00",
                    }
                ]
            },
        },
    ]

    before_edges_on = build_before_after_edges(timed_case_docs)

    untimed_case_docs = [
        {
            "doc_name": entry["doc_name"],
            "doc_type": entry["doc_type"],
            "processed_doc": {
                "events": [
                    {"event_id": event["event_id"], "description": event["description"]}
                    for event in entry["processed_doc"]["events"]
                ]
            },
        }
        for entry in timed_case_docs
    ]
    before_edges_off = build_before_after_edges(untimed_case_docs)

    print("Temporal reasoning ON:")
    print(json.dumps({
        "timeline_edges": before_edges_on,
        "edge_count": len(before_edges_on),
            "resolved_times": [event.get("time_resolved") for entry in timed_case_docs for event in entry["processed_doc"]["events"]],
    }, indent=2))

    print("Temporal reasoning OFF:")
    print(json.dumps({
        "timeline_edges": before_edges_off,
        "edge_count": len(before_edges_off),
    }, indent=2))

    assert len(before_edges_on) == 2
    assert before_edges_on[0]["relation"] == "BEFORE"
    assert before_edges_on[1]["relation"] == "BEFORE"
    assert before_edges_off == []
