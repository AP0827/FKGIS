from __future__ import annotations

import pytest

from FKGIS.nlp_pipeline.evaluation import compare_graph_variants, evaluate_documents, graph_summary


def test_entity_and_relation_prf1_with_normalization() -> None:
    gold_docs = [
        {
            "doc_id": "Interview_Kimberley_Friend",
            "entities": [
                {"text": "Cheryl Weston", "label": "PERSON"},
                {"text": "Kimberly Pace", "label": "PERSON"},
                {"text": "The Lucky Café", "label": "ORG"},
                {"text": "Oxford", "label": "GPE"},
            ],
            "relations": [
                {"subject": "Cheryl Weston", "relation_type": "FRIEND_OF", "object": "Kimberly Pace"},
                {"subject": "Cheryl Weston", "relation_type": "OWNS_BUSINESS", "object": "The Lucky Café"},
                {"subject": "Cheryl Weston", "relation_type": "LIVES_AT", "object": "Oxford"},
            ],
        }
    ]

    predicted_docs = [
        {
            "doc_id": "Interview_Kimberley_Friend",
            "entities": [
                {"text": "Cheryl Weston", "label": "PERSON"},
                {"text": "Kimberly Pace", "label": "PERSON"},
                {"text": "The Lucky Cafe", "label": "ORG"},
                {"text": "Oxford", "label": "GPE"},
                {"text": "Paul Evans", "label": "PERSON"},
            ],
            "relations": [
                {"subject": "Cheryl Weston", "relation_type": "FRIEND_OF", "object": "Kimberly Pace"},
                {"subject": "Cheryl Weston", "relation_type": "OWNS_BUSINESS", "object": "The Lucky Cafe"},
                {"subject": "Cheryl Weston", "relation_type": "LIVES_AT", "object": "Oxford"},
                {"subject": "Cheryl Weston", "relation_type": "STATED", "object": "We were going to Big Bad Breakfast"},
            ],
        }
    ]

    metrics = evaluate_documents(gold_docs, predicted_docs)

    assert metrics["entities"]["tp"] == 4
    assert metrics["entities"]["fp"] == 1
    assert metrics["entities"]["fn"] == 0
    assert metrics["entities"]["precision"] == pytest.approx(0.8, rel=1e-6)
    assert metrics["entities"]["recall"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["entities"]["f1"] == pytest.approx(0.8888888889, rel=1e-6)

    assert metrics["relations"]["tp"] == 3
    assert metrics["relations"]["fp"] == 1
    assert metrics["relations"]["fn"] == 0
    assert metrics["relations"]["precision"] == pytest.approx(0.75, rel=1e-6)
    assert metrics["relations"]["recall"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["relations"]["f1"] == pytest.approx(0.8571428571, rel=1e-6)


def test_graph_summary_and_variant_comparison() -> None:
    raw_nodes = [
        {"id": "ENT_A", "type": "Entity", "label": "PERSON"},
        {"id": "ENT_B", "type": "Entity", "label": None},
        {"id": "EV_1", "type": "Event"},
    ]
    raw_edges = [
        {"source": "ENT_A", "target": "EV_1", "edge_type": "PARTICIPATED_IN"},
        {"source": "EV_1", "target": "ENT_B", "edge_type": "LOCATED_AT"},
    ]
    refined_nodes = [
        {"id": "ENT_A", "type": "Entity", "label": "PERSON"},
        {"id": "EV_1", "type": "Event"},
    ]
    refined_edges = [
        {"source": "ENT_A", "target": "EV_1", "edge_type": "PARTICIPATED_IN"},
    ]

    summary = graph_summary(raw_nodes, raw_edges)
    assert summary["node_records"] == 3
    assert summary["edge_records"] == 2
    assert summary["nodes"] == 3
    assert summary["edges"] == 2
    assert summary["unique_relation_types"] == 2
    assert summary["nullish_entities"] == 2
    assert summary["node_types"]["Entity"] == 2

    comparison = compare_graph_variants(raw_nodes, raw_edges, refined_nodes, refined_edges)
    assert comparison["delta_node_records"] == -1
    assert comparison["delta_edge_records"] == -1
    assert comparison["delta_nodes"] == -1
    assert comparison["delta_edges"] == -1
    assert comparison["delta_nullish_entities"] == -1
