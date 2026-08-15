from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import networkx as nx


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def normalize_text(value: Any) -> str:
    """Normalize text for exact-match evaluation."""
    text = str(value or "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def entity_key(entity: Mapping[str, Any]) -> tuple[str, str]:
    """Create a stable entity key from the surface text and label."""
    text = entity.get("text") or entity.get("name") or entity.get("surface") or ""
    label = entity.get("label") or entity.get("type") or ""
    return normalize_text(text), normalize_text(label)


def relation_key(relation: Mapping[str, Any]) -> tuple[str, str, str]:
    """Create a stable relation key from subject, relation type/predicate, and object."""
    subject = relation.get("subject") or relation.get("source_text") or relation.get("from_text") or ""
    object_ = relation.get("object") or relation.get("target_text") or relation.get("to_text") or ""
    relation_type = (
        relation.get("relation_type")
        or relation.get("edge_type")
        or relation.get("type")
        or relation.get("predicate")
        or relation.get("relation")
        or ""
    )
    return normalize_text(subject), normalize_text(relation_type), normalize_text(object_)


def prf1(gold_items: Iterable[tuple[str, ...]], pred_items: Iterable[tuple[str, ...]]) -> dict[str, float | int]:
    """Compute exact-match precision/recall/F1 using multiset counts."""
    gold_counts = Counter(gold_items)
    pred_counts = Counter(pred_items)

    keys = set(gold_counts) | set(pred_counts)
    true_positive = sum(min(gold_counts[key], pred_counts[key]) for key in keys)
    false_positive = sum(max(pred_counts[key] - gold_counts[key], 0) for key in keys)
    false_negative = sum(max(gold_counts[key] - pred_counts[key], 0) for key in keys)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_documents(
    gold_documents: Sequence[Mapping[str, Any]],
    predicted_documents: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Evaluate entity and relation extraction against gold documents."""
    gold_by_id = {str(doc.get("doc_id")): doc for doc in gold_documents}
    pred_by_id = {str(doc.get("doc_id")): doc for doc in predicted_documents}
    all_doc_ids = sorted(set(gold_by_id) | set(pred_by_id))

    gold_entities: list[tuple[str, str]] = []
    pred_entities: list[tuple[str, str]] = []
    gold_relations: list[tuple[str, str, str]] = []
    pred_relations: list[tuple[str, str, str]] = []

    for doc_id in all_doc_ids:
        gold_doc = gold_by_id.get(doc_id, {})
        pred_doc = pred_by_id.get(doc_id, {})

        gold_entities.extend(entity_key(entity) for entity in gold_doc.get("entities", []))
        pred_entities.extend(entity_key(entity) for entity in pred_doc.get("entities", []))

        gold_relations.extend(relation_key(relation) for relation in gold_doc.get("relations", []))
        pred_relations.extend(relation_key(relation) for relation in pred_doc.get("relations", []))

    return {
        "entities": prf1(gold_entities, pred_entities),
        "relations": prf1(gold_relations, pred_relations),
    }


def build_graph(nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> nx.Graph:
    """Build an undirected graph from node and edge lists for summary metrics."""
    graph = nx.Graph()

    for node in nodes:
        node_id = node.get("id") or node.get("text") or node.get("name")
        if node_id is None:
            continue
        graph.add_node(node_id, **dict(node))

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source is None or target is None:
            continue
        graph.add_edge(source, target, **dict(edge))

    return graph


def graph_summary(nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the graph with structural and label statistics."""
    graph = build_graph(nodes, edges)
    degrees = [degree for _, degree in graph.degree()]
    components = list(nx.connected_components(graph)) if graph.number_of_nodes() else []
    relation_types = Counter(
        (edge.get("edge_type") or edge.get("relation") or edge.get("type") or "UNSPECIFIED")
        for edge in edges
    )
    node_types = Counter(node.get("type", "unknown") for node in nodes)
    entity_labels = Counter(
        node.get("label", "unknown") for node in nodes if node.get("type") == "Entity"
    )
    nullish_entities = sum(
        1
        for node in nodes
        if not normalize_text(node.get("label")) or normalize_text(node.get("label")) in {"unknown", "none", "null"}
    )

    density = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
    avg_degree = mean(degrees) if degrees else 0.0
    degree_std = pstdev(degrees) if len(degrees) > 1 else 0.0
    avg_clustering = nx.average_clustering(graph) if graph.number_of_nodes() > 2 else 0.0

    return {
        "node_records": len(nodes),
        "edge_records": len(edges),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": density,
        "avg_degree": avg_degree,
        "degree_std": degree_std,
        "components": len(components),
        "largest_component": max((len(component) for component in components), default=0),
        "avg_clustering": avg_clustering,
        "isolates": nx.number_of_isolates(graph) if graph.number_of_nodes() else 0,
        "node_types": node_types,
        "entity_labels": entity_labels,
        "nullish_entities": nullish_entities,
        "unique_relation_types": len(relation_types),
        "top_relation_types": relation_types.most_common(10),
    }


def compare_graph_variants(
    raw_nodes: Sequence[Mapping[str, Any]],
    raw_edges: Sequence[Mapping[str, Any]],
    refined_nodes: Sequence[Mapping[str, Any]],
    refined_edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a compact comparison between raw and refined graph outputs."""
    raw_summary = graph_summary(raw_nodes, raw_edges)
    refined_summary = graph_summary(refined_nodes, refined_edges)

    return {
        "raw": raw_summary,
        "refined": refined_summary,
        "delta_node_records": refined_summary["node_records"] - raw_summary["node_records"],
        "delta_edge_records": refined_summary["edge_records"] - raw_summary["edge_records"],
        "delta_nodes": refined_summary["nodes"] - raw_summary["nodes"],
        "delta_edges": refined_summary["edges"] - raw_summary["edges"],
        "delta_nullish_entities": refined_summary["nullish_entities"] - raw_summary["nullish_entities"],
        "delta_unique_relation_types": refined_summary["unique_relation_types"] - raw_summary["unique_relation_types"],
    }
