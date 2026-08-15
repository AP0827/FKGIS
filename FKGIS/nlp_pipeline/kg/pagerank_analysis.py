"""Reusable graph analytics for knowledge graph artifacts.

Provides a deterministic PageRank implementation over the node/edge lists
produced by the knowledge graph builder, returning the same result schema as
``pagerank_results.json``. This lets the web app (and any caller) compute
importance scores without shelling out to the standalone ``output/pagerank.py``
script.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx


def compute_pagerank(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    alpha: float = 0.85,
    threshold_mode: str = "mean_0.5std",
) -> Dict[str, Any]:
    """Compute PageRank scores and significance flags for graph nodes.

    Args:
        nodes: List of node dicts, each with an ``id``.
        edges: List of edge dicts, each with ``source`` and ``target``.
        alpha: PageRank damping factor.
        threshold_mode: How to decide which nodes are "significant".
            ``mean_0.5std`` uses mean + 0.5*std (robust for larger graphs);
            ``max_0.3`` uses 30% of the max score (small graphs).

    Returns:
        The standard pagerank results payload:
        ``{"metadata": {...}, "nodes": [{id, text, label, score, is_significant}]}``
    """
    G = nx.DiGraph()
    for node in nodes:
        node_id = node.get("id")
        if node_id:
            G.add_node(node_id)
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source in G and target in G and source != target:
            G.add_edge(source, target)

    id_to_node = {n.get("id"): n for n in nodes if n.get("id")}

    if G.number_of_nodes() == 0:
        return {
            "metadata": {
                "total_nodes": 0,
                "total_edges": 0,
                "significant_nodes_count": 0,
                "pagerank_method": "standard pagerank",
                "threshold_used": 0.0,
                "score_statistics": {"min": 0, "max": 0, "mean": 0, "median": 0},
            },
            "nodes": [],
        }

    pagerank = nx.pagerank(G, alpha=alpha, max_iter=200, tol=1e-6)

    scores = list(pagerank.values())
    mean_score = statistics.mean(scores)
    if threshold_mode == "max_0.3" or len(scores) <= 10:
        threshold = max(scores) * 0.3
    else:
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        threshold = mean_score + (0.5 * std_dev)

    sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    output_all = []
    significant = 0
    for node_id, score in sorted_nodes:
        node = id_to_node.get(node_id, {})
        is_sig = score >= threshold
        if is_sig:
            significant += 1
        output_all.append(
            {
                "id": node_id,
                "text": node.get("text", "Unknown"),
                "label": node.get("label", "Unknown"),
                "score": round(score, 6),
                "is_significant": is_sig,
            }
        )

    return {
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "significant_nodes_count": significant,
            "pagerank_method": "standard pagerank",
            "threshold_used": round(threshold, 6),
            "score_statistics": {
                "min": round(min(scores), 6),
                "max": round(max(scores), 6),
                "mean": round(mean_score, 6),
                "median": round(statistics.median(scores), 6),
            },
        },
        "nodes": output_all,
    }


def save_pagerank_results(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    output_path: str | Path,
) -> Dict[str, Any]:
    """Compute PageRank and persist the results JSON (and a .txt companion)."""
    results = compute_pagerank(nodes, edges)
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    text_path = output_path.with_suffix(".txt")
    meta = results["metadata"]
    lines = [
        "=== PAGE RANK RESULTS ===",
        f"Method: {meta['pagerank_method']}",
        f"Threshold: {meta['threshold_used']:.6f}",
        f"Significant nodes: {meta['significant_nodes_count']}/{meta['total_nodes']}",
        "",
        "ALL NODES (sorted by importance):",
        "-" * 80,
    ]
    for entry in results["nodes"]:
        indicator = " ***" if entry["is_significant"] else ""
        lines.append(
            f"{entry['text']} [{entry['label']}] → {entry['score']:.6f}{indicator}"
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return results