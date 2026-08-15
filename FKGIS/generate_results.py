#!/usr/bin/env python3
"""
Generate comprehensive results file with NLP pipeline metrics and analysis.

This script collects and analyzes the results from the NLP pipeline, including:
- Pipeline timing information
- Node and edge counts
- PageRank analysis
- Entity extraction statistics
- Relation category analysis
"""

import json
import time
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List


SCRIPT_DIR = Path(__file__).resolve().parent
NLP_OUTPUT_DIR = SCRIPT_DIR / "nlp_pipeline" / "output"


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON file safely."""
    try:
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def analyze_relation_categories(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze dominant relation categories."""
    relation_counts = Counter(
        (edge.get("edge_type") or edge.get("relation") or edge.get("type") or "unknown")
        for edge in edges
    )

    return {
        "total_relations": len(edges),
        "unique_relation_types": len(relation_counts),
        "dominant_relations": dict(relation_counts.most_common(10)),
        "relation_distribution": dict(relation_counts)
    }


def extract_sample_entities(nodes: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    """Extract sample entities for display."""
    entities = [node for node in nodes if node.get('type') == 'Entity']
    return entities[:limit]


def _safe_density(node_count: int, edge_count: int) -> float:
    if node_count < 2:
        return 0.0
    return (2.0 * edge_count) / (node_count * (node_count - 1))


def generate_results_file(output_path: Path = Path("nlp_pipeline_results.json")) -> None:
    """Generate comprehensive results file with all pipeline metrics."""

    # Define paths
    nlp_output_dir = NLP_OUTPUT_DIR

    # Load all relevant data files
    pagerank_results = load_json_file(nlp_output_dir / "pagerank_results.json")
    graph_nodes_data = load_json_file(nlp_output_dir / "graph_nodes.json")
    graph_edges_data = load_json_file(nlp_output_dir / "graph_edges.json")
    case_output = load_json_file(nlp_output_dir / "case_output.json")

    # Extract lists from loaded data
    graph_nodes = graph_nodes_data if isinstance(graph_nodes_data, list) else graph_nodes_data.get('nodes', []) if isinstance(graph_nodes_data, dict) else []
    graph_edges = graph_edges_data if isinstance(graph_edges_data, list) else graph_edges_data.get('edges', []) if isinstance(graph_edges_data, dict) else []

    # Extract timing information (if available from recent runs)
    timing_info = {}
    if "timing" in case_output:
        timing_info = case_output["timing"]

    # Analyze PageRank results
    top_pagerank_entities = []
    if "nodes" in pagerank_results:
        # Sort by score descending and take top 10
        sorted_nodes = sorted(
            pagerank_results["nodes"],
            key=lambda x: x.get("score", 0),
            reverse=True
        )
        top_pagerank_entities = [
            {"name": node.get("text", "Unknown"), "score": node.get("score", 0)}
            for node in sorted_nodes[:10]
        ]

    # Analyze relation categories
    relation_analysis = analyze_relation_categories(graph_edges) if graph_edges else {}

    node_type_counts = Counter(node.get('type', 'unknown') for node in graph_nodes)
    entity_labels = Counter(
        node.get('label', 'unknown')
        for node in graph_nodes
        if node.get('type') == 'Entity'
    )
    nullish_entities = sum(
        1
        for node in graph_nodes
        if str(node.get('label', '')).strip().lower() in {'', 'none', 'null', 'unknown'}
    )

    # Extract sample entities
    sample_entities = extract_sample_entities(graph_nodes)

    # Compile comprehensive results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "pipeline_metrics": {
            "total_pipeline_time_seconds": timing_info.get("total_pipeline_time", "Not available"),
            "document_discovery_time": timing_info.get("document_discovery", "Not available"),
            "pipeline_setup_time": timing_info.get("pipeline_setup", "Not available"),
            "document_processing_time": timing_info.get("document_processing", "Not available"),
            "timeline_building_time": timing_info.get("timeline_building", "Not available"),
            "kg_creation_time": timing_info.get("kg_creation", "Not available"),
            "verification_time": timing_info.get("verification", "Not available")
        },
        "graph_statistics": {
            "total_nodes": len(graph_nodes),
            "total_edges": len(graph_edges),
            "density": _safe_density(len(graph_nodes), len(graph_edges)),
            "node_types": node_type_counts,
            "entity_labels": entity_labels,
            "nullish_entities": nullish_entities,
        },
        "pagerank_analysis": {
            "method_used": pagerank_results.get("metadata", {}).get("pagerank_method", "Unknown"),
            "significant_nodes_count": pagerank_results.get("metadata", {}).get("significant_nodes_count", 0),
            "score_statistics": pagerank_results.get("metadata", {}).get("score_statistics", {}),
            "top_10_entities": top_pagerank_entities
        },
        "relation_analysis": relation_analysis,
        "sample_extracted_entities": [
            {
                "id": entity.get("id"),
                "text": entity.get("text"),
                "label": entity.get("label"),
                "type": entity.get("type")
            }
            for entity in sample_entities
        ],
        "case_information": {
            "case_id": case_output.get("case_id", "Unknown"),
            "documents_processed": case_output.get("documents", []),
            "timeline_events": len(case_output.get("timeline", {}).get("events", [])),
            "verification_status": case_output.get("verification", {})
        }
    }

    # Save results
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Comprehensive results saved to {output_path}")
    print(f"Total nodes extracted: {results['graph_statistics']['total_nodes']}")
    print(f"Total edges created: {results['graph_statistics']['total_edges']}")
    print(f"Significant nodes (PageRank): {results['pagerank_analysis']['significant_nodes_count']}")
    print(f"Total pipeline time: {results['pipeline_metrics']['total_pipeline_time_seconds']}")


if __name__ == "__main__":
    generate_results_file()