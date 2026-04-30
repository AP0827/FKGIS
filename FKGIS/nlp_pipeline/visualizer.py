import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse

def get_entity_neighborhood(entity_name: str, nodes_data: List[dict], edges_data: List[dict]) -> List[str]:
    """Get entity IDs that are directly connected to the specified entity."""
    # Find the entity ID by name (case-insensitive partial match)
    target_entity_id = None
    entity_name_lower = entity_name.lower().replace('_', ' ')

    # First try exact ID match
    for node in nodes_data:
        if node['id'].lower() == entity_name_lower:
            target_entity_id = node['id']
            break

    # If not found, try partial text match
    if not target_entity_id:
        for node in nodes_data:
            node_text = node.get('text', '').lower()
            if entity_name_lower in node_text or node_text in entity_name_lower:
                target_entity_id = node['id']
                break

    if not target_entity_id:
        print(f"Entity '{entity_name}' not found in graph")
        return []

    # Find all directly connected entities
    connected_entities = {target_entity_id}  # Include the target entity itself

    for edge in edges_data:
        if edge['source'] == target_entity_id:
            connected_entities.add(edge['target'])
        elif edge['target'] == target_entity_id:
            connected_entities.add(edge['source'])

    return list(connected_entities)

def visualize_graph(
    nodes_path: Path = Path("output/graph_nodes.json"),
    edges_path: Path = Path("output/graph_edges.json"),
    pagerank_path: Path = Path("output/pagerank_results.json"),
    output_image_path: Path = Path("output/knowledge_graph_visualization.png"),
    node_types: Optional[List[str]] = None,
    top_nodes: Optional[int] = None,
    min_score: Optional[float] = None,
    significant_only: bool = False,
    labels: Optional[List[str]] = None,
    entity_view: Optional[str] = None,
):
    """
    Visualizes the knowledge graph with nodes colored by type and sized by PageRank score.

    Args:
        nodes_path: Path to the graph_nodes.json file.
        edges_path: Path to the graph_edges.json file.
        pagerank_path: Path to the pagerank_results.json file.
        output_image_path: Path to save the visualization image.
        node_types: List of node types to include (e.g., ['Entity', 'Event', 'Location']).
        top_nodes: Number of top PageRank nodes to show.
        min_score: Minimum PageRank score threshold.
        significant_only: Show only significant nodes.
        labels: List of node labels to include.
        entity_view: Entity name to show neighborhood view (shows the entity and all directly connected entities).
    """
    if not nodes_path.exists() or not edges_path.exists():
        print(f"Error: Node file '{nodes_path}' or edge file '{edges_path}' not found.")
        return

    with nodes_path.open("r", encoding="utf-8") as f:
        nodes_data = json.load(f)
    with edges_path.open("r", encoding="utf-8") as f:
        edges_data = json.load(f)

    # Load PageRank results if available
    pagerank_scores = {}
    significant_nodes = set()
    if pagerank_path.exists():
        with pagerank_path.open("r", encoding="utf-8") as f:
            pagerank_data = json.load(f)
            for node_entry in pagerank_data["nodes"]:
                pagerank_scores[node_entry["id"]] = node_entry["score"]
                if node_entry.get("is_significant", False):
                    significant_nodes.add(node_entry["id"])

    # First, identify connected nodes (nodes that have at least one edge)
    connected_node_ids = set()
    for edge in edges_data:
        connected_node_ids.add(edge["source"])
        connected_node_ids.add(edge["target"])

    # Apply filters to nodes
    filtered_nodes = []
    entity_view_ids = get_entity_neighborhood(entity_view, nodes_data, edges_data) if entity_view else None

    for node in nodes_data:
        node_id = node["id"]
        node_type = node.get("type", "")
        node_label = node.get("label", "")
        pagerank_score = pagerank_scores.get(node_id, 0.0)

        # Filter out isolated nodes (nodes with no connections)
        if node_id not in connected_node_ids:
            continue

        # Filter by entity view (predefined sets)
        if entity_view_ids is not None and node_id not in entity_view_ids:
            continue

        # Filter by node types
        if node_types and node_type not in node_types:
            continue

        # Filter by labels
        if labels and node_label not in labels:
            continue

        # Filter by significance
        if significant_only and node_id not in significant_nodes:
            continue

        # Filter by minimum score
        if min_score is not None and pagerank_score < min_score:
            continue

        filtered_nodes.append(node)

    # If top_nodes is specified, sort by PageRank and take top N
    if top_nodes:
        filtered_nodes.sort(key=lambda x: pagerank_scores.get(x["id"], 0.0), reverse=True)
        filtered_nodes = filtered_nodes[:top_nodes]

    print(f"Filtered to {len(filtered_nodes)} nodes from {len(nodes_data)} total")

    G = nx.DiGraph()

    # Add nodes with attributes
    for node in filtered_nodes:
        pagerank_score = pagerank_scores.get(node["id"], 0.0)
        G.add_node(node["id"], type=node.get("type", "Unknown"), label=node.get("text", node["id"]), pagerank=pagerank_score)

    # Add edges with attributes (only between filtered nodes)
    filtered_node_ids = {node["id"] for node in filtered_nodes}
    for edge in edges_data:
        if edge["source"] in filtered_node_ids and edge["target"] in filtered_node_ids:
            G.add_edge(edge["source"], edge["target"], type=edge["edge_type"])

    # Prepare for visualization
    plt.figure(figsize=(18, 15))
    pos = nx.spring_layout(G, k=0.3, iterations=50) # Use spring layout for better node distribution

    node_colors = []
    node_sizes = []
    node_labels = {}

    # Define color map for node types
    color_map = {
        "Entity": "skyblue",
        "Event": "lightcoral",
        "Location": "lightgreen",
    }

    for node_id in G.nodes():
        node_attr = G.nodes[node_id]
        node_colors.append(color_map.get(node_attr["type"], "gray"))
        
        # Scale node size by PageRank score
        # Normalize PageRank scores for better visualization, handle cases with no pagerank
        pagerank_score = node_attr.get("pagerank", 0.0)
        node_sizes.append(500 + pagerank_score * 10000) # Base size + scaled pagerank

        node_labels[node_id] = node_attr["label"]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray", alpha=0.6)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight="bold")

    # Draw edge labels (relation types)
    edge_labels = nx.get_edge_attributes(G, "type")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7) # Removed 'color' parameter

    # Create dynamic title based on filters
    title_parts = ["Knowledge Graph Visualization"]
    if entity_view:
        title_parts.append(f"View: {entity_view.title()}")
    if node_types:
        title_parts.append(f"Types: {','.join(node_types)}")
    if top_nodes:
        title_parts.append(f"Top {top_nodes} Nodes")
    if min_score is not None:
        title_parts.append(f"Min Score: {min_score}")
    if significant_only:
        title_parts.append("Significant Only")
    if labels:
        title_parts.append(f"Labels: {','.join(labels)}")

    plt.title(" | ".join(title_parts), size=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_image_path, format="png", dpi=300)
    print(f"Knowledge graph visualization saved to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize knowledge graph with filtering options")
    parser.add_argument("--nodes", type=str, default="output/graph_nodes.json",
                        help="Path to nodes JSON file")
    parser.add_argument("--edges", type=str, default="output/graph_edges.json",
                        help="Path to edges JSON file")
    parser.add_argument("--pagerank", type=str, default="output/pagerank_results.json",
                        help="Path to PageRank results JSON file")
    parser.add_argument("--output", type=str, default="output/knowledge_graph_visualization.png",
                        help="Output image path")
    parser.add_argument("--node-types", type=str,
                        help="Comma-separated list of node types to include (e.g., Entity,Event,Location)")
    parser.add_argument("--top-nodes", type=int,
                        help="Show only top N PageRank nodes")
    parser.add_argument("--min-score", type=float,
                        help="Minimum PageRank score threshold")
    parser.add_argument("--significant-only", action="store_true",
                        help="Show only significant nodes")
    parser.add_argument("--labels", type=str,
                        help="Comma-separated list of node labels to include")
    parser.add_argument("--entity-view", type=str,
                        help="Entity name to show neighborhood view (shows entity and directly connected entities)")

    args = parser.parse_args()

    # Parse comma-separated arguments
    node_types = args.node_types.split(',') if args.node_types else None
    labels = args.labels.split(',') if args.labels else None

    visualize_graph(
        nodes_path=Path(args.nodes),
        edges_path=Path(args.edges),
        pagerank_path=Path(args.pagerank),
        output_image_path=Path(args.output),
        node_types=node_types,
        top_nodes=args.top_nodes,
        min_score=args.min_score,
        significant_only=args.significant_only,
        labels=labels,
        entity_view=args.entity_view,
    )