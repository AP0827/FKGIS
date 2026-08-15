"""Paper-quality knowledge graph rendering.

This module produces two kinds of output from the raw ``graph_nodes.json`` /
``graph_edges.json`` artifacts:

1. A high-resolution static render (PNG at 300 DPI and/or vector SVG) suitable
   for figures in a research paper. The layout is community-aware and tuned for
   readability: nodes are grouped by detected community, colored by node type,
   and sized by PageRank importance.
2. A JSON payload designed for an interactive browser visualizer (vis-network),
   with node/edge attributes, colors, sizes and tooltips pre-computed.

It is self-contained (no database or LLM required) and can also be run from the
command line:

    python -m FKGIS.nlp_pipeline.graph_render \
        --nodes output/graph_nodes.json \
        --edges output/graph_edges.json \
        --pagerank output/pagerank_results.json \
        --output output/kg_figure.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx

# ---------------------------------------------------------------------------
# Node / edge styling
# ---------------------------------------------------------------------------
NODE_STYLE: Dict[str, Dict[str, Any]] = {
    "Entity": {"color": "#4C72B0", "shape": "o", "size_scale": 2200},
    "Event": {"color": "#DD8452", "shape": "s", "size_scale": 1600},
    "Location": {"color": "#55A868", "shape": "D", "size_scale": 1800},
    "Time": {"color": "#C44E52", "shape": "v", "size_scale": 1200},
}
DEFAULT_NODE_STYLE = {"color": "#7F7F7F", "shape": "o", "size_scale": 1400}

# Edge types we consider semantically informative enough to label in the
# static figure. Everything else is drawn thin and gray to reduce clutter.
LABELED_EDGE_TYPES = {
    "PARTICIPATED_IN",
    "LOCATED_AT",
    "BEFORE",
    "AFTER",
    "HAS_TIME",
}
# Edge types that convey little information when rendered densely.
SUPPRESSED_EDGE_TYPES = {"PARTICIPATED_IN"}


def load_nodes_edges(
    nodes_path: str | Path,
    edges_path: str | Path,
    pagerank_path: Optional[str | Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    """Load node/edge lists plus PageRank scores if available."""
    with Path(nodes_path).open("r", encoding="utf-8") as f:
        nodes = json.load(f)
    with Path(edges_path).open("r", encoding="utf-8") as f:
        edges = json.load(f)

    pagerank: Dict[str, float] = {}
    if pagerank_path and Path(pagerank_path).exists():
        with Path(pagerank_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("nodes", []):
            pagerank[entry["id"]] = entry.get("score", 0.0)

    return nodes, edges, pagerank


def build_graph(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    drop_isolates: bool = True,
) -> nx.Graph:
    """Build an undirected graph with node/edge attributes attached."""
    G = nx.Graph()

    node_attrs: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        attrs = dict(node)
        attrs["type"] = node.get("type") or "Entity"
        attrs["label"] = node.get("text") or node.get("id") or node_id
        node_attrs[node_id] = attrs
        G.add_node(node_id, **attrs)

    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source in node_attrs and target in node_attrs:
            G.add_edge(source, target, edge_type=edge.get("edge_type") or "RELATED")

    if drop_isolates:
        G.remove_nodes_from(list(nx.isolates(G)))

    return G


def _scale_size(pagerank: float, pageranks: Sequence[float], scale: float, base: int = 200) -> int:
    """Map a PageRank score onto a sensible node size using a log scale."""
    if pagerank <= 0 or not pageranks:
        return base
    max_score = max(pageranks)
    if max_score <= 0:
        return base
    ratio = pagerank / max_score
    return int(base + scale * ratio)


def community_layout(G: nx.Graph, seed: int = 7) -> Dict[Any, Tuple[float, float]]:
    """Position nodes by community on concentric circles.

    Communities are detected with greedy modularity maximization. Each community
    is assigned a sector around a circle and its members are placed inside that
    sector with a seeded spring layout, which keeps related clusters visually
    distinct while remaining deterministic for reproducible figures.
    """
    communities = list(nx.community.greedy_modularity_communities(G))
    pos: Dict[Any, Tuple[float, float]] = {}
    if not communities:
        return nx.spring_layout(G, seed=seed, k=0.5, iterations=100)

    n = G.number_of_nodes()
    radius = max(1.0, n ** 0.5 / 4.0)
    angle_step = 2 * 3.141592653589793 / len(communities)

    for comm_idx, community in enumerate(communities):
        sub = G.subgraph(community)
        # Per-community spring layout, scaled to the community radius
        local = nx.spring_layout(sub, seed=seed + comm_idx, k=1.0 / max(1, len(sub) ** 0.5), iterations=100)
        if len(communities) == 1:
            return local

        center_angle = angle_step * comm_idx
        center_x = radius * 2.5 * (1.0 if comm_idx % 2 == 0 else 1.3) * max(0.5, (len(sub) / n))
        center_y = 0.0
        # Rotate community center around the ring
        cx = center_x * (1.0 if comm_idx % 2 == 0 else -1.0)
        cy = center_y + ((comm_idx - (len(communities) - 1) / 2.0) * 0.8)

        scale = max(1.0, (len(sub) ** 0.5) * 1.1)
        for node, (x, y) in local.items():
            pos[node] = (cx + x * scale, cy + y * scale)

    return pos


def build_vis_payload(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    pagerank: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build a vis-network friendly payload with pre-computed styling.

    Returns:
        {
          "nodes": [{"id", "label", "group", "value", "color", "shape", "title", ...}, ...],
          "edges": [{"from", "to", "label", "color", "title", ...}, ...],
          "stats": {...}
        }
    """
    pagerank = pagerank or {}
    scores = [s for s in pagerank.values() if s and s > 0]

    vis_nodes: List[Dict[str, Any]] = []
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        node_type = node.get("type") or "Entity"
        style = NODE_STYLE.get(node_type, DEFAULT_NODE_STYLE)
        score = pagerank.get(node_id, 0.0)
        vis_nodes.append(
            {
                "id": node_id,
                "label": node.get("text") or node_id,
                "group": node_type,
                "value": _scale_size(score, scores, 30, base=8),
                "color": style["color"],
                "shape": style["shape"],
                "title": (
                    f"<b>{node.get('text') or node_id}</b><br>"
                    f"type: {node_type}<br>"
                    f"label: {node.get('label') or '—'}<br>"
                    f"PageRank: {score:.5f}"
                ),
            }
        )

    node_ids = {n["id"] for n in vis_nodes}
    vis_edges: List[Dict[str, Any]] = []
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source not in node_ids or target not in node_ids:
            continue
        edge_type = edge.get("edge_type") or "RELATED"
        vis_edges.append(
            {
                "from": source,
                "to": target,
                "label": "" if edge_type in SUPPRESSED_EDGE_TYPES else edge_type,
                "title": f"<b>{edge_type}</b><br>{source} → {target}",
                "color": {"color": "#9AA5B1", "highlight": "#C44E52", "opacity": 0.6},
                "arrows": "to" if edge_type in {"BEFORE", "AFTER"} else "",
            }
        )

    stats = {
        "nodes": len(vis_nodes),
        "edges": len(vis_edges),
        "node_types": {
            t: sum(1 for n in vis_nodes if n["group"] == t) for t in sorted({n["group"] for n in vis_nodes})
        },
        "pagerank_top": sorted(
            ((pagerank.get(n["id"], 0.0), n["label"]) for n in vis_nodes),
            reverse=True,
        )[:10],
    }

    return {"nodes": vis_nodes, "edges": vis_edges, "stats": stats}


def render_static(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    pagerank: Optional[Dict[str, float]] = None,
    output_path: str | Path = "knowledge_graph_figure.png",
    drop_isolates: bool = True,
    label_edges: bool = True,
    label_important_edges_only: bool = True,
    dpi: int = 300,
    seed: int = 7,
) -> str:
    """Render the knowledge graph to a publication-quality image.

    The output format is inferred from the file extension (PNG, SVG, PDF, ...).
    Returns the path of the written file.
    """
    output_path = Path(output_path)
    G = build_graph(nodes, edges, drop_isolates=drop_isolates)
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty after filtering; nothing to render.")

    pagerank = pagerank or {}
    scores = [s for s in pagerank.values() if s and s > 0]

    pos = community_layout(G, seed=seed)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(20, 16))
    ax.axis("off")

    # Draw edges first (under nodes)
    edge_types = nx.get_edge_attributes(G, "edge_type")
    for (u, v), etype in edge_types.items():
        if etype in SUPPRESSED_EDGE_TYPES and label_important_edges_only:
            continue
        ax.plot(
            *zip(pos[u], pos[v]),
            color="#C7CDD6",
            linewidth=0.7,
            alpha=0.55,
            zorder=1,
        )

    # Draw nodes
    node_type_by_id = {node: G.nodes[node].get("type", "Entity") for node in G.nodes}
    for node_type in sorted({t for t in node_type_by_id.values()}):
        style = NODE_STYLE.get(node_type, DEFAULT_NODE_STYLE)
        members = [n for n in G.nodes if node_type_by_id[n] == node_type]
        sizes = [
            _scale_size(pagerank.get(n, 0.0), scores, style["size_scale"]) for n in members
        ]
        ax.scatter(
            [pos[n][0] for n in members],
            [pos[n][1] for n in members],
            s=sizes,
            c=style["color"],
            marker=style["shape"],
            alpha=0.9,
            edgecolors="white",
            linewidths=0.6,
            zorder=2,
            label=node_type,
        )

    # Node labels (font small, but readable at print resolution)
    for node in G.nodes:
        ax.annotate(
            G.nodes[node].get("label", node),
            xy=pos[node],
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#1B2631",
            zorder=3,
        )

    # Edge labels for semantically important relationships only
    if label_edges:
        drawn = set()
        for (u, v), etype in edge_types.items():
            if label_important_edges_only and etype not in LABELED_EDGE_TYPES:
                continue
            key = tuple(sorted((u, v)))
            if key in drawn:
                continue
            drawn.add(key)
            mid = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
            ax.annotate(
                etype,
                xy=mid,
                xytext=(0, 2),
                textcoords="offset points",
                fontsize=5.5,
                color="#5D6D7E",
                ha="center",
                zorder=4,
            )

    # Legend
    legend_handles = [
        Patch(
            facecolor=NODE_STYLE[t]["color"],
            label=f"{t} ({sum(1 for k in node_type_by_id if node_type_by_id[k] == t)})",
        )
        for t in sorted({v for v in node_type_by_id.values()})
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=11)

    ax.set_title(
        f"Knowledge Graph — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        fontsize=16,
        pad=20,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def export_figure(
    nodes_path: str | Path,
    edges_path: str | Path,
    pagerank_path: Optional[str | Path] = None,
    output_path: str | Path = "knowledge_graph_figure.png",
    **kwargs: Any,
) -> str:
    """Convenience wrapper: load artifacts from disk and render them."""
    nodes, edges, pagerank = load_nodes_edges(nodes_path, edges_path, pagerank_path)
    return render_static(nodes, edges, pagerank, output_path=output_path, **kwargs)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render a paper-quality knowledge graph figure.")
    parser.add_argument("--nodes", required=True, help="Path to graph_nodes.json")
    parser.add_argument("--edges", required=True, help="Path to graph_edges.json")
    parser.add_argument("--pagerank", help="Optional path to pagerank_results.json")
    parser.add_argument("--output", default="knowledge_graph_figure.png", help="Output image path (.png/.svg/.pdf)")
    parser.add_argument("--keep-isolates", action="store_true", help="Do not drop isolated nodes")
    parser.add_argument("--no-edge-labels", action="store_true", help="Do not draw any edge labels")
    parser.add_argument("--label-all-edges", action="store_true", help="Label every edge (cluttered)")
    parser.add_argument("--dpi", type=int, default=300, help="Raster resolution (PNG)")
    parser.add_argument("--seed", type=int, default=7, help="Layout random seed")
    args = parser.parse_args()

    out = export_figure(
        args.nodes,
        args.edges,
        args.pagerank,
        output_path=args.output,
        drop_isolates=not args.keep_isolates,
        label_edges=not args.no_edge_labels,
        label_important_edges_only=not args.label_all_edges,
        dpi=args.dpi,
        seed=args.seed,
    )
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()