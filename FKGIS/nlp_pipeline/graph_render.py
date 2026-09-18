"""Paper-quality knowledge graph rendering.

This module produces two kinds of output from the raw ``graph_nodes.json`` /
``graph_edges.json`` artifacts:

1. A high-resolution static render (PNG at 300 DPI and/or vector SVG) suitable
   for figures in a research paper. The layout is community-aware and tuned for
   readability: nodes are grouped by detected community, colored by node type,
   and sized by PageRank importance. By default it renders an investigator-
   facing view -- direct entity-to-entity/location relations only (no Event/
   Time scaffolding nodes), trimmed to the most salient entities -- rather
   than a full dump of every extracted fact.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx

from .kg.graph_builder import RELATION_TYPE_TO_EDGE
from .kg.lead_signals import compute_lead_signals

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

# Edge types that are structural scaffolding -- every event gets these
# automatically (who was in it, where, when, its place in the timeline).
# They're what holds the graph together, not a lead, so they're always drawn
# (never suppressed -- an entity linked only by PARTICIPATED_IN should not
# look orphaned) but kept visually quiet.
STRUCTURAL_EDGE_TYPES = {"PARTICIPATED_IN", "LOCATED_AT", "HAS_TIME", "BEFORE", "AFTER"}

# Edge types the relation extractor mapped to a recognized semantic label
# (WITNESSED, INTERVIEWED, OWNS, ...) -- these came from real narrative text
# and got curated, so they're the actual investigative leads.
CURATED_EDGE_TYPES = set(RELATION_TYPE_TO_EDGE.values()) - STRUCTURAL_EDGE_TYPES

EdgeTier = str  # "structural" | "investigative" | "generic"

# Per-tier visual weight. "generic" edges are still real, entity-grounded
# relations (an uncurated verb lemma, e.g. HAVE/BE/GO) -- drawn, just faint,
# so they don't compete visually with the curated leads.
EDGE_TIER_STYLE: Dict[EdgeTier, Dict[str, Any]] = {
    "investigative": {"color": "#5C7CAF", "linewidth": 1.3, "alpha": 0.85, "zorder": 2, "vis_color": "#5C7CAF", "vis_opacity": 0.85},
    "structural": {"color": "#DCE1E8", "linewidth": 0.5, "alpha": 0.45, "zorder": 1, "vis_color": "#D3D9E0", "vis_opacity": 0.35},
    "generic": {"color": "#C7CDD6", "linewidth": 0.6, "alpha": 0.5, "zorder": 1, "vis_color": "#B9C0CA", "vis_opacity": 0.45},
}


def edge_tier(edge_type: str) -> EdgeTier:
    """Classify an edge type into a visual-prominence tier.

    - "structural": scaffolding produced by every event (who/where/when/order).
    - "investigative": a relation with a recognized semantic label -- the
      actual leads (WITNESSED, INTERVIEWED, OWNS, ...).
    - "generic": a real, entity-grounded relation whose verb has no curated
      label, so it falls back to the uppercased lemma (HAVE, BE, GO, ...).
    """
    if edge_type in STRUCTURAL_EDGE_TYPES:
        return "structural"
    if edge_type in CURATED_EDGE_TYPES:
        return "investigative"
    return "generic"


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


# Node types that represent scaffolding (an event's own record, and the
# timestamp it's pinned to) rather than a case actor/location. The default,
# investigator-facing view excludes them: "who connects to whom" reads far
# better as direct entity-to-entity edges than as two-hop chains routed
# through an event node.
NON_ENTITY_NODE_TYPES = {"Event", "Time"}


def build_graph(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    drop_isolates: bool = True,
    entity_only: bool = True,
) -> nx.Graph:
    """Build an undirected graph with node/edge attributes attached.

    When entity_only is True (the default), Event/Time nodes -- and any edge
    touching one (PARTICIPATED_IN, HAS_TIME, BEFORE/AFTER, event-routed
    LOCATED_AT) -- are excluded. Direct entity-to-entity and entity-to-location
    relations (e.g. INTERVIEWED, OWNS, ARRIVED_AT) are unaffected, since those
    already connect real entities/locations directly.
    """
    G = nx.Graph()

    node_attrs: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        node_type = node.get("type") or "Entity"
        if entity_only and node_type in NON_ENTITY_NODE_TYPES:
            continue
        attrs = dict(node)
        attrs["type"] = node_type
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


def _rank_nodes(G: nx.Graph, pagerank: Dict[str, float]) -> List[Any]:
    """Rank nodes by salience: PageRank score when available, else degree."""
    def score(n: Any) -> float:
        return pagerank.get(n, 0.0) or float(G.degree(n))

    return sorted(G.nodes, key=score, reverse=True)


def _trim_to_salient(G: nx.Graph, pagerank: Dict[str, float], max_nodes: int) -> nx.Graph:
    """Keep only the top `max_nodes` most salient nodes and re-drop isolates.

    A full case graph can legitimately have hundreds of real entities; that's
    still too dense to read as a static figure. This keeps the figure to the
    entities that actually matter, while the full data remains available in
    the underlying JSON / interactive view.
    """
    if G.number_of_nodes() <= max_nodes:
        return G
    keep = set(_rank_nodes(G, pagerank)[:max_nodes])
    trimmed = G.subgraph(keep).copy()
    trimmed.remove_nodes_from(list(nx.isolates(trimmed)))
    return trimmed


def _truncate_label(label: str, max_len: int = 28) -> str:
    """Shorten an overly long node label for display (full text stays
    available in the underlying data/tooltips) -- a mis-resolved entity
    shouldn't dominate the figure with a full sentence fragment."""
    label = str(label)
    return label if len(label) <= max_len else label[: max_len - 1].rstrip() + "…"


def _declutter_labels(fig, annotations: Sequence[Any], iterations: int = 80, push: float = 1.6) -> None:
    """Greedily nudge overlapping label bounding boxes apart, in display
    (pixel) space. A lightweight, dependency-free stand-in for a real label-
    placement solver: not perfect, but removes most direct label-on-label
    stacking in a moderately dense figure. Cosmetic only -- any failure here
    is swallowed so it can never break the render.
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for _ in range(iterations):
            moved = False
            boxes = [a.get_window_extent(renderer=renderer) for a in annotations]
            for i in range(len(annotations)):
                for j in range(i + 1, len(annotations)):
                    bi, bj = boxes[i], boxes[j]
                    if not bi.overlaps(bj):
                        continue
                    dx = (bj.x0 + bj.x1) / 2 - (bi.x0 + bi.x1) / 2
                    dy = (bj.y0 + bj.y1) / 2 - (bi.y0 + bi.y1) / 2
                    if dx == 0 and dy == 0:
                        dx, dy = 1.0, 1.0
                    norm = (dx ** 2 + dy ** 2) ** 0.5
                    ux, uy = dx / norm, dy / norm
                    oxi, oyi = annotations[i].xyann
                    oxj, oyj = annotations[j].xyann
                    annotations[i].xyann = (oxi - ux * push, oyi - uy * push)
                    annotations[j].xyann = (oxj + ux * push, oyj + uy * push)
                    moved = True
            if not moved:
                break
    except Exception:
        pass


def _scale_size(pagerank: float, pageranks: Sequence[float], scale: float, base: int = 200) -> int:
    """Map a PageRank score onto a sensible node size using a log scale."""
    if pagerank <= 0 or not pageranks:
        return base
    max_score = max(pageranks)
    if max_score <= 0:
        return base
    ratio = pagerank / max_score
    return int(base + scale * ratio)


def _local_layout(sub: nx.Graph, local_seed: int) -> Dict[Any, Tuple[float, float]]:
    """Kamada-Kawai gives a good starting arrangement (it directly minimizes
    stress between graph distance and drawn distance, so it rarely tangles),
    but on a graph with long pendant chains (a leaf connected only through a
    single path back to the core) it can stretch those chains far from
    everything else while satisfying that objective. A short spring-layout
    refinement seeded from the Kamada-Kawai result pulls those stray chains
    back in via ordinary edge attraction, without discarding the good global
    arrangement KK already found.
    """
    if sub.number_of_nodes() == 1:
        return {next(iter(sub.nodes)): (0.0, 0.0)}
    try:
        kk_pos = nx.kamada_kawai_layout(sub)
    except Exception:
        return nx.spring_layout(
            sub, seed=local_seed, k=1.3 / max(1, sub.number_of_nodes() ** 0.5), iterations=200
        )
    if sub.number_of_nodes() <= 3:
        return kk_pos
    try:
        return nx.spring_layout(
            sub, pos=kk_pos, seed=local_seed,
            k=1.1 / max(1, sub.number_of_nodes() ** 0.5), iterations=60,
        )
    except Exception:
        return kk_pos


def _bbox(positions: Dict[Any, Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """(min_x, min_y, width, height) of a position dict, width/height >= 0.6
    so a single-node "box" still reserves a sane amount of room."""
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    min_x, max_x = (min(xs), max(xs)) if xs else (0.0, 0.0)
    min_y, max_y = (min(ys), max(ys)) if ys else (0.0, 0.0)
    return min_x, min_y, max(0.6, max_x - min_x), max(0.6, max_y - min_y)


def _layout_connected(G: nx.Graph, seed: int) -> Dict[Any, Tuple[float, float]]:
    """Lay out one connected graph: communities placed by how connected they
    are to each other (a meta-graph spring layout, weighted by inter-community
    edge count), each community's own members placed with Kamada-Kawai.
    """
    if G.number_of_nodes() <= 2:
        return nx.spring_layout(G, seed=seed, k=0.5, iterations=100)

    communities = list(nx.community.greedy_modularity_communities(G))
    if len(communities) <= 1:
        return _local_layout(G, seed)

    meta = nx.Graph()
    comm_of: Dict[Any, int] = {}
    for idx, community in enumerate(communities):
        meta.add_node(idx)
        for node in community:
            comm_of[node] = idx
    for u, v in G.edges():
        cu, cv = comm_of[u], comm_of[v]
        if cu != cv:
            if meta.has_edge(cu, cv):
                meta[cu][cv]["weight"] += 1
            else:
                meta.add_edge(cu, cv, weight=1)

    meta_pos = nx.spring_layout(meta, seed=seed, k=2.2, iterations=300, weight="weight")
    meta_spread = max(3.0, len(communities) * 1.3)

    pos: Dict[Any, Tuple[float, float]] = {}
    for idx, community in enumerate(communities):
        sub = G.subgraph(community)
        local = _local_layout(sub, seed + idx)
        scale = max(0.7, (len(sub) ** 0.5) * 0.6)
        cx, cy = meta_pos[idx]
        cx *= meta_spread
        cy *= meta_spread
        for node, (x, y) in local.items():
            pos[node] = (cx + x * scale, cy + y * scale)

    return pos


def community_layout(G: nx.Graph, seed: int = 7) -> Dict[Any, Tuple[float, float]]:
    """Position every node, packing genuinely disconnected components tightly
    instead of scattering them across the canvas.

    A case graph is often one large connected component plus a handful of
    small disconnected pairs/triples (an aside fact with no link to the main
    story). Spring-layout has no attractive force between components that
    share no edge, so laying the whole graph out at once can fling those tiny
    components arbitrarily far away, wasting most of the canvas on empty
    space and squeezing everything that actually matters into a corner. Here,
    the largest connected component gets the full community-aware treatment
    (see _layout_connected); every smaller component is laid out on its own
    and packed into a compact row directly above it, since there's no
    meaningful "near/far" to preserve between pieces that don't connect.
    """
    if G.number_of_nodes() <= 2:
        return nx.spring_layout(G, seed=seed, k=0.5, iterations=100)

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    main_component = components[0]
    pos = _layout_connected(G.subgraph(main_component), seed)
    if len(components) == 1:
        return pos

    min_x, min_y, main_w, main_h = _bbox(pos)
    cursor_x = min_x
    row_y = min_y + main_h + 1.2
    row_gap = max(0.5, main_w * 0.05)

    for i, comp in enumerate(components[1:]):
        sub = G.subgraph(comp)
        local = _local_layout(sub, seed + 100 + i)
        lx, ly, lw, lh = _bbox(local)
        ox, oy = cursor_x - lx, row_y - ly
        for node, (x, y) in local.items():
            pos[node] = (x + ox, y + oy)
        cursor_x += lw + row_gap

    return pos


def build_vis_payload(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    pagerank: Optional[Dict[str, float]] = None,
    entity_only: bool = False,
) -> Dict[str, Any]:
    """Build a vis-network friendly payload with pre-computed styling.

    entity_only defaults to False here (unlike render_static): this payload
    feeds an interactive viewer that already lets a user toggle Entity/Event/
    Location/Time visibility live, so the full graph is sent by default.

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
        if entity_only and node_type in NON_ENTITY_NODE_TYPES:
            continue
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
        tier = edge_tier(edge_type)
        style = EDGE_TIER_STYLE[tier]
        vis_edges.append(
            {
                "from": source,
                "to": target,
                "label": edge_type if tier == "investigative" else "",
                "title": f"<b>{edge_type}</b> ({tier})<br>{source} → {target}",
                "color": {"color": style["vis_color"], "highlight": "#C44E52", "opacity": style["vis_opacity"]},
                "width": 2.2 if tier == "investigative" else 1,
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
    entity_only: bool = True,
    max_nodes: Optional[int] = 45,
    label_edges: bool = True,
    label_important_edges_only: bool = True,
    highlight_leads: bool = True,
    dpi: int = 300,
    seed: int = 7,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Render the knowledge graph to a publication-quality image.

    By default this produces an investigator-facing view: only entities and
    locations (no Event/Time scaffolding nodes), trimmed to the `max_nodes`
    most salient ones by PageRank/degree. Pass entity_only=False and/or
    max_nodes=None for the full, unfiltered graph.

    When highlight_leads is True (default), nodes flagged by
    kg.lead_signals.compute_lead_signals (hubs, connectors, cross-document
    corroboration, multi-claim overlaps) get a highlight ring and a "points of
    interest" panel on the figure. These are structural heuristics, not
    conclusions -- they mark what's worth a second look, not what's true.

    The output format is inferred from the file extension (PNG, SVG, PDF, ...).
    Returns (path_to_written_file, lead_signals) -- lead_signals is the same
    list rendered in the panel, for callers that want it (e.g. to print or
    log) without re-parsing the figure.
    """
    output_path = Path(output_path)
    pagerank = pagerank or {}
    G = build_graph(nodes, edges, drop_isolates=drop_isolates, entity_only=entity_only)
    if max_nodes is not None:
        G = _trim_to_salient(G, pagerank, max_nodes)
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty after filtering; nothing to render.")

    scores = [s for s in pagerank.values() if s and s > 0]

    lead_signals: List[Dict[str, Any]] = []
    if highlight_leads:
        lead_signals = compute_lead_signals(G, edges, pagerank=pagerank, top_n=10)
    flagged_ids = {ls["id"] for ls in lead_signals}

    pos = community_layout(G, seed=seed)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(24, 19))
    ax.axis("off")
    # Reserve a footer strip outside the data area for the points-of-interest
    # panel, so it can never land on top of a node/label no matter where the
    # layout happens to place things.
    if highlight_leads:
        fig.subplots_adjust(bottom=0.10)

    # Draw edges first (under nodes). Every edge is drawn -- including
    # structural ones -- so a node connected only by scaffolding (e.g. an
    # entity linked to an event via PARTICIPATED_IN) doesn't look orphaned.
    # Visual weight is what signals importance, not omission.
    edge_types = nx.get_edge_attributes(G, "edge_type")
    for (u, v), etype in edge_types.items():
        style = EDGE_TIER_STYLE[edge_tier(etype)]
        ax.plot(
            *zip(pos[u], pos[v]),
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=style["zorder"],
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

    # Highlight ring for flagged "points of interest" -- drawn as a hollow
    # outline so it doesn't obscure the node's own type color/shape.
    if flagged_ids:
        flagged_present = [n for n in G.nodes if n in flagged_ids]
        ring_sizes = [
            _scale_size(pagerank.get(n, 0.0), scores, NODE_STYLE.get(node_type_by_id[n], DEFAULT_NODE_STYLE)["size_scale"])
            + 260
            for n in flagged_present
        ]
        ax.scatter(
            [pos[n][0] for n in flagged_present],
            [pos[n][1] for n in flagged_present],
            s=ring_sizes,
            facecolors="none",
            edgecolors="#C0392B",
            linewidths=1.8,
            zorder=2,
        )

    # Node labels (font small, but readable at print resolution). Flagged
    # nodes get a bold, marked label so they stand out from the text soup.
    # Long labels are truncated for display -- a rare mis-resolved entity
    # shouldn't dominate the figure with a full sentence fragment.
    label_annotations: List[Any] = []
    for node in G.nodes:
        is_flagged = node in flagged_ids
        label = _truncate_label(G.nodes[node].get("label", node))
        ann = ax.annotate(
            f"★ {label}" if is_flagged else label,
            xy=pos[node],
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.5 if is_flagged else 7.5,
            fontweight="bold" if is_flagged else "normal",
            color="#8B1A1A" if is_flagged else "#1B2631",
            zorder=5 if is_flagged else 3,
        )
        label_annotations.append(ann)

    # Spread apart any node labels that ended up overlapping -- the layout
    # change above reduces this a lot, but dense pockets can still collide.
    _declutter_labels(fig, label_annotations)

    # Edge labels for semantically important relationships only
    if label_edges:
        drawn = set()
        for (u, v), etype in edge_types.items():
            if label_important_edges_only and edge_tier(etype) != "investigative":
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

    # Legend: node types, plus edge tiers so readers know why some lines are
    # bold/labeled (investigative leads) and others are faint (scaffolding or
    # low-signal relations).
    legend_handles = [
        Patch(
            facecolor=NODE_STYLE[t]["color"],
            label=f"{t} ({sum(1 for k in node_type_by_id if node_type_by_id[k] == t)})",
        )
        for t in sorted({v for v in node_type_by_id.values()})
    ]
    edge_tier_counts: Dict[str, int] = {"investigative": 0, "structural": 0, "generic": 0}
    for etype in edge_types.values():
        edge_tier_counts[edge_tier(etype)] += 1
    edge_tier_labels = {
        "investigative": "investigative relation (labeled)",
        "structural": "structural (participant/location/time/order)",
        "generic": "generic relation (uncurated verb)",
    }
    for tier in ("investigative", "structural", "generic"):
        if edge_tier_counts[tier] == 0:
            continue
        style = EDGE_TIER_STYLE[tier]
        legend_handles.append(
            Line2D(
                [0], [0],
                color=style["color"],
                linewidth=max(style["linewidth"], 1.5),
                alpha=max(style["alpha"], 0.7),
                label=f"{edge_tier_labels[tier]} ({edge_tier_counts[tier]})",
            )
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", frameon=True, fontsize=10)

    # Points-of-interest panel: what's flagged and why, in plain language.
    # These are structural heuristics (hub/connector/corroborated/multi-claim),
    # not conclusions -- worth a second look, not proof of anything.
    if lead_signals:
        lines = ["Points of interest (structural heuristics, not conclusions):  "]
        for ls in lead_signals[:8]:
            top_reason = ls["reasons"][0] if ls["reasons"] else ""
            lines.append(f"★ {ls['label']} — {top_reason}")
        # Figure coordinates, not axes coordinates: this lives in the footer
        # strip reserved above, entirely outside the plotted network, so it
        # cannot overlap a node no matter where the layout puts things.
        fig.text(
            0.5, 0.01,
            "     ".join(lines),
            transform=fig.transFigure,
            ha="center", va="bottom",
            fontsize=8,
            color="#4A2020",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FBEAEA", edgecolor="#C0392B", alpha=0.9),
        )

    ax.set_title(
        f"Knowledge Graph — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        fontsize=16,
        pad=20,
    )

    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return str(output_path), lead_signals


def export_figure(
    nodes_path: str | Path,
    edges_path: str | Path,
    pagerank_path: Optional[str | Path] = None,
    output_path: str | Path = "knowledge_graph_figure.png",
    **kwargs: Any,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Convenience wrapper: load artifacts from disk and render them.

    Returns (path_to_written_file, lead_signals) -- see render_static.
    """
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
    parser.add_argument("--include-events", action="store_true", help="Include Event/Time scaffolding nodes (default: entities/locations only)")
    parser.add_argument("--max-nodes", type=int, default=45, help="Trim to the N most salient nodes (0 = no trim)")
    parser.add_argument("--no-edge-labels", action="store_true", help="Do not draw any edge labels")
    parser.add_argument("--label-all-edges", action="store_true", help="Label every edge (cluttered)")
    parser.add_argument("--dpi", type=int, default=300, help="Raster resolution (PNG)")
    parser.add_argument("--seed", type=int, default=7, help="Layout random seed")
    parser.add_argument("--no-leads", action="store_true", help="Do not compute/highlight points of interest")
    args = parser.parse_args()

    out, lead_signals = export_figure(
        args.nodes,
        args.edges,
        args.pagerank,
        output_path=args.output,
        drop_isolates=not args.keep_isolates,
        entity_only=not args.include_events,
        max_nodes=(args.max_nodes or None),
        label_edges=not args.no_edge_labels,
        label_important_edges_only=not args.label_all_edges,
        highlight_leads=not args.no_leads,
        dpi=args.dpi,
        seed=args.seed,
    )
    print(f"Figure saved to {out}")
    if lead_signals:
        print("\nPoints of interest (structural heuristics, not conclusions):")
        for ls in lead_signals:
            print(f"  * {ls['label']} (score {ls['score']})")
            for reason in ls["reasons"]:
                print(f"      - {reason}")


if __name__ == "__main__":
    main()