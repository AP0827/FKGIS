"""Heuristic "points of interest" signals over a case knowledge graph.

These are structural / corroboration heuristics computed purely from the
graph itself -- centrality, bridging, cross-document corroboration, and
conflicting claims. They are not proof of anything and not a substitute for
investigative judgment; they exist to point a reader at nodes worth a second
look, each with a plain-language reason attached.

Deliberately case-agnostic: nothing here references specific names, roles, or
verbs from any one case. It only uses graph structure and the doc_name/
edge_type metadata every case graph already carries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
from collections import defaultdict

import networkx as nx


def compute_lead_signals(
    G: nx.Graph,
    edges: Sequence[Dict[str, Any]],
    pagerank: Dict[str, float] | None = None,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Rank nodes by how much they're worth a second look, and say why.

    Signals (each purely structural/metadata-driven, no case-specific rules):

    - hub: unusually high PageRank/degree relative to the rest of the graph --
      a lot of the case's extracted facts run through this entity.
    - connector: high betweenness centrality -- this entity bridges two parts
      of the case that would otherwise be unconnected. An articulation point
      (removing the node disconnects the graph outright) is called out
      explicitly as the strongest form of this signal.
    - corroborated: relations touching this entity come from multiple
      independent source documents -- more than one witness/report puts them
      in the picture.
    - multi-claim: two or more distinct entities are each asserted to have the
      same relationship to this entity (e.g. two different people "OWNS" the
      same item, or three different "ARRIVED_AT" claims for the same place).
      This is deliberately neutral -- from graph structure alone there's no
      way to tell whether the underlying accounts actually agree (three
      officers who did all arrive at the scene) or conflict (two people who
      can't both own the same item). It just means: multiple independent
      claims land on the same fact, so it's worth reading them side by side.

    Returns a list of {"id", "label", "score", "reasons": [str, ...]},
    sorted by score descending, longest list length top_n.
    """
    if G.number_of_nodes() == 0:
        return []

    pagerank = pagerank or {}
    n = G.number_of_nodes()

    # --- hub ---
    degree = dict(G.degree())
    max_degree = max(degree.values()) if degree else 1
    hub_score: Dict[Any, float] = {}
    for node in G.nodes:
        pr = pagerank.get(node, 0.0)
        deg_norm = degree.get(node, 0) / max_degree if max_degree else 0.0
        hub_score[node] = pr if pr else deg_norm

    # --- connector (betweenness centrality + articulation points) ---
    betweenness = nx.betweenness_centrality(G) if n > 2 else {node: 0.0 for node in G.nodes}
    articulation = set(nx.articulation_points(G)) if n > 2 and nx.is_connected(G) else set()
    if n > 2 and not nx.is_connected(G):
        # Graph has multiple components -- articulation points are only
        # meaningful within each connected component.
        for component in nx.connected_components(G):
            sub = G.subgraph(component)
            if sub.number_of_nodes() > 2:
                articulation |= set(nx.articulation_points(sub))

    # --- corroborated (distinct source documents touching this entity) ---
    docs_by_node: Dict[Any, set] = defaultdict(set)
    for edge in edges:
        doc = edge.get("doc_name")
        if not doc:
            continue
        for side in (edge.get("source"), edge.get("target")):
            if side in G.nodes:
                docs_by_node[side].add(doc)

    # --- multi-claim (same edge_type into the same target from >1 source) ---
    claims: Dict[tuple, set] = defaultdict(set)
    for edge in edges:
        src, tgt, etype = edge.get("source"), edge.get("target"), edge.get("edge_type")
        if src in G.nodes and tgt in G.nodes and etype:
            claims[(tgt, etype)].add(src)
    multi_claim_by_node: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for (tgt, etype), sources in claims.items():
        if len(sources) > 1:
            multi_claim_by_node[tgt].append({"edge_type": etype, "sources": sorted(sources)})

    results: List[Dict[str, Any]] = []
    for node in G.nodes:
        reasons: List[str] = []
        score = 0.0

        hs = hub_score.get(node, 0.0)
        if hs > 0:
            score += hs * 1.0
            if hs >= sorted(hub_score.values(), reverse=True)[min(4, len(hub_score) - 1)]:
                reasons.append("hub: unusually central, many facts connect through this entity")

        bc = betweenness.get(node, 0.0)
        if node in articulation:
            score += 1.0
            reasons.append("connector: bridges parts of the case that would otherwise be disconnected")
        elif bc > 0 and bc >= sorted(betweenness.values(), reverse=True)[min(4, len(betweenness) - 1)]:
            score += bc * 0.8
            reasons.append("connector: sits between otherwise-separate clusters of the case")

        n_docs = len(docs_by_node.get(node, set()))
        if n_docs >= 2:
            score += 0.3 * n_docs
            reasons.append(f"corroborated: appears across {n_docs} independent source documents")

        multi_claim = multi_claim_by_node.get(node)
        if multi_claim:
            score += 0.4 * len(multi_claim)
            for c in multi_claim:
                who = ", ".join(G.nodes[s].get("label", s) for s in c["sources"])
                reasons.append(
                    f"multi-claim: {len(c['sources'])} independent sources each assert "
                    f"{c['edge_type']} here ({who}) -- worth reading side by side"
                )

        if reasons:
            results.append(
                {
                    "id": node,
                    "label": G.nodes[node].get("label", node),
                    "score": round(score, 4),
                    "reasons": reasons,
                }
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]
