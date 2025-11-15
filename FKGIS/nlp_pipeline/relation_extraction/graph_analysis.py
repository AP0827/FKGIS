#!/usr/bin/env python3
"""
graph_analysis.py
Phase 2: Build graph from relations file, compute centrality and community metrics.

Usage:
  python graph_analysis.py relations.json graph_report.json graph.gexf
"""

import json, math, sys
import networkx as nx
from networkx.algorithms import community

def load_relations(rel_json):
    return json.load(open(rel_json, "r", encoding="utf8"))

def build_graph(candidates, prob_threshold=0.5):
    G = nx.DiGraph()
    # add nodes & edges with weight=prob
    for c in candidates:
        p = float(c.get("prob",0.0))
        if p < prob_threshold: continue
        u = c["head"]; v = c["tail"]
        # keep max weight if multiple edges between same u->v
        if G.has_edge(u,v):
            prev = G[u][v]["weight"]
            if p > prev:
                G[u][v]["weight"] = p
                G[u][v]["evidence"].append(c["sentence_text"])
        else:
            G.add_edge(u,v, weight=p, evidence=[c["sentence_text"]])
    # ensure isolated nodes are present (optional)
    for c in candidates:
        G.add_node(c["head"]); G.add_node(c["tail"])
    return G

def analyze_graph(G):
    report = {}
    report["num_nodes"] = G.number_of_nodes()
    report["num_edges"] = G.number_of_edges()
    report["density"] = nx.density(G)
    # components (undirected)
    comp = list(nx.connected_components(G.to_undirected())) if G.number_of_nodes()>0 else []
    report["num_components"] = len(comp)
    report["component_sizes"] = [len(c) for c in comp]

    # degree metrics
    deg = dict(G.degree())
    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())
    wdeg = {n: sum(d.get("weight",0.0) for _,_,d in G.edges(n,data=True)) + sum(d.get("weight",0.0) for _,_,d in G.in_edges(n,data=True)) for n in G.nodes()}
    report["top_degree"] = sorted(deg.items(), key=lambda x:-x[1])[:10]
    report["top_weighted_degree"] = sorted(wdeg.items(), key=lambda x:-x[1])[:10]

    # PageRank
    if G.number_of_nodes()>0:
        pr = nx.pagerank(G, weight="weight")
        report["top_pagerank"] = sorted(pr.items(), key=lambda x:-x[1])[:10]
    else:
        pr = {}

    # Betweenness & closeness (on undirected version for stability)
    if G.number_of_nodes()>0:
        und = G.to_undirected()
        bet = nx.betweenness_centrality(und, weight="weight")
        clo = nx.closeness_centrality(und)
        report["top_betweenness"] = sorted(bet.items(), key=lambda x:-x[1])[:10]
        report["top_closeness"] = sorted(clo.items(), key=lambda x:-x[1])[:10]
    else:
        bet = {}; clo = {}

    # communities (greedy modularity)
    if G.number_of_nodes()>0:
        communities = list(community.greedy_modularity_communities(G.to_undirected(), weight="weight"))
        report["num_communities"] = len(communities)
        report["community_sizes"] = [len(c) for c in communities]
        # map node->community id (first occurrence)
        node_community = {}
        for i,cset in enumerate(communities):
            for n in cset:
                node_community[n] = i
    else:
        node_community = {}

    # diameter (largest shortest path) per component (only meaningful for connected comps)
    diameters = []
    for cset in comp:
        if len(cset) <= 1: diameters.append(0)
        else:
            sub = G.subgraph(cset).to_undirected()
            try:
                diameters.append(nx.diameter(sub))
            except Exception:
                diameters.append(None)
    report["component_diameters"] = diameters

    # attach node metrics
    node_metrics = {}
    for n in G.nodes():
        node_metrics[n] = {
            "degree": deg.get(n,0),
            "in_degree": indeg.get(n,0),
            "out_degree": outdeg.get(n,0),
            "weighted_degree": wdeg.get(n,0.0),
            "pagerank": pr.get(n,0.0),
            "betweenness": bet.get(n,0.0),
            "closeness": clo.get(n,0.0),
            "community": node_community.get(n)
        }
    report["node_metrics"] = node_metrics
    return report

def save_gexf(G, path):
    nx.write_gexf(G, path)
    print("Wrote GEXF for visualization to", path)

def main(rel_json, out_json, out_gexf, threshold=0.5):
    cands = load_relations(rel_json)
    G = build_graph(cands, prob_threshold=threshold)
    report = analyze_graph(G)
    with open(out_json, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)
    save_gexf(G, out_gexf)
    print("Graph analysis complete. Nodes:", report["num_nodes"], "Edges:", report["num_edges"])

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python graph_analysis.py relations.json graph_report.json graph.gexf")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
