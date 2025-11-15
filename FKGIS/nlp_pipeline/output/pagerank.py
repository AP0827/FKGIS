import json
import networkx as nx
import statistics

# ====== LOAD JSON ======
print("Loading graph data...")
with open("graph_nodes_refined.json") as f:
    nodes = json.load(f)

with open("graph_edges_refined.json") as f:
    edges = json.load(f)

# ====== BUILD GRAPH ======
print("Building graph...")
G = nx.DiGraph()

# Add nodes
id_to_node = {n["id"]: n for n in nodes}
for n in nodes:
    G.add_node(n["id"], **n)

# Add edges
for e in edges:
    G.add_edge(e["source"], e["target"], edge_type=e["edge_type"])

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ====== RUN PAGE RANK (Updated for modern NetworkX) ======
print("Running PageRank...")

# OPTION 1: Standard PageRank (recommended)
try:
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
    method_used = "standard pagerank"
except Exception as e:
    print(f"Standard PageRank failed: {e}")
    
    # OPTION 2: Try with different parameters
    try:
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6)
        method_used = "standard pagerank (relaxed convergence)"
    except Exception as e2:
        print(f"Relaxed PageRank failed: {e2}")
        
        # OPTION 3: Use eigenvector centrality as fallback
        try:
            print("Using eigenvector centrality as fallback...")
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
            # Normalize to similar scale as PageRank
            max_centrality = max(centrality.values()) if centrality.values() else 1
            pagerank = {node: score/max_centrality for node, score in centrality.items()}
            method_used = "eigenvector centrality (normalized)"
        except Exception as e3:
            print(f"All centrality measures failed: {e3}")
            # Final fallback: degree centrality
            print("Using degree centrality as final fallback...")
            degree_centrality = nx.degree_centrality(G)
            pagerank = degree_centrality
            method_used = "degree centrality"

print(f"PageRank completed using: {method_used}")

# Sort by score
sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

# ====== IMPROVED THRESHOLD CALCULATION ======
if pagerank:
    scores = list(pagerank.values())
    median_score = statistics.median(scores)
    mean_score = statistics.mean(scores)
    
    # More robust threshold calculation
    if len(scores) > 10:
        # Use mean + 0.5 standard deviations for better threshold
        std_dev = statistics.stdev(scores)
        threshold = mean_score + (0.5 * std_dev)
    else:
        # For small graphs, use a percentage of max score
        max_score = max(scores)
        threshold = max_score * 0.3
    
    print(f"Score statistics: min={min(scores):.6f}, max={max(scores):.6f}, mean={mean_score:.6f}, median={median_score:.6f}")
else:
    threshold = 0
    print("No scores calculated")

print(f"Threshold used: {threshold:.6f}")

# ====== DISPLAY SIGNIFICANT NODES ======
print(f"\n=== Significant Nodes (>= {threshold:.6f}) ===")
significant_nodes = []
for node_id, score in sorted_nodes:
    if score >= threshold:
        node = id_to_node.get(node_id, {"text": "Unknown", "label": "Unknown"})
        print(f"{node['text']} ({node['label']}) → {score:.6f}")
        significant_nodes.append({
            "id": node_id,
            "text": node["text"],
            "label": node.get("label", "Unknown"),
            "score": score
        })

print(f"\nFound {len(significant_nodes)} significant nodes out of {len(nodes)} total")

# ====== SAVE COMPREHENSIVE RESULTS ======
output_all = []
for node_id, score in sorted_nodes:
    node = id_to_node.get(node_id, {"text": "Unknown", "label": "Unknown"})
    output_all.append({
        "id": node_id,
        "text": node["text"],
        "label": node.get("label", "Unknown"),
        "score": round(score, 6),
        "is_significant": score >= threshold
    })

# Save JSON with metadata
results_data = {
    "metadata": {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "significant_nodes_count": len(significant_nodes),
        "pagerank_method": method_used,
        "threshold_used": round(threshold, 6),
        "score_statistics": {
            "min": round(min(pagerank.values()), 6) if pagerank else 0,
            "max": round(max(pagerank.values()), 6) if pagerank else 0,
            "mean": round(statistics.mean(pagerank.values()), 6) if pagerank else 0,
            "median": round(statistics.median(pagerank.values()), 6) if pagerank else 0
        }
    },
    "nodes": output_all
}

with open("pagerank_results.json", "w") as f:
    json.dump(results_data, f, indent=2)

# Save text file (human-readable)
with open("pagerank_results.txt", "w") as f:
    f.write("=== PAGE RANK RESULTS ===\n\n")
    f.write(f"Method: {method_used}\n")
    f.write(f"Threshold: {threshold:.6f}\n")
    f.write(f"Significant nodes: {len(significant_nodes)}/{len(nodes)}\n\n")
    
    f.write("ALL NODES (sorted by importance):\n")
    f.write("-" * 80 + "\n")
    for entry in output_all:
        significance_indicator = " ***" if entry["is_significant"] else ""
        f.write(f"{entry['text']} [{entry['label']}] → {entry['score']:.6f}{significance_indicator}\n")

print(f"\nSaved: pagerank_results.json & pagerank_results.txt")
print(f"Method used: {method_used}")
print(f"Significant nodes: {len(significant_nodes)}/{len(nodes)}")