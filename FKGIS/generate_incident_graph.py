#!/usr/bin/env python3
"""
Simple script to generate graph visualization for the Incident Report narrative.

This script processes only the IncidentReport_RO.txt file and creates a knowledge graph
visualization showing entities, events, and their relationships.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

def run_incident_report_pipeline():
    """Run the pipeline for just the Incident Report and generate visualization."""

    print("🚀 Processing Incident Report narrative...")

    # Define paths
    base_dir = Path(__file__).parent
    nlp_dir = base_dir / "nlp_pipeline"
    output_dir = nlp_dir / "output"
    incident_file = nlp_dir / "case_docs" / "IncidentReport_RO.txt"

    if not incident_file.exists():
        print(f"❌ Incident report file not found: {incident_file}")
        return False

    # Step 1: Run preprocessing
    print("📝 Step 1: Preprocessing...")
    preprocess_cmd = [
        sys.executable, "-m", "nlp_pipeline.pipeline",
        "--input", str(incident_file),
        "--case", "INCIDENT_REPORT",
        "--step", "full",
        "--output-dir", str(output_dir)
    ]

    result = subprocess.run(preprocess_cmd, cwd=str(base_dir), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Preprocessing failed: {result.stderr}")
        return False

    print("✅ Preprocessing completed")

    # Step 2: Generate visualization
    print("🎨 Step 2: Generating graph visualization...")

    # Use the refined files if they exist, otherwise use the original
    nodes_file = output_dir / "graph_nodes_refined.json"
    edges_file = output_dir / "graph_edges_refined.json"

    if not nodes_file.exists() or not edges_file.exists():
        # Try original files
        nodes_file = output_dir / "graph_nodes.json"
        edges_file = output_dir / "graph_edges.json"

    if not nodes_file.exists() or not edges_file.exists():
        print("❌ Graph files not found. Please run the full case pipeline first.")
        return False

    # Generate visualization
    viz_cmd = [
        sys.executable, str(nlp_dir / "visualizer.py"),
        "--nodes", str(nodes_file),
        "--edges", str(edges_file),
        "--output", str(output_dir / "incident_report_graph.png")
    ]

    result = subprocess.run(viz_cmd, cwd=str(base_dir), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Visualization failed: {result.stderr}")
        return False

    print("✅ Graph visualization generated successfully!")
    print(f"📁 Output saved to: {output_dir / 'incident_report_graph.png'}")

    # Show some basic stats
    try:
        with open(nodes_file, 'r') as f:
            nodes = json.load(f)
        with open(edges_file, 'r') as f:
            edges = json.load(f)

        print("\n📊 Graph Statistics:")
        print(f"   • Nodes: {len(nodes)}")
        print(f"   • Edges: {len(edges)}")

        # Count node types
        node_types = {}
        for node in nodes:
            node_type = node.get('type', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print("   • Node types:")
        for node_type, count in node_types.items():
            print(f"     - {node_type}: {count}")

    except Exception as e:
        print(f"⚠️  Could not read graph statistics: {e}")

    return True

def main():
    """Main function."""
    print("🔍 Incident Report Graph Generator")
    print("=" * 40)

    success = run_incident_report_pipeline()

    if success:
        print("\n🎉 Success! Open the PNG file to view the knowledge graph.")
        print("💡 Tip: Use --entity-view 'EntityName' to focus on specific entities")
    else:
        print("\n❌ Failed to generate graph. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()