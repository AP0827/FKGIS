# Incident Report Graph Generator

A simple script to generate knowledge graph visualizations for the Incident Report narrative.

## Quick Start

```bash
cd FKGIS
python3 generate_incident_graph.py
```

This will:
1. Process the Incident Report narrative through the NLP pipeline
2. Generate a knowledge graph with entities, events, and relationships
3. Create a PNG visualization at `nlp_pipeline/output/incident_report_graph.png`

## What It Does

- **Processes**: Only the `IncidentReport_RO.txt` file
- **Extracts**: Entities, events, locations, and their relationships
- **Filters**: Automatically removes isolated nodes (nodes with no connections)
- **Visualizes**: Clean knowledge graph with PageRank-based node sizing
- **Colors**: Entity (blue), Event (red), Location (green)

## Graph Statistics (Latest Run)
- **Original Nodes**: 270 total
- **Filtered Nodes**: 126 connected nodes (144 isolated nodes removed)
- **Edges**: 143 relationships
- **Node Types**: Entities, Events, Locations (with connections only)
- **Top Entities**: Reporting officers, victims, evidence items

## Advanced Usage

### Focus on Specific Entities
```bash
# View relationships for a specific entity
python3 nlp_pipeline/visualizer.py --entity-view "Cheryl Weston"

# View only significant nodes
python3 nlp_pipeline/visualizer.py --significant-only

# View top 20 PageRank nodes
python3 nlp_pipeline/visualizer.py --top-nodes 20
```

### Filter by Node Types
```bash
# Show only entities
python3 nlp_pipeline/visualizer.py --node-types Entity

# Show entities and events
python3 nlp_pipeline/visualizer.py --node-types Entity,Event
```

## Output Files

- `incident_report_graph.png` - Main visualization
- `IncidentReport_RO_enriched.json` - Processed document data
- `INCIDENT_REPORT_global_entities.json` - Extracted entities

## Requirements

- Python 3.7+
- NLP pipeline dependencies (spaCy, networkx, matplotlib)
- Incident Report text file in `nlp_pipeline/case_docs/`

The script handles all dependencies automatically and provides clear error messages if anything goes wrong.