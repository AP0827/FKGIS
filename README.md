# FKGIS: Forensic Knowledge Graph Intelligence System

> **Empowering Digital Investigations with AI-Driven Knowledge Graph Construction**

A comprehensive AI framework designed to automate the extraction, organization, and analysis of unstructured evidence from digital forensic investigations. FKGIS transforms fragmented case data into interconnected knowledge graphs, enabling investigators to uncover hidden relationships, verify timelines, and generate defensible case narratives.

## Problem Statement

Digital investigations are flooded with unstructured evidence—PDF reports, scanned notes, chat logs, witness interviews, and more. Investigators spend countless hours manually:
- Extracting key facts from disparate sources
- Aligning conflicting timelines
- Piecing together what happened
- Documenting their findings

This manual process is not just time-consuming; it is **inconsistent** and **highly prone to human bias**—both critical concerns in forensic investigation where evidence integrity is paramount.

## Solution Overview

FKGIS automates evidence processing through an intelligent pipeline that:

1. **Ingests** unstructured data (documents, images, audio)
2. **Extracts** entities, relationships, and events using NLP and LLMs
3. **Reconstructs** accurate timelines from fragmented information
4. **Visualizes** complex case connections as knowledge graphs
5. **Generates** consistent, legally defensible narratives

## Key Features

### Multimodal Evidence Processing
- Support for diverse input types: documents, images, CCTV frames, audio files
- Intelligent preprocessing using OCR, speech-to-text, and NLP
- Unified processing pipeline regardless of evidence format

### Knowledge Graph Construction
- Automatic entity recognition (people, locations, objects, events)
- Relationship extraction connecting entities across case documents
- PageRank-based significance scoring for critical nodes
- Interactive graph visualization for investigative insights

### Temporal Reasoning & Timeline Reconstruction
- Chronological event sequencing across multiple documents
- Automated conflict detection and flagging
- Temporal entity normalization (dates, times, relative references)
- Visual timeline representation for case review

### Advanced NLP Pipeline
- **Named Entity Recognition (NER)**: Identifies persons, organizations, locations, dates, times
- **Dependency Parsing**: Resolves grammatical relationships and statement nuances
- **Coreference Resolution**: Links references to same entities ("John", "he", "the suspect")
- **Relation Extraction**: Uncovers semantic relationships (associated with, testified against, etc.)
- **Event Extraction**: Structures discrete incidents with participants and context
- **LLM-Assisted Analysis**: Leverages large language models for enhanced accuracy

### Case Intelligence & Reporting
- Automated case summarization and briefs
- Structured JSON output for case records
- Hash-chained audit trails for evidence provenance
- Tamper-proof documentation of analysis steps

## Evaluation and Reproducibility

The repository currently contains descriptive pipeline outputs and graph statistics, but a full gold-labeled evaluation harness is still needed for publication-grade claims. To make the system scientifically defensible, the following evaluation setup should be reported and/or added to the codebase.

### Quantitative Metrics

Report entity and relation extraction quality on a manually annotated gold set using:

- Precision
- Recall
- F1 score
- Exact-match and relaxed-match variants where appropriate

For timeline reconstruction, report event ordering accuracy or another order-sensitive metric on the same held-out set.

### Baselines

Compare the full system against the following baselines to show the contribution of each modeling choice:

| System | Description |
|--------|-------------|
| spaCy-only | Rule-based or classical spaCy pipeline without LLM refinement |
| LLM-only | LLM-based extraction without the structured spaCy pipeline |
| FKGIS full system | Hybrid pipeline with preprocessing, NLP, LLM refinement, timeline reasoning, and graph construction |

Recommended reporting for each baseline:

| Model | Entity P/R/F1 | Relation P/R/F1 | Timeline score | Notes |
|-------|---------------|-----------------|----------------|-------|

### Ablation Study

To isolate the effect of each component, compare the full model with the following ablations:

| Variant | Removed Component | Expected Observation |
|---------|-------------------|----------------------|
| Full system | None | Best overall performance |
| No LLM refinement | Remove LLM post-processing and refinement | Lower relation quality and weaker entity normalization |
| No temporal reasoning | Disable event ordering and timeline reconstruction | More inconsistent chronology |
| No coreference resolution | Skip mention linking | More duplicate entities and fragmented graphs |
| No graph filtering | Keep all nodes and edges | Noisier graph, weaker interpretability |

### Error Analysis

Document the main failure modes explicitly so that the limits of the system are visible:

- Null or empty entities introduced by extraction or normalization
- Wrong relation direction or overly generic relation labels
- Ambiguous mentions that are linked to the wrong canonical entity
- Duplicate entities that survive mention unification
- Timeline ambiguities when the source text omits explicit dates or ordering cues
- Hallucinated or over-confident LLM refinements when the input is underspecified

### Graph Evaluation

PageRank is useful for ranking salient nodes, but it should not be the only graph metric. The current implementation already computes PageRank and can be extended with:

- Degree centrality for local connectivity
- Betweenness centrality for bridge entities
- Connected component coverage for graph cohesion
- Clustering coefficient for local structure
- Graph density for overall sparsity or saturation

These metrics help explain whether the graph is informative, overly fragmented, or dominated by a few hubs.

### LLM Configuration

The current LLM prototype code uses spaCy-LLM with the OpenAI family of models:

- NER: `spacy.NER.v1` with `spacy.GPT-3-5.v3` and `gpt-3.5-turbo`
- Relation extraction: `spacy.REL.v1` with the same OpenAI-backed adapter
- Coreference-style clustering: `spacy.Generic.v1` with a JSON-only prompt and `spacy.Json.v1` parser

The coreference prompt is intentionally constrained to return only JSON with a fixed schema, which helps reduce hallucination and makes the output easier to validate. For formal experiments, also pin and document any decoding parameters you use, such as temperature, top-p, and max tokens, because they are not currently fixed in this repository snapshot.

### Reproducibility Notes

To make runs reproducible, document the following alongside each experiment:

- Exact code revision or commit hash
- Input document set and annotation version
- Model name and provider adapter
- Prompt template or prompt file version
- Decoding parameters
- Seed, if the runtime exposes one
- Evaluation script and matching rules

The pipeline itself can be summarized as:

```text
documents -> preprocessing -> NER/dependency parsing -> coreference -> relation extraction -> event construction -> timeline reasoning -> knowledge graph -> verification -> reporting
```

## System Architecture

![FKGIS System Architecture Diagram](Architecture%20Diagram.png)

## Project Structure

```
FKGIS/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── setup.sh                           # Setup script
├── test_llm.py                        # LLM testing utility
├── test_supabase_connection.py        # Database testing
│
├── FKGIS/                             # Main package
│   ├── generate_incident_graph.py     # Graph generation script
│   ├── generate_results.py            # Results compilation
│   ├── INCIDENT_GRAPH_README.md       # Incident graph documentation
│   │
│   ├── backend/                       # Backend services
│   │   ├── database/                 # Database connectors
│   │   │   ├── mongo_connection.py   # MongoDB driver
│   │   │   ├── neo4j_connection.py   # Neo4j driver
│   │   │   ├── models/               # Data models
│   │   │   └── SQL/                  # SQL schemas
│   │   └── services/                 # Business logic services
│   │
│   ├── nlp_pipeline/                 # Core NLP processing
│   │   ├── pipeline.py               # Main pipeline orchestrator
│   │   ├── llm_prototypes.py         # LLM integration
│   │   ├── visualizer.py             # Graph visualization
│   │   ├── README.md                 # Pipeline documentation
│   │   │
│   │   ├── document_loader/          # Document ingestion
│   │   ├── preprocessing/            # Text normalization
│   │   ├── ner_and_dependency_parsing/    # Entity & syntax analysis
│   │   ├── coreference/              # Reference resolution
│   │   ├── relation_extraction/      # Relationship detection
│   │   ├── event_extraction/         # Event identification
│   │   ├── event_construction/       # Event structuring
│   │   ├── mention_unification/      # Entity linking
│   │   ├── timeline_extraction/      # Timeline building
│   │   ├── verification/             # Output validation
│   │   ├── kg/                       # Knowledge graph ops
│   │   ├── case_docs/                # Sample case documents
│   │   └── output/                   # Generated outputs
│   │
│   ├── knowledge_graph/              # KG specific modules
│   │   ├── relation_builder/         # KG construction
│   │   │   └── case_kg_builder.py
│   │   ├── schema/                   # KG schema definitions
│   │   └── timeline_builder/         # Timeline generation
│   │
│   ├── data_ingestion/               # Multi-modal data loading
│   │   ├── audio/                    # Audio processing
│   │   ├── image/                    # Image/OCR processing
│   │   ├── text/                     # Text document handling
│   │   └── preprocessing/            # Data normalization
│   │
│   ├── frontend/                     # Web interface (optional)
│   │   ├── components/               # React/Vue components
│   │   ├── pages/                    # Page templates
│   │   └── static/                   # Static assets
│   │
│   ├── configs/                      # Configuration files
│   │   ├── database/                 # DB configs
│   │   ├── env/                      # Environment settings
│   │   └── models/                   # Model configurations
│   │
│   ├── docs/                         # Documentation
│   │   ├── architecture/             # System design docs
│   │   ├── methodology/              # Methodology details
│   │   └── readme/                   # Supplementary guides
│   │
│   └── evaluation/                   # Testing & benchmarking
│       ├── benchmarks/               # Performance metrics
│       ├── datasets/                 # Test datasets
│       └── tests/                    # Unit tests
│
└── summarizer/                       # Text summarization module
    ├── src/
    │   └── webcrawlagent/
    ├── tests/
    │   ├── test_analyzer.py
    │   ├── test_gemini_client.py
    │   └── test_llm_factory.py
    └── pyproject.toml
```

## Quick Start

### Quick Start (Web App — no database required)

The simplest way to try FKGIS is the session-based web app. It needs no
database: each case is a folder under `FKGIS/webapp/sessions/` with its own
documents, work files and generated outputs.

```bash
./run_webapp.sh
# open http://127.0.0.1:8000
```

The first run creates a `.venv`, installs `requirements.txt`, downloads the
spaCy transformer model (`en_core_web_trf`) and starts the server. Then:

1. Click **Load Sample Case** to seed a ready-made case from the bundled
   pre-generated outputs (no GPU/pipeline run needed to view the graph).
2. Or **Create** a case, upload plain-text documents (choose Narrative /
   Biography / Interview type), then hit **Run Pipeline**.
3. Explore the interactive graph (vis-network): toggle Raw/Refined variants,
   filter node types, show only significant (PageRank) nodes, search/focus an
   entity, and export a paper-quality figure as PNG/SVG/PDF.

To enable Gemini refinement of the graph (the same model family the NLP
pipeline uses), create a `.env`:

```bash
GEMINI_API_KEY=your_key
FKGIS_USE_LLM=true
```

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- At least 4GB RAM for NLP models
- Optional: GPU support for faster processing

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd FKGIS
```

2. **Run the setup script:**
```bash
chmod +x setup.sh
./setup.sh
```

   Or manually:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Configure environment (optional):**
```bash
cp .env.example .env
# Edit .env with your database and API credentials
```

### Basic Usage

#### Process a Single Document
```bash
python -m FKGIS.nlp_pipeline.pipeline \
  --case CASE123 \
  --input path/to/document.txt \
  --output_dir ./output
```

#### Process Entire Case
```bash
python -m FKGIS.nlp_pipeline.pipeline \
  --case CASE123 \
  --step case \
  --output_dir ./output
```

#### Generate Incident Graph Visualization
```bash
cd FKGIS
python generate_incident_graph.py
```

## Current Run Statistics

These are descriptive outputs from the saved pipeline run, not a substitute for precision/recall/F1 evaluation.

| Metric | Result |
|--------|--------|
| **Entities Extracted** | 199 |
| **Knowledge Graph Nodes** | 270 |
| **Relationships (Edges)** | 149 |
| **Significant Nodes (PageRank)** | 44 |
| **Processing Time** | 72 seconds |
| **Timeline Events** | Auto-sequenced |

### Observed Comparison From Saved Outputs

These measurements come from the saved raw and refined graph artifacts in the repository and are useful as implementation-level comparison parameters until a gold-labeled benchmark is added.

| Metric | Raw Output | Refined Output |
|--------|------------|----------------|
| Total nodes | 270 | 287 |
| Total edges | 149 | 134 |
| Connected components | 157 | 198 |
| Largest component | 99 | 81 |
| Graph density | 0.004103 | 0.003265 |
| Average degree | 1.015 | 0.934 |
| Average clustering | 0.010979 | 0.000000 |
| Nullish / unknown entities | 40 | 93 |
| Unique relation types | 26 | 14 |

The current outputs show that refinement reduces relation variety and graph density, but it also increases the number of nullish entities in the saved artifact. That is a useful signal for the error analysis section: the refinement step should be audited for entity normalization regressions before these numbers are presented as final results.

## Core Dependencies

| Package | Purpose |
|---------|---------|
| **spaCy** | NLP models, NER, dependency parsing |
| **spaCy-LLM** | LLM-powered NLP enhancement |
| **spaCy-Transformers** | Advanced language models |
| **Coreferee** | Coreference resolution |
| **Neo4j/Neomodel** | Knowledge graph storage |
| **MongoDB** | Document storage |
| **SQLAlchemy** | SQL database abstraction |
| **Supabase** | Cloud database backend |
| **python-dateutil** | Temporal logic |

See [requirements.txt](requirements.txt) for the complete list.

## Documentation

- **[NLP Pipeline Guide](FKGIS/nlp_pipeline/README.md)** - Detailed pipeline architecture and stages
- **[Incident Graph Documentation](FKGIS/INCIDENT_GRAPH_README.md)** - Graph generation and analysis
- **[Methodology](FKGIS/docs/methodology/)** - Research methodology and approach
- **[Architecture Docs](FKGIS/docs/architecture/)** - System design and components

## Advanced Features

### Filtering Graph Outputs
```bash
# View only significant nodes
python FKGIS/nlp_pipeline/visualizer.py --significant-only

# View top N PageRank nodes
python FKGIS/nlp_pipeline/visualizer.py --top-nodes 20

# Filter by node type
python FKGIS/nlp_pipeline/visualizer.py --node-types Entity,Event
```

### Entity-Focused Analysis
```bash
# Analyze specific entity relationships
python FKGIS/nlp_pipeline/visualizer.py --entity-view "Suspect Name"
```

## Development

### Running Tests
```bash
# Run all tests
python -m pytest FKGIS/evaluation/tests/ -v

# Run specific test
python -m pytest FKGIS/evaluation/tests/test_ner.py -v

# With coverage
python -m pytest --cov=FKGIS FKGIS/evaluation/tests/
```

### Testing Connections
```bash
# Test database connections
python test_supabase_connection.py

# Test LLM integration
python test_llm.py
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new functionality
- Update documentation for API changes
- Ensure all tests pass before submitting PR

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Authors

- **Chethana Murthy** - Assistant Professor, RV College of Engineering
- **Aayush Pandey** - Computer Science Engineering, RV College of Engineering
- **Monal Rai** - Computer Science Engineering, RV College of Engineering
- **Navya Madiraju** - Computer Science Engineering, RV College of Engineering
- **A A Siddeshwaran** - Computer Science Engineering, RV College of Engineering

## Acknowledgments

- RV College of Engineering for research support
- spaCy and open-source NLP community for excellent tools
- Forensic investigation domain experts for guidance

## Support

For questions, issues, or suggestions:
- Open an issue on the repository
- Contact the development team
- Refer to the documentation in `/FKGIS/docs/`

---

**Made with care for digital forensics and investigative excellence**
