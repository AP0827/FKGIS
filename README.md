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

## Performance & Results

Based on testing with a forensic investigation dataset:

| Metric | Result |
|--------|--------|
| **Entities Extracted** | 199 |
| **Knowledge Graph Nodes** | 270 |
| **Relationships (Edges)** | 143 |
| **Significant Nodes (PageRank)** | 44 |
| **Processing Time** | 72 seconds |
| **Timeline Events** | Auto-sequenced |

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
