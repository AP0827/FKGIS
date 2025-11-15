# NLP Pipeline

This project presents a comprehensive Natural Language Processing (NLP) pipeline designed for the systematic extraction and structuring of information from diverse text-based documents. It aims to transform unstructured data into actionable insights through a series of sophisticated analytical stages, particularly relevant for crime analysis and investigation.

## Methodology

The NLP pipeline employs a structured, multi-stage approach to process and analyze textual data, transforming raw documents into structured knowledge representations. Each stage is designed to build upon the previous, ensuring comprehensive information extraction critical for understanding complex crime narratives:

1.  **Document Discovery & Loading**:
    *   **Technical Detail**: Utilizes `document_loader.load_docs` to programmatically identify and ingest various document types (e.g., incident reports, witness statements, suspect interviews, biographical data) relevant to a case.
    *   **Crime Domain Relevance**: Ensures all pertinent textual evidence is captured, forming the foundation for subsequent analysis. This is crucial for consolidating information from disparate sources in an investigation.

2.  **Type-Specific Preprocessing**:
    *   **Technical Detail**: Employs tailored preprocessing modules (e.g., `prep_ROnarrative`, `prep_biography`, `prep_interview`) to segment documents into sentences and identify potential event boundaries. This step normalizes text for consistent downstream processing.
    *   **Crime Domain Relevance**: Breaks down lengthy reports or interviews into manageable units, facilitating accurate entity and event recognition. It prepares the text for precise linguistic analysis, crucial for distinguishing factual accounts from subjective statements.

3.  **Named Entity Recognition (NER)**:
    *   **Technical Detail**: Leverages advanced NLP models (via `ner_and_dependency_parsing.ner_module`) to identify and classify key entities such as persons, organizations, locations, dates, and times. It also attaches token-level dependency parsing information. This stage can leverage LLM-assisted NER for enhanced accuracy and broader entity recognition.
    *   **Crime Domain Relevance**: Crucial for identifying all involved parties (victims, suspects, witnesses), locations of interest, and temporal markers. Dependency parsing helps understand grammatical relationships, which can reveal nuances in statements or descriptions.

4.  **Coreference Resolution & Mention Unification**:
    *   **Technical Detail**: Employs `coreference.coref_global_pool` and `mention_unification.unify_mentions` to link mentions of the same entity (e.g., "John Smith," "he," "the suspect") and resolve pronouns/abbreviations. LLM-assisted coreference can further improve the accuracy of entity linking.
    *   **Crime Domain Relevance**: Essential for accurately tracking individuals and entities throughout a case narrative, preventing confusion and ensuring a coherent understanding of who did what, where, and when. This is vital for building a clear picture of events and relationships.

5.  **Relation Extraction**:
    *   **Technical Detail**: Identifies semantic relationships between entities, extracting structured Subject-Predicate-Object (SPO) triples using modules like `relation_extraction.basic_spacy_rel_extractor`. LLM-assisted relation extraction can uncover more complex or nuanced relationships.
    *   **Crime Domain Relevance**: Uncovers connections between entities, such as "Suspect X *associated with* Location Y" or "Victim Z *testified against* Suspect A." This helps map out networks of individuals and their interactions.

6.  **Event Extraction**:
    *   **Technical Detail**: Detects and structures discrete events from the text, identifying participants, locations, and temporal context using `event_extraction.build_events`.
    *   **Crime Domain Relevance**: Reconstructs the sequence of actions and occurrences relevant to the crime. This allows for the identification of key incidents, motives, and the timeline of criminal activity.

7.  **Timeline Construction**:
    *   **Technical Detail**: Normalizes temporal information extracted from the text and uses it to build a chronological sequence of events, managed by `timeline.timeline_builder`.
    *   **Crime Domain Relevance**: Establishes a clear timeline of events, which is fundamental in crime investigation for corroborating alibis, identifying inconsistencies, and understanding the sequence of actions leading up to, during, and after an incident.

8.  **Knowledge Graph Construction**:
    *   **Technical Detail**: Synthesizes all extracted entities, relations, and events into a structured knowledge graph (nodes and edges) using `kg.graph_builder`.
    *   **Crime Domain Relevance**: Provides a powerful, interconnected representation of case data. Investigators can query the graph to uncover complex relationships, identify patterns, and explore connections that might not be apparent in linear text.

9.  **Output Refinement & Verification**:
    *   **Technical Detail**: Generates enriched JSON outputs and the knowledge graph, with final verification checks performed by `verification.verify_outputs` to ensure data integrity and consistency. LLM-assisted components contribute to the refinement of extracted information throughout the pipeline.
    *   **Crime Domain Relevance**: Ensures the reliability and accuracy of the extracted information, which is critical for evidence-based decision-making in investigations and legal proceedings.

## Features

*   **Multi-Document Processing**: Capable of handling and integrating information from various document types within a single case, providing a holistic view.
*   **Advanced NLP Techniques**: Employs state-of-the-art NER, dependency parsing, coreference resolution, relation extraction, and event extraction for deep textual understanding.
*   **Structured Knowledge Representation**: Generates knowledge graphs and timelines, enabling efficient querying, pattern discovery, and hypothesis generation.
*   **Modular Design**: Each stage of the pipeline is modular, allowing for flexibility, targeted execution, and easier updates or improvements.

## Usage

The pipeline can be executed via its command-line interface (CLI) for ad-hoc runs or integrated into other services.

1.  **Full Narrative Pipeline**: Processes a single narrative text file through all stages to produce enriched event data.
    ```bash
    python -m FKGIS.nlp_pipeline.pipeline --case <case_id> --input <input_txt_path> [--output_dir <output_directory>]
    ```

2.  **Full Case Pipeline**: Orchestrates the end-to-end processing for a multi-document case, generating a comprehensive output summary.
    ```bash
    python -m FKGIS.nlp_pipeline.pipeline --case <case_id> --step case [--output_dir <output_directory>]
    ```

3.  **Individual Step Execution**: Allows for running specific stages of the pipeline for targeted analysis or debugging.
    *   **Preprocessing**:
        ```bash
        python -m FKGIS.nlp_pipeline.pipeline --input <input_txt_path> --step preprocess
        ```
    *   **NER**:
        ```bash
        python -m FKGIS.nlp_pipeline.pipeline --input <preprocessed_json_path> --step ner
        ```
    *   **Coreference**:
        ```bash
        python -m FKGIS.nlp_pipeline.pipeline --case <case_id> --input <ner_json_path> --step coref [--output_dir <output_directory>]
        ```

**Note**: Replace placeholders like `<case_id>`, `<input_txt_path>`, etc., with your specific values.

## Dependencies

The project relies on several Python libraries for NLP tasks. Please install them using pip:

```bash
pip install -r requirements.txt
```

Key dependencies include:
*   spaCy (for NER, dependency parsing, and coreference)
*   spaCy-LLM (for LLM-based models)
*   OpenAI (if using OpenAI models)
*   NLTK (potentially for certain text processing tasks)
*   Pandas (for data manipulation)
*   NumPy (for numerical operations)

(A more exhaustive list can be found in `requirements.txt`.)

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` file (if it exists) for guidelines.

## License

This project is licensed under the [License Name] License - see the `LICENSE` file for details.