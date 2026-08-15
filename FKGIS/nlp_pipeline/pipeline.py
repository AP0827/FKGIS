"""High-level NLP pipeline orchestration utilities.

This module lets you run the narrative pipeline step-by-step or end-to-end:

1. Preprocessing (sentence & event segmentation)
2. NER (and dependency parsing via spaCy's parser, if needed)
3. Coreference / global entity pool

It is intended to be imported from other services, but also exposes a small
CLI for ad-hoc runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List, Any
import time

from .preprocessing.prep_ROnarrative import process_file as preprocess_narrative
from .preprocessing.prep_biography import process_file as preprocess_biography
from .preprocessing.prep_interview import process_file as preprocess_interview
from .ner_and_dependency_parsing.ner_module import (
    setup_nlp_pipeline,
    process_json_file as run_ner_on_json,
)
from .dependency_parsing.dep_module import process_json_file as run_deps_on_json
from .coreference.coref_global_pool import (
    process_ner_json as build_global_entities,
    create_coref_pipeline,
    run_coref_on_document,
)
from .document_loader.load_docs import discover_case_docs
from .mention_unification.unify_mentions import unify_mentions
from .relation_extraction.basic_spacy_rel_extractor import refine_case_relations
from .event_extraction.build_events import build_events
from .timeline.timeline_builder import build_timeline_for_case
from .kg.graph_builder import build_nodes_and_edges_for_case, save_graph_json
from .verification.verify_outputs import verify_case_outputs


def run_preprocessing(input_txt_path: str) -> str:
    """Run preprocessing on a raw narrative .txt file.

    Args:
        input_txt_path: Path to the raw narrative .txt file.

    Returns:
        Path to the generated *_ROnarrative_processed.json file.
    """
    return preprocess_narrative(input_txt_path)


def run_ner(preprocessed_json_path: str, nlp=None) -> str:
    """Run NER on a preprocessed narrative JSON file.

    Args:
        preprocessed_json_path: Path to the *_ROnarrative_processed.json file.
        nlp: Optional pre-loaded spaCy pipeline. If None, a new
             transformer NER pipeline is created.

    Returns:
        Path to the generated *_ROnarrative_processed_ner.json file.
    """
    if nlp is None:
        nlp = setup_nlp_pipeline()
    return run_ner_on_json(preprocessed_json_path, nlp)


def run_coreference(case_id: str, ner_json_path: str, output_dir: Optional[str] = None) -> str:
    """Run coreference / global entity pooling over a NER-enriched JSON file.

    Args:
        case_id: Identifier for the case; used to name the output file.
        ner_json_path: Path to the *_ROnarrative_processed_ner.json file.
        output_dir: Optional directory for the output JSON.

    Returns:
        Path to the generated {case_id}_global_entities.json file.
    """
    entities = build_global_entities(case_id, ner_json_path)

    if output_dir is None:
        out_path = Path(f"{case_id}_global_entities.json")
    else:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        out_path = output_dir_path / f"{case_id}_global_entities.json"

    import json

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(entities, f, indent=4, ensure_ascii=False)

    return str(out_path)


def run_full_pipeline(case_id: str, input_txt_path: str, output_dir: Optional[str] = None) -> Dict[str, str]:
    """Run the full narrative pipeline from raw text to enriched events.

    This legacy helper is kept primarily for backward compatibility and
    focuses on the Reporting Officer's narrative only. For the full
    multi-document case pipeline (narrative + biography + interviews),
    use run_case_pipeline().
    """
    input_txt = Path(input_txt_path)

    if output_dir is None:
        preprocessed_path = preprocess_narrative(str(input_txt))
        ner_path = run_ner(preprocessed_path)
        global_entities_path = run_coreference(case_id, ner_path)
    else:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Preprocessing writes next to the input; move or re-point into output_dir
        preprocessed_path = preprocess_narrative(str(input_txt))
        preprocessed_target = output_dir_path / Path(preprocessed_path).name
        Path(preprocessed_path).replace(preprocessed_target)
        preprocessed_path = str(preprocessed_target)

        ner_path = run_ner(preprocessed_path)
        ner_target = output_dir_path / Path(ner_path).name
        Path(ner_path).replace(ner_target)
        ner_path = str(ner_target)

        global_entities_path = run_coreference(case_id, ner_path, output_dir=str(output_dir_path))

    # ------------------------------------------------------------------
    # Post-coreference document-level enrichment:
    #   coref_entities, unified_sentences, relations, events
    # ------------------------------------------------------------------
    import json

    with open(ner_path, "r", encoding="utf-8") as f:
        processed_doc = json.load(f)

    # Ensure we have a sentence_id on each original sentence for downstream mapping
    sentences = processed_doc.get("sentences") or []
    for idx, sent in enumerate(sentences):
        sent.setdefault("sentence_id", idx)

    # Document-level coreference: populate processed_doc["coref_entities"]
    coref_nlp = create_coref_pipeline()
    doc_type = processed_doc.get("doc_type", "narrative")
    processed_doc = run_coref_on_document(coref_nlp, processed_doc, doc_type=doc_type)

    # Optional convenience alias so downstream code can refer to "coref"
    if "coref_entities" in processed_doc:
        processed_doc.setdefault("coref", processed_doc["coref_entities"])

    # Pronoun substitution + mention unification
    processed_doc = unify_mentions(processed_doc)

    # Basic relation extraction (rule-based)
    # Need to pass the nlp object for matcher-based extraction
    processed_doc = refine_case_relations(processed_doc, nlp=coref_nlp) # Using coref_nlp as it's already loaded

    # Event construction from relations (+ optional time)
    processed_doc = build_events(processed_doc, doc_name=processed_doc.get("doc_name", "unknown_doc"))

    # Persist enriched document
    enriched_path = ner_path.replace(".json", "_events.json")
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(processed_doc, f, indent=4, ensure_ascii=False)

    return {
        "preprocessed": preprocessed_path,
        "ner": ner_path,
        "global_entities": global_entities_path,
        "enriched": enriched_path,
    }


def run_case_pipeline(
    case_id: str,
    docs_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the full multi-document case pipeline for the Kimberly Pace case.

    Order of operations:
      1) load_docs          - discover narrative / biography / interview docs
      2) preprocessing      - type-specific preprocessors to processed_doc JSON
      3) ner                - NER over each processed_doc
      4) dependency parse   - attach token-level deps (for downstream use)
      5) coreference        - document-level coref_entities
      6) mention unification- pronoun + abbreviation expansion
      7) relation extraction- refined, case-specific SPO triples
      8) event extraction   - enriched events with participants/locations
      9) timeline builder   - normalize times and build BEFORE edges
     10) KG builder         - graph_nodes + graph_edges
     11) output verification- sanity checks over entities/relations/events/timeline/KG

    The final summary is written to nlp_pipeline/output/case_output.json.

    Args:
        case_id: Identifier for the case.
        docs_dir: Directory containing raw case documents (txt). Defaults to
            the bundled ``case_docs`` folder.
        output_dir: Directory where graph artifacts and the case summary are
            written. Defaults to ``nlp_pipeline/output``.
        progress_callback: Optional callable ``(stage: str, message: str) -> None``
            invoked at the start of each major pipeline stage. Useful for
            reporting progress to a UI without tight coupling.

    Returns:
        Dict containing the output path and timing information.
    """
    import json as _json

    def _report(stage: str, message: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(stage, message)
            except Exception:
                pass

    pipeline_start_time = time.time()
    timing_info = {}

    base_dir = Path(__file__).resolve().parent
    docs_base = Path(docs_dir) if docs_dir else (base_dir / "case_docs")
    output_root = Path(output_dir) if output_dir else (base_dir / "output")
    output_root.mkdir(parents=True, exist_ok=True)

    # 1) Discover case documents
    _report("discovery", "Discovering case documents...")
    discover_start = time.time()
    discovered = discover_case_docs(str(docs_base))
    if not discovered:
        raise RuntimeError(f"No case documents found under {docs_base}")
    timing_info["document_discovery"] = time.time() - discover_start

    # 2–8) Per-document pipeline
    _report("setup", "Loading NLP models...")
    setup_start = time.time()
    ner_nlp = setup_nlp_pipeline()
    coref_nlp = create_coref_pipeline()
    timing_info["pipeline_setup"] = time.time() - setup_start

    case_docs_entries: List[Dict[str, Any]] = []
    doc_processing_start = time.time()

    for d in discovered:
        raw_path = d.path
        doc_type = d.doc_type
        doc_name = d.doc_id

        # Preprocessing
        _report("processing", f"Preprocessing {doc_name} ({doc_type})...")
        if doc_type == "narrative":
            preprocessed_path = preprocess_narrative(raw_path)
        elif doc_type == "biography":
            preprocessed_path = preprocess_biography(raw_path)
        elif doc_type == "interview":
            preprocessed_path = preprocess_interview(raw_path)
        else:
            # Skip unknown types silently
            continue

        # NER
        _report("processing", f"Running NER on {doc_name}...")
        ner_path = run_ner(preprocessed_path, nlp=ner_nlp)

        # Dependency parsing
        _report("processing", f"Parsing dependencies for {doc_name}...")
        dep_path = run_deps_on_json(ner_path, nlp=None)

        # Load processed_doc for higher-level steps
        with open(dep_path, "r", encoding="utf-8") as f:
            processed_doc = _json.load(f)

        # Ensure sentence_id exists
        sentences = processed_doc.get("sentences") or []
        for idx, sent in enumerate(sentences):
            sent.setdefault("sentence_id", idx)

        # Coreference
        _report("processing", f"Resolving coreference for {doc_name}...")
        processed_doc = run_coref_on_document(coref_nlp, processed_doc, doc_type=doc_type)
        if "coref_entities" in processed_doc:
            processed_doc.setdefault("coref", processed_doc["coref_entities"])

        # Mention unification (pronouns + abbreviations)
        _report("processing", f"Unifying mentions for {doc_name}...")
        processed_doc = unify_mentions(processed_doc)

        # Refined, case-specific relations
        _report("processing", f"Extracting relations for {doc_name}...")
        processed_doc = refine_case_relations(processed_doc, nlp=ner_nlp)

        # Event extraction (participants, locations, event_ids)
        _report("processing", f"Building events for {doc_name}...")
        processed_doc = build_events(processed_doc, doc_name=doc_name)

        # Persist per-doc enriched JSON for inspection
        enriched_doc_path = output_root / f"{doc_name}_enriched.json"
        with enriched_doc_path.open("w", encoding="utf-8") as f:
            _json.dump(processed_doc, f, indent=4, ensure_ascii=False)

        case_docs_entries.append(
            {
                "doc_name": doc_name,
                "doc_type": doc_type,
                "processed_doc": processed_doc,
            }
        )

    timing_info["document_processing"] = time.time() - doc_processing_start

    # 9) Timeline builder
    _report("timeline", "Building timeline...")
    timeline_start = time.time()
    timeline_info = build_timeline_for_case(case_id, case_docs_entries)
    timing_info["timeline_building"] = time.time() - timeline_start

    # 10) KG builder (nodes + edges JSON)
    _report("graph", "Building knowledge graph...")
    kg_start = time.time()
    graph = build_nodes_and_edges_for_case(case_id, case_docs_entries, timeline_info)
    nodes_path, edges_path = save_graph_json(graph, base_dir=str(output_root))
    kg_total_time = time.time() - kg_start

    # Extract detailed KG timing if available
    kg_timing = graph.get("timing", {})
    timing_info["kg_creation"] = kg_total_time
    timing_info["kg_pagerank_calculation"] = kg_timing.get("pagerank_calculation", 0)
    timing_info["kg_total_internal"] = kg_timing.get("total_kg_creation", 0)

    # 11) Verification
    _report("verification", "Verifying outputs...")
    verification_start = time.time()
    verification_report = verify_case_outputs(
        case_id,
        case_docs_entries,
        timeline_info,
        graph_nodes=graph.get("nodes") or [],
        graph_edges=graph.get("edges") or [],
        min_relations=5,
    )
    timing_info["verification"] = time.time() - verification_start

    timing_info["total_pipeline_time"] = time.time() - pipeline_start_time

    # Final case_output.json
    case_output = {
        "case_id": case_id,
        "documents": [
            {"doc_name": d["doc_name"], "doc_type": d["doc_type"]}
            for d in case_docs_entries
        ],
        "timeline": timeline_info,
        "graph_nodes_path": nodes_path,
        "graph_edges_path": edges_path,
        "verification": verification_report,
    }

    case_output_path = output_root / "case_output.json"
    with case_output_path.open("w", encoding="utf-8") as f:
        _json.dump(case_output, f, indent=4, ensure_ascii=False)

    return {
        "output_path": str(case_output_path),
        "timing": timing_info,
        "case_id": case_id
    }


def main() -> None:
    """Small CLI wrapper for ad-hoc runs.

    Examples
    --------
    Run full pipeline:
        python -m FKGIS.nlp_pipeline.pipeline --case CASE123 --input sample.txt

    Or run an individual step:
        python -m FKGIS.nlp_pipeline.pipeline --input sample.txt --step preprocess
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the narrative or case-level NLP pipeline.")
    parser.add_argument("--case", dest="case_id", required=False, help="Case ID (required for coref / full / case pipeline).")
    parser.add_argument(
        "--input",
        dest="input",
        required=False,
        help="Path to the raw narrative .txt file or intermediate JSON (depending on step).",
    )
    parser.add_argument(
        "--step",
        choices=["preprocess", "ner", "coref", "full", "case"],
        default="full",
        help="Which part of the pipeline to run.",
    )
    parser.add_argument("--output-dir", dest="output_dir", help="Optional base directory for outputs.")

    args = parser.parse_args()

    if args.step in {"coref", "full", "case"} and not args.case_id:
        parser.error("--case is required when step is 'coref', 'full', or 'case'.")

    if args.step in {"preprocess", "ner", "coref", "full"} and not args.input:
        parser.error("--input is required for steps 'preprocess', 'ner', 'coref', and 'full'.")

    if args.step == "preprocess":
        out = run_preprocessing(args.input)
        print(out)
    elif args.step == "ner":
        out = run_ner(args.input)
        print(out)
    elif args.step == "coref":
        out = run_coreference(args.case_id, args.input, output_dir=args.output_dir)
        print(out)
    elif args.step == "case":
        result = run_case_pipeline(args.case_id, output_dir=args.output_dir)
        print(f"Output: {result['output_path']}")
        print(f"Total time: {result['timing']['total_pipeline_time']:.2f}s")
    else:  # full
        results = run_full_pipeline(args.case_id, args.input, output_dir=args.output_dir)
        for k, v in results.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()