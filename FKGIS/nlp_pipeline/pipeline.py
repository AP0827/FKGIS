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
from typing import Dict, Optional

from .preprocessing.prep_ROnarrative import process_file as preprocess_narrative
from .ner_and_dependency_parsing.ner_module import (
    setup_nlp_pipeline,
    process_json_file as run_ner_on_json,
)
from .coreference.coref_global_pool import (
    process_ner_json as build_global_entities,
    create_coref_pipeline,
    run_coref_on_document,
)
from .mention_unification.mention_unifier import unify_mentions
from .relation_extraction.basic_rel_extractor import extract_relations
from .event_construction.event_builder import build_events


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

    Steps:
        1) Preprocessing (sentence & event segmentation) -> processed_doc JSON
        2) NER -> entities attached per sentence + flattened processed_doc["entities"]
        3) Coreference / global entity pool (file-level, legacy)
        4) Document-level coref to add processed_doc["coref_entities"]
        5) Pronoun substitution + mention unification -> processed_doc["unified_sentences"]
        6) Basic relation extraction -> processed_doc["relations"]
        7) Event construction -> processed_doc["events"]

    Args:
        case_id: Identifier for the case.
        input_txt_path: Path to the raw narrative .txt file.
        output_dir: Optional base directory for all outputs. If provided,
            all outputs are written there; otherwise they are created
            alongside the input.

    Returns:
        Dict with keys:
            "preprocessed"    -> *_ROnarrative_processed.json
            "ner"             -> *_ROnarrative_processed_ner.json
            "global_entities" -> {case_id}_global_entities.json
            "enriched"        -> *_ROnarrative_processed_ner_events.json
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

    # 4) Pronoun substitution + mention unification
    processed_doc = unify_mentions(processed_doc)

    # 5) Basic relation extraction (rule-based)
    processed_doc = extract_relations(processed_doc)

    # 6) Event construction from relations (+ optional time)
    processed_doc = build_events(processed_doc)

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

    parser = argparse.ArgumentParser(description="Run the narrative NLP pipeline.")
    parser.add_argument("--case", dest="case_id", required=False, help="Case ID (required for coref / full pipeline).")
    parser.add_argument(
        "--input",
        dest="input",
        required=True,
        help="Path to the raw narrative .txt file or intermediate JSON (depending on step).",
    )
    parser.add_argument(
        "--step",
        choices=["preprocess", "ner", "coref", "full"],
        default="full",
        help="Which part of the pipeline to run.",
    )
    parser.add_argument("--output-dir", dest="output_dir", help="Optional base directory for outputs.")

    args = parser.parse_args()

    if args.step in {"coref", "full"} and not args.case_id:
        parser.error("--case is required when step is 'coref' or 'full'.")

    if args.step == "preprocess":
        out = run_preprocessing(args.input)
        print(out)
    elif args.step == "ner":
        out = run_ner(args.input)
        print(out)
    elif args.step == "coref":
        out = run_coreference(args.case_id, args.input, output_dir=args.output_dir)
        print(out)
    else:  # full
        results = run_full_pipeline(args.case_id, args.input, output_dir=args.output_dir)
        for k, v in results.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()