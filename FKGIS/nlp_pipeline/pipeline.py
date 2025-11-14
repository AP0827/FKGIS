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
from .coreference.coref_global_pool import process_ner_json as build_global_entities


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
    """Run the full narrative pipeline from raw text to global entities.

    Args:
        case_id: Identifier for the case.
        input_txt_path: Path to the raw narrative .txt file.
        output_dir: Optional base directory for all outputs. If provided,
            all outputs are written there; otherwise they are created
            alongside the input.

    Returns:
        Dict with keys: "preprocessed", "ner", "global_entities"
        mapping to their respective file paths.
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

    return {
        "preprocessed": preprocessed_path,
        "ner": ner_path,
        "global_entities": global_entities_path,
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