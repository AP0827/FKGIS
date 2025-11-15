from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from FKGIS.nlp_pipeline.ner_and_dependency_parsing.ner_module import (
    setup_nlp_pipeline,
    run_ner_on_document,
)
from FKGIS.nlp_pipeline.coreference.coref_global_pool import (
    create_coref_pipeline,
    run_coref_on_document,
    GlobalEntityPool,
    integrate_processed_doc,
)
from FKGIS.nlp_pipeline.mention_unification.mention_unifier import unify_mentions
from FKGIS.nlp_pipeline.relation_extraction.basic_rel_extractor import extract_relations
from FKGIS.nlp_pipeline.event_construction.event_builder import build_events


def _ensure_sentence_ids(processed_doc: Dict[str, Any]) -> None:
    """Ensure every sentence has a stable sentence_id."""
    sentences = processed_doc.get("sentences") or []
    for idx, sent in enumerate(sentences):
        sent.setdefault("sentence_id", idx)


def _enrich_document(
    processed_doc: Dict[str, Any],
    doc_type: str,
    ner_nlp,
    coref_nlp,
) -> Dict[str, Any]:
    """Run NER, coref, unification, relations, and events on a processed_doc if needed."""
    # 1) NER (if not already present)
    if not processed_doc.get("entities"):
        processed_doc = run_ner_on_document(processed_doc, ner_nlp)

    # 2) sentence_id for mapping
    _ensure_sentence_ids(processed_doc)

    # 3) Document-level coref
    processed_doc = run_coref_on_document(coref_nlp, processed_doc, doc_type=doc_type)
    if "coref_entities" in processed_doc:
        processed_doc.setdefault("coref", processed_doc["coref_entities"])

    # 4) Mention unification
    processed_doc = unify_mentions(processed_doc)

    # 5) Relation extraction
    processed_doc = extract_relations(processed_doc)

    # 6) Event construction
    processed_doc = build_events(processed_doc)

    return processed_doc


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_doc_type(processed_doc: Dict[str, Any], fallback: str = "unknown") -> str:
    return processed_doc.get("doc_type") or fallback


def build_case_kg(
    case_id: str,
    doc_paths: List[Path],
    global_entities_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a unified case-level knowledge graph JSON.

    The output structure:

        {
          "case_id": "CASE123",
          "global_entities": [...],
          "documents": [...],
          "nodes_local_entities": [...],
          "edges_relations": [...],
          "edges_events": [...]
        }
    """
    ner_nlp = setup_nlp_pipeline()
    coref_nlp = create_coref_pipeline()
    pool = GlobalEntityPool(case_id)

    global_entities: List[Dict[str, Any]] = []
    if global_entities_path and global_entities_path.is_file():
        try:
            loaded = _load_json(global_entities_path)
            if isinstance(loaded, list):
                global_entities = loaded
            elif isinstance(loaded, dict) and "global_entities" in loaded:
                val = loaded.get("global_entities")
                if isinstance(val, list):
                    global_entities = val
        except json.JSONDecodeError:
            global_entities = []

    documents_meta: List[Dict[str, Any]] = []
    nodes_local_entities: List[Dict[str, Any]] = []
    edges_relations: List[Dict[str, Any]] = []
    edges_events: List[Dict[str, Any]] = []

    for path in doc_paths:
        if not path.is_file():
            continue
        processed_doc = _load_json(path)
        doc_type = _infer_doc_type(processed_doc, fallback="unknown")

        # Simple name derived from file stem
        doc_name = path.stem

        documents_meta.append(
            {
                "name": doc_name,
                "path": str(path),
                "doc_type": doc_type,
            }
        )

        # Enrich the document if relations/events are not already present
        if not processed_doc.get("relations") or not processed_doc.get("events"):
            processed_doc = _enrich_document(processed_doc, doc_type, ner_nlp, coref_nlp)

        # Merge into global pool (for future use or inspection)
        integrate_processed_doc(pool, processed_doc, doc_name=doc_name, doc_type=doc_type)

        # Collect local entities
        for ent in processed_doc.get("entities", []):
            nodes_local_entities.append(
                {
                    "doc_name": doc_name,
                    "doc_type": doc_type,
                    "text": ent.get("text"),
                    "label": ent.get("label"),
                    "norm": ent.get("norm"),
                    "segment_index": ent.get("segment_index"),
                    "sentence_index": ent.get("sentence_index"),
                }
            )

        # Collect relations as edges
        for rel in processed_doc.get("relations", []):
            edges_relations.append(
                {
                    "doc_name": doc_name,
                    "doc_type": doc_type,
                    "subject": rel.get("subject"),
                    "predicate": rel.get("predicate"),
                    "object": rel.get("object"),
                    "sentence_id": rel.get("sentence_id"),
                }
            )

        # Collect events as edges
        for ev in processed_doc.get("events", []):
            edges_events.append(
                {
                    "doc_name": doc_name,
                    "doc_type": doc_type,
                    "description": ev.get("description"),
                    "subject": ev.get("subject"),
                    "predicate": ev.get("predicate"),
                    "object": ev.get("object"),
                    "sentence_id": ev.get("sentence_id"),
                    "time_raw": ev.get("time_raw"),
                    "time_resolved": ev.get("time_resolved"),
                }
            )

    # If global_entities_path not provided or broken, export from pool
    if not global_entities:
        global_entities = pool.export()

    return {
        "case_id": case_id,
        "global_entities": global_entities,
        "documents": documents_meta,
        "nodes_local_entities": nodes_local_entities,
        "edges_relations": edges_relations,
        "edges_events": edges_events,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a unified case-level knowledge graph JSON from processed documents."
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Case ID (e.g., CASE123).",
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        required=True,
        help="Paths to processed_doc JSON files (narrative, biography, interview).",
    )
    parser.add_argument(
        "--global-entities",
        help="Optional path to CASEID_global_entities.json.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional output path for the unified KG JSON. "
            "Default: {case_id}_case_kg.json in current directory."
        ),
    )

    args = parser.parse_args()

    case_id = args.case
    doc_paths = [Path(p) for p in args.docs]

    global_entities_path = None
    if args.global_entities:
        global_entities_path = Path(args.global_entities)

    kg = build_case_kg(case_id, doc_paths, global_entities_path=global_entities_path)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"{case_id}_case_kg.json")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(kg, f, indent=4, ensure_ascii=False)

    print(f"Unified case KG written to: {out_path}")


if __name__ == "__main__":
    main()