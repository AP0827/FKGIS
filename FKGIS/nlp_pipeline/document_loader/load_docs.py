from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional


CASE_DOCS_DIR = Path(__file__).resolve().parent.parent / "case_docs"


@dataclass
class CaseDocument:
    """Lightweight descriptor for a raw case document.

    This is intentionally simple: it just records where the file is and
    what type of document it is (narrative / interview / biography), so
    that later pipeline stages can decide which preprocessor to use.
    """

    doc_id: str
    doc_type: str  # "narrative" | "interview" | "biography"
    title: str
    path: str


def _detect_doc_type(path: Path) -> Optional[str]:
    """Infer document type from filename and content conventions.

    Heuristics:
      - IncidentReport_RO.txt        -> "narrative"
      - BIO_*.txt                    -> "biography"
      - Interview_*.txt              -> "interview"
    """
    name = path.name.lower()
    if "incidentreport" in name or "narrative" in name:
        return "narrative"
    if name.startswith("bio_"):
        return "biography"
    if name.startswith("interview_"):
        return "interview"
    return None


def _extract_title(path: Path) -> str:
    """Extract a simple title from the first non-empty line."""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    return stripped
    except OSError:
        pass
    return path.stem


def discover_case_docs(base_dir: Optional[str] = None) -> List[CaseDocument]:
    """Discover all case documents under nlp_pipeline/case_docs/.

    Returns a list of CaseDocument descriptors that can be fed into
    downstream preprocessing modules.

    Each descriptor only contains metadata (no full text), to keep
    this stage lightweight and deterministic.
    """
    base = Path(base_dir) if base_dir else CASE_DOCS_DIR
    docs: List[CaseDocument] = []

    if not base.is_dir():
        return docs

    for path in sorted(base.glob("*.txt")):
        doc_type = _detect_doc_type(path)
        if not doc_type:
            # Skip unknown types for now; can be extended later.
            continue
        title = _extract_title(path)
        doc_id = path.stem
        docs.append(
            CaseDocument(
                doc_id=doc_id,
                doc_type=doc_type,
                title=title,
                path=str(path),
            )
        )

    return docs


def docs_to_jsonable(docs: List[CaseDocument]) -> List[Dict[str, Any]]:
    """Convert a list of CaseDocument objects into JSON-serializable dicts."""
    return [asdict(d) for d in docs]


def save_manifest(
    docs: List[CaseDocument],
    output_path: str,
    case_id: Optional[str] = None,
) -> str:
    """Save a simple manifest JSON summarizing case documents.

    Example structure:

        {
          "case_id": "CASE123",
          "docs": [
            {"doc_id": "...", "doc_type": "narrative", "title": "...", "path": "..."},
            ...
          ]
        }
    """
    manifest = {
        "case_id": case_id,
        "docs": docs_to_jsonable(docs),
    }
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    return str(out_path)


def load_manifest(path: str) -> Dict[str, Any]:
    """Load a saved manifest JSON file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """CLI helper to list case documents and optionally save a manifest.

    Usage:

        python -m FKGIS.nlp_pipeline.document_loader.load_docs
        python -m FKGIS.nlp_pipeline.document_loader.load_docs --output nlp_pipeline/output/docs_manifest.json --case CASE123
    """
    import argparse

    parser = argparse.ArgumentParser(description="Discover case documents under nlp_pipeline/case_docs.")
    parser.add_argument(
        "--base-dir",
        help="Optional base directory to search instead of the default case_docs/.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save a JSON manifest of discovered documents.",
    )
    parser.add_argument(
        "--case",
        dest="case_id",
        help="Optional case ID to include in the manifest.",
    )

    args = parser.parse_args()

    docs = discover_case_docs(base_dir=args.base_dir)
    if not docs:
        print("No case documents found.")
    else:
        print("Discovered documents:")
        for d in docs:
            print(f"  - {d.doc_id} ({d.doc_type}) @ {d.path}")

    if args.output:
        path = save_manifest(docs, args.output, case_id=args.case_id)
        print(f"Manifest saved to: {path}")


if __name__ == "__main__":
    main()