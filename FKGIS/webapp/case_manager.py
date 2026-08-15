"""Session / case management for the FKGIS web app.

Each "session" is one forensic case. A case is a directory on disk:

    sessions/
      cases.json                      # lightweight index (no database needed)
      <case_id>/
        case_docs/                    # uploaded document originals (.txt)
        work/                         # pipeline working copies (type-prefixed)
        output/                       # generated graph artifacts & summaries
        status.json                   # last pipeline run status / timing

The index is a single JSON file, which keeps setup trivial: no database
server, no ORM, no connection strings.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DOC_TYPES = ("narrative", "biography", "interview")

# Filename prefixes used so the pipeline's document discovery can infer the
# document type from the working copies.
TYPE_PREFIX = {
    "narrative": "IncidentReport_",
    "biography": "BIO_",
    "interview": "Interview_",
}


class CaseError(RuntimeError):
    """Raised for invalid case/directory operations."""


class CaseManager:
    def __init__(
        self,
        sessions_dir: str | Path,
        bundled_case_docs: Optional[str | Path] = None,
        bundled_outputs: Optional[str | Path] = None,
    ):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.sessions_dir / "cases.json"
        self.bundled_case_docs = Path(bundled_case_docs) if bundled_case_docs else None
        self.bundled_outputs = Path(bundled_outputs) if bundled_outputs else None
        self._cases: Dict[str, Dict[str, Any]] = self._load_index()

    # ------------------------------------------------------------------ #
    # Index persistence
    # ------------------------------------------------------------------ #
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if not self.index_path.exists():
            return {}
        try:
            with self.index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {str(k): v for k, v in data.items()}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_index(self) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._cases, f, indent=2, ensure_ascii=False)
        tmp.replace(self.index_path)

    # ------------------------------------------------------------------ #
    # Case CRUD
    # ------------------------------------------------------------------ #
    def list_cases(self) -> List[Dict[str, Any]]:
        cases = []
        for case_id, info in self._cases.items():
            if self.case_dir(case_id).exists():
                cases.append(self._public_case(info))
        cases.sort(key=lambda c: c.get("created_at", ""), reverse=True)
        return cases

    def get_case(self, case_id: str) -> Dict[str, Any]:
        self._require_case(case_id)
        return self._public_case(self._cases[case_id])

    def create_case(self, name: str, case_id: Optional[str] = None) -> Dict[str, Any]:
        name = (name or "Untitled Case").strip()
        case_id = (case_id or name).strip()
        case_id = re.sub(r"[^A-Za-z0-9_\-]+", "_", case_id).strip("_") or f"case_{uuid.uuid4().hex[:8]}"
        case_id = case_id.upper()
        if case_id in self._cases:
            raise CaseError(f"A case with id '{case_id}' already exists.")

        case_dir = self.case_dir(case_id)
        for sub in ("case_docs", "work", "output"):
            (case_dir / sub).mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()
        info = {
            "id": case_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "documents": [],
            "outputs": {},
            "status": "idle",
        }
        self._cases[case_id] = info
        self._save_index()
        return self._public_case(info)

    def delete_case(self, case_id: str) -> None:
        self._require_case(case_id)
        case_dir = self.case_dir(case_id)
        if case_dir.exists():
            shutil.rmtree(case_dir, ignore_errors=True)
        del self._cases[case_id]
        self._save_index()

    def create_sample_case(self) -> Dict[str, Any]:
        """Create (or reuse) a case pre-populated with the bundled sample docs."""
        case_id = "SAMPLE"
        if case_id in self._cases:
            return self.get_case(case_id)
        self.create_case("Sample Case (Kimberley Pace)", case_id=case_id)
        self._seed_sample_docs(case_id)
        return self.get_case(case_id)

    # ------------------------------------------------------------------ #
    # Documents
    # ------------------------------------------------------------------ #
    def add_document(self, case_id: str, filename: str, content: bytes, doc_type: str) -> Dict[str, Any]:
        """Save an uploaded document to the case and register it in the index."""
        self._require_case(case_id)
        doc_type = (doc_type or "").strip().lower()
        if doc_type not in DOC_TYPES:
            raise CaseError(f"doc_type must be one of {DOC_TYPES}, got '{doc_type}'.")

        stem = Path(filename).stem
        stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_") or "document"
        safe_name = f"{stem}.txt"

        docs_dir = self.case_docs_dir(case_id)
        docs_dir.mkdir(parents=True, exist_ok=True)
        target = docs_dir / safe_name
        with target.open("wb") as f:
            f.write(content)

        documents = self._cases[case_id].setdefault("documents", [])
        existing = [d for d in documents if d["name"] == safe_name]
        record = {"name": safe_name, "doc_type": doc_type, "size": len(content)}
        if existing:
            existing[0].update(record)
        else:
            documents.append(record)

        # A change to the document set invalidates prior outputs.
        self._cases[case_id]["outputs"] = {}
        self._cases[case_id]["status"] = "idle"
        self._cases[case_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_index()
        return record

    def remove_document(self, case_id: str, doc_name: str) -> None:
        self._require_case(case_id)
        documents = self._cases[case_id].get("documents", [])
        remaining = [d for d in documents if d["name"] != doc_name]
        if len(remaining) == len(documents):
            raise CaseError(f"Document '{doc_name}' not found in case '{case_id}'.")
        self._cases[case_id]["documents"] = remaining

        for path in (self.case_docs_dir(case_id) / doc_name,):
            if path.exists():
                path.unlink()

        self._cases[case_id]["outputs"] = {}
        self._cases[case_id]["status"] = "idle"
        self._cases[case_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_index()

    def list_documents(self, case_id: str) -> List[Dict[str, Any]]:
        self._require_case(case_id)
        return list(self._cases[case_id].get("documents", []))

    # ------------------------------------------------------------------ #
    # Pipeline support
    # ------------------------------------------------------------------ #
    def prepare_work_dir(self, case_id: str) -> Path:
        """Copy uploaded documents into the working dir with type prefixes.

        The pipeline discovers document type from the filename, so each copy
        is prefixed accordingly. Returns the working directory path.
        """
        self._require_case(case_id)
        work_dir = self.work_dir(case_id)
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        docs_dir = self.case_docs_dir(case_id)
        for record in self._cases[case_id].get("documents", []):
            name = record["name"]
            src = docs_dir / name
            if not src.exists():
                continue
            doc_type = record.get("doc_type", "narrative")
            prefix = TYPE_PREFIX.get(doc_type, "")
            # Avoid double-prefixing names that already carry their type prefix.
            if prefix and name.lower().startswith(prefix.lower()):
                target_name = name
            else:
                target_name = f"{prefix}{name}"
            shutil.copyfile(src, work_dir / target_name)
        return work_dir

    def mark_outputs(self, case_id: str, outputs: Dict[str, Any], status: str, timing: Optional[Dict[str, Any]] = None) -> None:
        self._require_case(case_id)
        self._cases[case_id]["outputs"] = outputs
        self._cases[case_id]["status"] = status
        if timing:
            self._cases[case_id]["timing"] = timing
        self._cases[case_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_index()

    def get_outputs(self, case_id: str) -> Dict[str, Any]:
        self._require_case(case_id)
        return dict(self._cases[case_id].get("outputs", {}))

    # ------------------------------------------------------------------ #
    # Paths / helpers
    # ------------------------------------------------------------------ #
    def case_dir(self, case_id: str) -> Path:
        return self.sessions_dir / case_id

    def case_docs_dir(self, case_id: str) -> Path:
        return self.case_dir(case_id) / "case_docs"

    def work_dir(self, case_id: str) -> Path:
        return self.case_dir(case_id) / "work"

    def output_dir(self, case_id: str) -> Path:
        return self.case_dir(case_id) / "output"

    def status_path(self, case_id: str) -> Path:
        return self.case_dir(case_id) / "status.json"

    def _require_case(self, case_id: str) -> None:
        if case_id not in self._cases:
            raise CaseError(f"Case '{case_id}' does not exist.")

    def _public_case(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(info)
        info["documents"] = list(info.get("documents", []))
        info["outputs"] = dict(info.get("outputs", {}))
        return info

    # ------------------------------------------------------------------ #
    # Seed
    # ------------------------------------------------------------------ #
    def _seed_sample_docs(self, case_id: str) -> None:
        """Populate the sample case from bundled docs and pre-generated outputs."""
        if not self.bundled_case_docs or not self.bundled_case_docs.is_dir():
            raise CaseError("No bundled sample documents are available for seeding.")
        docs_dir = self.case_docs_dir(case_id)
        for src in sorted(self.bundled_case_docs.glob("*.txt")):
            shutil.copyfile(src, docs_dir / src.name)
            name = src.name
            if name.lower().startswith("bio_"):
                doc_type = "biography"
            elif name.lower().startswith("interview_"):
                doc_type = "interview"
            else:
                doc_type = "narrative"
            self._cases[case_id].setdefault("documents", []).append(
                {"name": name, "doc_type": doc_type, "size": src.stat().st_size}
            )

        outputs_dir = self.output_dir(case_id)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        outputs: Dict[str, Any] = {}
        for artifact in ("graph_nodes.json", "graph_edges.json",
                         "graph_nodes_refined.json", "graph_edges_refined.json",
                         "pagerank_results.json", "case_output.json"):
            src = self.bundled_outputs / artifact if self.bundled_outputs else None
            if src and src.exists():
                shutil.copyfile(src, outputs_dir / artifact)
                outputs[artifact] = str(outputs_dir / artifact)

        # Read stats from the pre-generated case summary if present.
        case_output_path = outputs_dir / "case_output.json"
        if case_output_path.exists():
            with case_output_path.open("r", encoding="utf-8") as f:
                case_output = json.load(f)
            outputs["case_output"] = case_output

        if (outputs_dir / "pagerank_results.json").exists():
            outputs["has_pagerank"] = True

        self._cases[case_id]["outputs"] = outputs
        self._cases[case_id]["status"] = "ready"
        self._cases[case_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_index()