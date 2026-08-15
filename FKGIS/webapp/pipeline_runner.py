"""Runs the FKGIS NLP pipeline for a single case session.

The runner is intentionally import-light at module load time: the heavyweight
spaCy/LLM dependencies are imported lazily inside ``run()`` so the web app can
start and serve existing results even on machines where the full NLP stack is
not yet installed.

Progress and status are written to ``<case>/status.json`` so the frontend can
poll without any long-lived server-side state.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .case_manager import CaseManager

ARTIFACT_NAMES = (
    "graph_nodes.json",
    "graph_edges.json",
    "graph_nodes_refined.json",
    "graph_edges_refined.json",
    "pagerank_results.json",
    "case_output.json",
)


class PipelineRunner:
    def __init__(
        self,
        manager: CaseManager,
        gemini_api_key: str = "",
        gemini_model: str = "gemini-2.0-flash",
        use_llm: bool = False,
    ):
        self.manager = manager
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.use_llm = use_llm

    # ------------------------------------------------------------------ #
    # Status helpers
    # ------------------------------------------------------------------ #
    def _write_status(self, case_id: str, state: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        status_path = self.manager.status_path(case_id)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status = {
            "case_id": case_id,
            "state": state,
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            status.update(extra)
        with status_path.open("w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)

    def read_status(self, case_id: str) -> Dict[str, Any]:
        status_path = self.manager.status_path(case_id)
        if not status_path.exists():
            return {"case_id": case_id, "state": "idle", "message": "No run started yet."}
        with status_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(self, case_id: str) -> Dict[str, Any]:
        """Run the full pipeline for a case and refresh its artifacts.

        Returns the pipeline summary dict (same shape as ``run_case_pipeline``).
        """
        self._write_status(case_id, "running", "Starting pipeline...")
        started = time.time()
        try:
            work_dir = self.manager.prepare_work_dir(case_id)
            output_dir = self.manager.output_dir(case_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Import lazily so the app can serve pre-existing results even
            # when the NLP stack is not installed yet.
            from ..nlp_pipeline.pipeline import run_case_pipeline

            def _progress(stage: str, message: str) -> None:
                self._write_status(case_id, "running", message, {"stage": stage})

            summary = run_case_pipeline(
                case_id,
                docs_dir=str(work_dir),
                output_dir=str(output_dir),
                progress_callback=_progress,
            )

            self._write_status(case_id, "running", "Refining graph (optional)...", {"stage": "refine"})
            self._maybe_refine(case_id, output_dir)

            self._write_status(case_id, "running", "Computing PageRank...", {"stage": "pagerank"})
            self._compute_pagerank(case_id, output_dir)

            timing = summary.get("timing", {})
            timing["total_pipeline_time"] = time.time() - started

            outputs = self._collect_outputs(case_id)
            self.manager.mark_outputs(case_id, outputs, "ready", timing)

            self._write_status(case_id, "done", "Pipeline completed.", {"timing": timing})
            return summary
        except Exception as exc:  # noqa: BLE001
            self._write_status(case_id, "error", f"Pipeline failed: {exc}")
            self.manager.mark_outputs(case_id, {}, "error")
            raise

    # ------------------------------------------------------------------ #
    # Pipeline stages
    # ------------------------------------------------------------------ #
    def _maybe_refine(self, case_id: str, output_dir: Path) -> None:
        """LLM-refine the graph artifacts when a Gemini key is available.

        Otherwise a cheap deterministic dedupe is applied so that the
        ``*_refined`` artifacts always exist and can be compared in the UI.
        """
        raw_nodes = output_dir / "graph_nodes.json"
        raw_edges = output_dir / "graph_edges.json"
        if not raw_nodes.exists() or not raw_edges.exists():
            return

        if self.use_llm and self.gemini_api_key:
            from ..nlp_pipeline.output.refining import refine_edges_parallel, refine_nodes_parallel

            refine_nodes_parallel(
                str(raw_nodes),
                str(output_dir / "graph_nodes_refined.json"),
                use_llm=True,
            )
            refine_edges_parallel(
                str(raw_edges),
                str(output_dir / "graph_edges_refined.json"),
                use_llm=True,
            )
        else:
            # Deterministic fallback: dedupe by (id) / (source, target, type).
            refine_nodes_parallel(
                str(raw_nodes),
                str(output_dir / "graph_nodes_refined.json"),
                use_llm=False,
            )
            refine_edges_parallel(
                str(raw_edges),
                str(output_dir / "graph_edges_refined.json"),
                use_llm=False,
            )

    def _compute_pagerank(self, case_id: str, output_dir: Path) -> None:
        from ..nlp_pipeline.kg.pagerank_analysis import save_pagerank_results

        with (output_dir / "graph_nodes.json").open("r", encoding="utf-8") as f:
            nodes = json.load(f)
        with (output_dir / "graph_edges.json").open("r", encoding="utf-8") as f:
            edges = json.load(f)
        save_pagerank_results(nodes, edges, output_dir / "pagerank_results.json")

    def _collect_outputs(self, case_id: str) -> Dict[str, Any]:
        output_dir = self.manager.output_dir(case_id)
        outputs: Dict[str, Any] = {}
        for artifact in ARTIFACT_NAMES:
            path = output_dir / artifact
            if path.exists():
                outputs[artifact] = str(path)

        case_output_path = output_dir / "case_output.json"
        if case_output_path.exists():
            with case_output_path.open("r", encoding="utf-8") as f:
                outputs["case_output"] = json.load(f)
        return outputs