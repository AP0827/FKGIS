from __future__ import annotations

import json
from pathlib import Path

from FKGIS.generate_results import generate_results_file
from FKGIS.nlp_pipeline.document_loader.load_docs import discover_case_docs


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "FKGIS" / "nlp_pipeline" / "output"


def test_case_docs_discovery_finds_expected_documents() -> None:
    docs = discover_case_docs(str(OUTPUT_DIR.parent / "case_docs"))

    assert [doc.doc_id for doc in docs] == [
        "BIO_Kimberley_Pace",
        "IncidentReport_RO",
        "Interview_Kimberley_Friend",
        "Interview_Kimberley_Neighbor",
        "Interview_Kimberley_Sister",
    ]
    assert [doc.doc_type for doc in docs] == ["biography", "narrative", "interview", "interview", "interview"]


def test_generate_results_uses_repository_outputs(tmp_path) -> None:
    output_path = tmp_path / "results.json"
    generate_results_file(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["graph_statistics"]["total_nodes"] == 270
    assert payload["graph_statistics"]["total_edges"] == 149
    assert payload["graph_statistics"]["nullish_entities"] == 111
    assert payload["pagerank_analysis"]["significant_nodes_count"] == 44
    assert payload["relation_analysis"]["total_relations"] == 149
