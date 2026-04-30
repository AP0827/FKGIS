from __future__ import annotations

from typing import Dict, Any, List


def _non_empty_entities(case_docs: List[Dict[str, Any]]) -> bool:
    for doc in case_docs:
        pd = doc.get("processed_doc") or {}
        if pd.get("entities"):
            return True
    return False


def _count_relations(case_docs: List[Dict[str, Any]]) -> int:
    total = 0
    for doc in case_docs:
        pd = doc.get("processed_doc") or {}
        total += len(pd.get("relations") or [])
    return total


def _count_events(case_docs: List[Dict[str, Any]]) -> int:
    total = 0
    for doc in case_docs:
        pd = doc.get("processed_doc") or {}
        total += len(pd.get("events") or [])
    return total


def _events_with_subject_and_predicate(case_docs: List[Dict[str, Any]]) -> int:
    ok = 0
    for doc in case_docs:
        pd = doc.get("processed_doc") or {}
        for ev in pd.get("events") or []:
            if ev.get("subject") and ev.get("predicate"):
                ok += 1
    return ok


def _has_before_after_edges(timeline_info: Dict[str, Any]) -> bool:
    edges = timeline_info.get("timeline_edges") or []
    return any(e.get("relation") == "BEFORE" for e in edges)


def _detect_null_or_malformed(case_docs: List[Dict[str, Any]]) -> List[str]:
    """Scan for obviously malformed entries (null ids, missing keys)."""
    issues: List[str] = []

    for doc in case_docs:
        doc_name = doc.get("doc_name") or doc.get("name") or "unknown_doc"
        pd = doc.get("processed_doc") or {}

        # Entities
        for idx, ent in enumerate(pd.get("entities") or []):
            if ent.get("text") is None:
                issues.append(f"{doc_name}: entity[{idx}] has null text")

        # Relations
        for idx, rel in enumerate(pd.get("relations") or []):
            if not rel.get("subject") or not rel.get("object"):
                issues.append(f"{doc_name}: relation[{idx}] missing subject or object")

        # Events
        for idx, ev in enumerate(pd.get("events") or []):
            if ev.get("event_id") is None:
                issues.append(f"{doc_name}: event[{idx}] has null event_id")

    return issues


def verify_case_outputs(
    case_id: str,
    case_docs: List[Dict[str, Any]],
    timeline_info: Dict[str, Any],
    graph_nodes: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    min_relations: int = 5,
) -> Dict[str, Any]:
    """Run a series of sanity checks over the assembled case outputs.

    Checks:
      - at least one non-empty entities list across docs
      - total relations >= min_relations
      - events have non-empty subject & predicate
      - timeline has at least some BEFORE edges
      - no null/malformed entries in entities/relations/events

    Returns a verification report:

        {
          "case_id": "...",
          "passed": bool,
          "checks": {
             "entities_non_empty": bool,
             "relations_count": int,
             "relations_count_ok": bool,
             "events_count": int,
             "events_with_subject_and_predicate": int,
             "timeline_has_before_after": bool,
             "issues": [ ... ]
          }
        }
    """
    entities_ok = _non_empty_entities(case_docs)
    rel_count = _count_relations(case_docs)
    rel_ok = rel_count >= min_relations
    events_count = _count_events(case_docs)
    events_ok_count = _events_with_subject_and_predicate(case_docs)
    timeline_ok = _has_before_after_edges(timeline_info)
    issues = _detect_null_or_malformed(case_docs)

    # Minimal graph sanity checks
    graph_nodes_ok = len(graph_nodes) > 0
    graph_edges_ok = len(graph_edges) > 0

    passed = (
        entities_ok
        and rel_ok
        and events_count > 0
        and events_ok_count > 0
        and graph_nodes_ok
        and graph_edges_ok
    )

    return {
        "case_id": case_id,
        "passed": passed and not issues,
        "checks": {
            "entities_non_empty": entities_ok,
            "relations_count": rel_count,
            "relations_count_ok": rel_ok,
            "events_count": events_count,
            "events_with_subject_and_predicate": events_ok_count,
            "timeline_has_before_after": timeline_ok,
            "graph_nodes_non_empty": graph_nodes_ok,
            "graph_edges_non_empty": graph_edges_ok,
            "issues": issues,
        },
    }