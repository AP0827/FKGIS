from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import re


# This module builds a simple, deterministic timeline for the Kimberly Pace
# case, based on explicit timestamps in the narrative and interviews.
#
# It is intentionally rule-based and case-specific:
#   - Assumes incident date is 2022-10-16 (Sunday, October 16, 2022).
#   - Resolves times like "11:06 a.m.", "11:42 a.m.", "12:53 p.m.", "4:35 p.m.",
#     "12:03 p.m.", "12:44 p.m.", "9:30 p.m." into ISO 8601 timestamps.
#   - Builds BEFORE edges between chronologically ordered events.
#
# It does not handle vague expressions like "later that morning" beyond
# leaving time_resolved = None.


INCIDENT_DATE_STR = "2022-10-16"
TIME_PATTERN = re.compile(
    r"\b(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.|am|pm)\b", flags=re.IGNORECASE
)


def _normalize_time_raw(time_raw: str) -> Optional[str]:
    """Normalize raw time expressions into ISO 8601 strings on INCIDENT_DATE_STR.

    Returns:
        ISO timestamp string (e.g., "2022-10-16T11:06:00") or None if parsing fails.
    """
    if not time_raw:
        return None

    m = TIME_PATTERN.search(time_raw)
    if not m:
        return None

    hour = int(m.group(1))
    minute = int(m.group(2))
    meridiem = m.group(3).lower().replace(".", "")

    if meridiem in {"pm"} and hour != 12:
        hour += 12
    if meridiem in {"am"} and hour == 12:
        hour = 0

    try:
        dt = datetime.strptime(INCIDENT_DATE_STR, "%Y-%m-%d")
        dt = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return dt.isoformat()
    except ValueError:
        return None


def normalize_event_times_for_doc(processed_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize time_raw for each event in a single processed_doc.

    Sets event["time_resolved"] when time_raw can be parsed, leaves it
    unchanged otherwise.
    """
    events: List[Dict[str, Any]] = processed_doc.get("events") or []
    for ev in events:
        time_raw = ev.get("time_raw")
        if time_raw:
            iso = _normalize_time_raw(str(time_raw))
            if iso:
                ev["time_resolved"] = iso
    return processed_doc


def _collect_timed_events(case_docs: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Collect (doc_name, doc_type, event) for events with time_resolved."""
    timed: List[Tuple[str, str, Dict[str, Any]]] = []
    for doc_entry in case_docs:
        doc_name = doc_entry.get("doc_name") or doc_entry.get("name") or "unknown_doc"
        doc_type = doc_entry.get("doc_type") or "unknown"
        processed_doc = doc_entry.get("processed_doc") or {}
        for ev in processed_doc.get("events", []):
            if ev.get("time_resolved"):
                timed.append((doc_name, doc_type, ev))
    return timed


def build_before_after_edges(case_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build BEFORE/AFTER edges between events based on time_resolved.

    Strategy:
      - Aggregate all events across documents that have time_resolved.
      - Sort them ascending by time_resolved.
      - For each consecutive pair (e_i, e_{i+1}), add:
            { "source_event_id": e_i["event_id"], "target_event_id": e_{i+1]["event_id"], "relation": "BEFORE" }
        The reverse AFTER edges can be derived later if needed.
    """
    timed = _collect_timed_events(case_docs)
    if not timed:
        return []

    # Convert time_resolved to datetime for sorting
    sortable: List[Tuple[datetime, str, str, Dict[str, Any]]] = []
    for doc_name, doc_type, ev in timed:
        try:
            ts = datetime.fromisoformat(ev["time_resolved"])
        except (KeyError, ValueError, TypeError):
            continue
        sortable.append((ts, doc_name, doc_type, ev))

    sortable.sort(key=lambda x: x[0])

    edges: List[Dict[str, Any]] = []
    for i in range(len(sortable) - 1):
        _, doc_name_src, doc_type_src, ev_src = sortable[i]
        _, doc_name_tgt, doc_type_tgt, ev_tgt = sortable[i + 1]

        src_id = ev_src.get("event_id")
        tgt_id = ev_tgt.get("event_id")
        if not src_id or not tgt_id:
            continue

        edges.append(
            {
                "source_event_id": src_id,
                "target_event_id": tgt_id,
                "relation": "BEFORE",
                "source_doc": doc_name_src,
                "target_doc": doc_name_tgt,
                "source_doc_type": doc_type_src,
                "target_doc_type": doc_type_tgt,
            }
        )

    return edges


def build_timeline_for_case(case_id: str, case_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build timeline information for a case.

    Args:
        case_id: Identifier for the case (e.g., "CASE123").
        case_docs: List of entries of the form:
            {
              "doc_name": str,
              "doc_type": str,
              "processed_doc": { ... }
            }

    Returns:
        Dict with:
          {
            "case_id": ...,
            "timeline_edges": [... BEFORE edges ...]
          }

    Side-effect:
        - Each processed_doc in case_docs is updated in-place with normalized
          time_resolved values for events.
    """
    # Normalize times per document
    for doc_entry in case_docs:
        processed_doc = doc_entry.get("processed_doc")
        if processed_doc:
            normalize_event_times_for_doc(processed_doc)

    edges = build_before_after_edges(case_docs)

    return {
        "case_id": case_id,
        "timeline_edges": edges,
    }