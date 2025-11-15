from __future__ import annotations

from typing import Dict, Any, List, Set

from ..event_construction.event_builder import build_events as _base_build_events

# Entity labels we treat as "participants" in events
PARTICIPANT_LABELS: Set[str] = {"PERSON", "ROLE", "ORG", "EVIDENCE"}

# Location-like tokens/entities we pay special attention to
LOCATION_KEYWORDS = {
    "residence",
    "house",
    "home",
    "basement",
    "stairs",
    "staircase",
    "bedroom",
    "living room",
    "kitchen",
    "bathroom",
    "porch",
    "gallery",
    "diner",
    "Big Bad Breakfast".lower(),
    "The Lucky Café".lower(),
    "C'est Belle".lower(),
}


def _collect_sentence_entities(processed_doc: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Map sentence_id -> list of entity dicts for that sentence."""
    entities = processed_doc.get("entities") or []
    segments = processed_doc.get("segments") or []

    # Build mapping from (seg_idx, sent_idx) to global sentence_id
    seg_sent_to_sid: Dict[tuple[int, int], int] = {}
    sid = 0
    for seg_idx, seg in enumerate(segments):
        for sent_idx, _ in enumerate(seg.get("sentences", [])):
            seg_sent_to_sid[(seg_idx, sent_idx)] = sid
            sid += 1

    by_sentence: Dict[int, List[Dict[str, Any]]] = {}
    for ent in entities:
        seg_idx = ent.get("segment_index")
        sent_idx = ent.get("sentence_index")
        if seg_idx is None or sent_idx is None:
            continue
        key = (seg_idx, sent_idx)
        if key not in seg_sent_to_sid:
            continue
        s_id = seg_sent_to_sid[key]
        by_sentence.setdefault(s_id, []).append(ent)

    return by_sentence


def _attach_participants_and_locations(
    processed_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Enrich processed_doc['events'] with case-specific participants and locations.

    For each event:
      - Determine which entities are involved based on subject/object/description.
      - Mark 'participants' as a list of entity texts (PERSON/ROLE/ORG/EVIDENCE).
      - Mark 'locations' as a list of location-like entity texts or keywords.
    """
    events: List[Dict[str, Any]] = processed_doc.get("events") or []
    if not events:
        return processed_doc

    sentence_entities = _collect_sentence_entities(processed_doc)
    sentences = processed_doc.get("sentences") or []

    for ev in events:
        sid = ev.get("sentence_id")
        participants: List[str] = []
        locations: List[str] = []

        ev_text = " ".join(
            part
            for part in [
                ev.get("description") or "",
                ev.get("subject") or "",
                ev.get("object") or "",
            ]
            if part
        ).lower()

        # Entity-based participants and locations
        if isinstance(sid, int) and 0 <= sid < len(sentences):
            ents_here = sentence_entities.get(sid, [])
            for ent in ents_here:
                text = ent.get("text")
                label = ent.get("label", "")
                if not text:
                    continue
                low = text.lower()
                if label in PARTICIPANT_LABELS and text not in participants:
                    participants.append(text)
                # Locations from LOC entities or ORG/ROLE that look like places
                if label in {"LOC"} or any(kw in low for kw in LOCATION_KEYWORDS):
                    if text not in locations:
                        locations.append(text)

        # Keyword-based locations from raw event text
        for kw in LOCATION_KEYWORDS:
            if kw in ev_text and kw not in [l.lower() for l in locations]:
                locations.append(kw)

        ev["participants"] = participants
        ev["locations"] = locations

    return processed_doc


def _assign_event_ids(processed_doc: Dict[str, Any], doc_name: str) -> Dict[str, Any]:
    """Ensure each event has a stable 'event_id' field, prefixed by doc_name.

    This prevents event ID collisions across different documents in the case.
    """
    events: List[Dict[str, Any]] = processed_doc.get("events") or []
    for idx, ev in enumerate(events):
        ev.setdefault("event_id", f"{doc_name}_EV{idx+1}")
    return processed_doc


def build_events(processed_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper around the base event builder with case-specific enrichment.

    Steps:
      1) Call the generic event_construction.event_builder.build_events() to
         create baseline events from relations + segment times.
      2) Attach 'participants' and 'locations' based on entities and keywords.
      3) Assign stable event IDs (EV1, EV2, ...).
    """
    processed_doc = _base_build_events(processed_doc)
    processed_doc = _attach_participants_and_locations(processed_doc)
    # Pass doc_name to ensure unique event IDs across documents
    doc_name = processed_doc.get("doc_name", "unknown_doc")
    processed_doc = _assign_event_ids(processed_doc, doc_name=doc_name)
    return processed_doc