from __future__ import annotations

from typing import Dict, Any, List


def _build_sentence_time_index(processed_doc: Dict[str, Any]) -> Dict[int, Any]:
    """Map sentence_id (index in processed_doc['sentences']) to a raw time value.

    We derive times from segments:
      - For each segment, use segment['time'] (if present).
      - Apply that same time value to all sentences within the segment,
        in the same flattening order used by preprocessing.
    """
    segments = processed_doc.get("segments") or []
    times_by_sentence_id: Dict[int, Any] = {}

    sentence_id = 0
    for seg in segments:
        seg_time = seg.get("time")
        for _sent in seg.get("sentences", []):
            if seg_time is not None:
                times_by_sentence_id[sentence_id] = seg_time
            sentence_id += 1

    return times_by_sentence_id


def build_events(processed_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert relation triples into simple event objects.

    Behavior:
      - For each relation triple in processed_doc['relations']:
          * Create an event with:
                description: a human-readable summary
                subject, predicate, object: copied from the triple
                sentence_id: index of the originating sentence
                time_raw: a raw time value, if the sentence is associated with one
                time_resolved: None (for future timeline normalization)
      - Time is derived from the sentence's segment:
          * If the sentence_id maps to a segment time, use that as time_raw.
          * Otherwise, time_raw is None.

    Output format:

        processed_doc["events"] = [
            {
               "description": "Officer Willits secured the scene",
               "subject": "...",
               "predicate": "...",
               "object": "...",
               "sentence_id": ...,
               "time_raw": "... or None ...",
               "time_resolved": None
            },
            ...
        ]
    """
    relations: List[Dict[str, Any]] = processed_doc.get("relations") or []
    sentences: List[Dict[str, Any]] = processed_doc.get("sentences") or []

    if not relations or not sentences:
        processed_doc["events"] = []
        return processed_doc

    times_by_sentence_id = _build_sentence_time_index(processed_doc)

    events: List[Dict[str, Any]] = []

    for rel in relations:
        sentence_id = rel.get("sentence_id")
        if sentence_id is None:
            continue
        if not isinstance(sentence_id, int):
            try:
                sentence_id = int(sentence_id)
            except (TypeError, ValueError):
                continue

        if sentence_id < 0 or sentence_id >= len(sentences):
            continue

        sent = sentences[sentence_id]
        sentence_text = sent.get("text") or ""

        subject = rel.get("subject", "")
        predicate = rel.get("predicate", "")
        obj = rel.get("object", "")

        if sentence_text:
            description = sentence_text
        else:
            description_parts = [subject, predicate, obj]
            description = " ".join(p for p in description_parts if p).strip()

        time_raw = times_by_sentence_id.get(sentence_id)

        event = {
            "description": description,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "sentence_id": sentence_id,
            "time_raw": time_raw,
            "time_resolved": None,
        }
        events.append(event)

    processed_doc["events"] = events
    return processed_doc