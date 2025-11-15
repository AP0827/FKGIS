from __future__ import annotations

from typing import Dict, Any, List

from .basic_rel_extractor import extract_relations as _base_extract_relations


# Case-specific verb lexicon (lemmas) for high-precision relations in the
# Kimberly Pace incident:
#
#   arrived, secured, observed, removed, pronounced, responded, identified,
#   met, directed, interviewed, examined
#
# Plus a few additional verbs that are important in this case narrative:
#   reported, said, indicated, called, dispatched
CASE_VERB_LEXICON = {
    "arrive",
    "secure",
    "observe",
    "remove",
    "pronounce",
    "respond",
    "identify",
    "meet",
    "direct",
    "interview",
    "examine",
    "report",
    "say",
    "indicate",
    "call",
    "dispatch",
}


VERB_TO_RELATION_TYPE = {
    "arrive": "ARRIVED_AT",
    "secure": "SECURED",
    "observe": "OBSERVED",
    "remove": "REMOVED",
    "pronounce": "PRONOUNCED",
    "respond": "RESPONDED",
    "identify": "IDENTIFIED",
    "meet": "MET",
    "direct": "DIRECTED",
    "interview": "INTERVIEWED",
    "examine": "EXAMINED",
    "report": "CLAIMED",
    "say": "CLAIMED",
    "indicate": "CLAIMED",
    "call": "DISPATCHED",
    "dispatch": "DISPATCHED",
}


def _normalize_predicate(pred: str) -> str:
    """Normalize predicate text to a lemma-like form for lexicon matching.

    The base extractor already uses verb.lemma_ when available, but we
    defensively lowercase and strip here.
    """
    return (pred or "").strip().lower()


def refine_case_relations(processed_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Refine relations for the Kimberly Pace case using a verb lexicon.

    Steps:
      1) Call the generic basic_rel_extractor.extract_relations() to populate
         processed_doc["relations"] with high-recall SPO triples.
      2) Filter relations to keep only those whose predicate lemma is in the
         case-specific verb lexicon, or whose subject/object clearly involve
         core case actors (officers, witnesses, victim, dog, etc.).
      3) For retained relations, attach a "relation_type" field using
         VERB_TO_RELATION_TYPE, when known.

    This function overwrites processed_doc["relations"] with the refined list.
    """
    processed_doc = _base_extract_relations(processed_doc)

    raw_relations: List[Dict[str, Any]] = processed_doc.get("relations") or []
    refined: List[Dict[str, Any]] = []

    # Key case actors / objects for backstop when predicate isn't in lexicon
    KEY_ACTORS = {
        "willits",
        "harding",
        "armstrong",
        "murphy",
        "johnson",
        "lukens",
        "sanchez",
        "rebecca pace",
        "becky pace",
        "cheryl weston",
        "jeremy gladwell",
        "kimberly pace",
        "kim",
        "thoreau",
        "dog",
        "body",
        "victim",
    }

    for rel in raw_relations:
        subj = (rel.get("subject") or "").strip()
        obj = (rel.get("object") or "").strip()
        pred = (rel.get("predicate") or "").strip()

        if not subj or not obj or not pred:
            continue

        pred_norm = _normalize_predicate(pred)

        keep = False

        # 1) Primary: verb in case lexicon
        if pred_norm in CASE_VERB_LEXICON:
            keep = True
        else:
            # 2) Secondary: subject or object mention key case actors / objects
            s_low = subj.lower()
            o_low = obj.lower()
            if any(name in s_low for name in KEY_ACTORS) or any(
                name in o_low for name in KEY_ACTORS
            ):
                keep = True

        if not keep:
            continue

        rel_out = dict(rel)
        rel_out["predicate"] = pred_norm  # store normalized predicate
        rel_type = VERB_TO_RELATION_TYPE.get(pred_norm)
        if rel_type:
            rel_out["relation_type"] = rel_type

        refined.append(rel_out)

    processed_doc["relations"] = refined
    return processed_doc