from __future__ import annotations

from typing import Dict, Any, List
import re

from .mention_unifier import unify_mentions as _base_unify_mentions

# Case-specific abbreviation expansions that may still appear after coref/unification.
# These are tailored to the Kimberly Pace incident report.
ABBREV_PATTERNS = [
    # Reporting Officer / Investigators / Crime Scene Unit
    (re.compile(r"\bR/O\b"), "Reporting Officer"),
    (re.compile(r"\bR/Os\b"), "Reporting Officers"),
    (re.compile(r"\bR/I\b"), "Reporting Investigator"),
    (re.compile(r"\bR/Is\b"), "Reporting Investigators"),
    (re.compile(r"\bCSU\b"), "Crime Scene Unit"),
    (re.compile(r"\bETA\b"), "Estimated Time of Arrival"),
]


def _expand_abbreviations(text: str) -> str:
    """Expand common incident-specific abbreviations in a sentence."""
    new_text = text
    for pattern, repl in ABBREV_PATTERNS:
        new_text = pattern.sub(repl, new_text)
    return new_text


def _postprocess_unified_sentences(unified_sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply case-specific surface normalizations on unified sentences.

    Currently:
      - Expand R/O, R/I, R/Is, CSU, ETA.
      - Keep all other fields intact.
    """
    out: List[Dict[str, Any]] = []
    for sent in unified_sentences:
        s = dict(sent)
        text = s.get("text", "")
        s["text"] = _expand_abbreviations(text)
        out.append(s)
    return out


def unify_mentions(processed_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper around the generic mention_unifier with case-specific tweaks.

    Steps:
      1) Call the base unify_mentions() to perform pronoun substitution based on
         processed_doc["coref_entities"].
      2) Apply abbreviation expansion on the resulting unified sentence texts,
         so that patterns like "R/O" become "Reporting Officer" and "R/Is"
         become "Reporting Investigators".
    """
    processed_doc = _base_unify_mentions(processed_doc)

    unified = processed_doc.get("unified_sentences") or []
    if unified:
        processed_doc["unified_sentences"] = _postprocess_unified_sentences(unified)

    return processed_doc