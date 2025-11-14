from __future__ import annotations

from typing import Dict, Any, List
import re

# Basic pronoun inventory for substitution (lowercased)
PRONOUNS = {
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
    "it",
    "its",
}


def _choose_main_mention(cluster: Dict[str, Any]) -> str | None:
    """Pick a main mention string from a coref cluster.

    Strategy:
      - Prefer the longest mention that is not a pure pronoun.
      - If all mentions are pronouns, return None (skip substitution for this cluster).
    """
    mentions: List[str] = cluster.get("mentions") or []
    if not mentions:
        return None

    # Normalize mentions to strings
    normed: List[str] = []
    for m in mentions:
        if isinstance(m, str):
            normed.append(m)
        elif isinstance(m, dict):
            text = m.get("text")
            if isinstance(text, str):
                normed.append(text)
        # ignore other types

    if not normed:
        return None

    # Helper to check if a mention is just a pronoun token
    def is_pronoun(token: str) -> bool:
        return token.lower() in PRONOUNS

    # Prefer non-pronoun mentions, sorted by length desc
    non_pronoun = [m for m in normed if not is_pronoun(m)]
    candidates = non_pronoun or normed
    candidates.sort(key=len, reverse=True)
    main = candidates[0].strip()
    if not main:
        return None
    # If the best candidate is still a pronoun, treat as no-op for substitution
    if is_pronoun(main):
        return None
    return main


def _build_pronoun_replacement_map(coref_entities: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build a mapping from pronoun string -> main mention string.

    If the same pronoun appears in multiple clusters, the first cluster wins.
    This is intentionally simple and heuristic.
    """
    mapping: Dict[str, str] = {}

    for cluster in coref_entities:
        main = _choose_main_mention(cluster)
        if not main:
            continue
        mentions: List[Any] = cluster.get("mentions") or []
        for m in mentions:
            if isinstance(m, str):
                text = m
            elif isinstance(m, dict):
                text = m.get("text", "")
            else:
                continue
            token = text.strip()
            if not token:
                continue
            low = token.lower()
            # Only consider pure pronoun tokens
            if low in PRONOUNS and low not in mapping:
                mapping[low] = main

    return mapping


def _replace_pronouns_in_text(text: str, replacement_map: Dict[str, str]) -> str:
    """Replace pronoun tokens in text using a simple regex-based approach.

    This operates at the surface string level, without token offsets, and is
    intentionally conservative:
      - word-boundary matches
      - case-insensitive
    """
    new_text = text

    # Sort pronouns by length desc to avoid partial overlaps (e.g., "her" in "hers")
    for pronoun in sorted(replacement_map.keys(), key=len, reverse=True):
        main = replacement_map[pronoun]
        # Compile pattern with word boundaries, case-insensitive
        pattern = re.compile(rf"\b{re.escape(pronoun)}\b", flags=re.IGNORECASE)
        new_text = pattern.sub(main, new_text)

    return new_text


def unify_mentions(processed_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Create pronoun-substituted sentence variants based on coreference clusters.

    Requirements:
      - Accepts a processed_doc where coreference clusters already exist, under
        processed_doc["coref_entities"] (list of clusters).
      - For each cluster, replace pronouns in sentences with the cluster's main mention.
      - Add processed_doc["unified_sentences"], a flat list of sentence dicts with
        `.text` updated to the pronoun-substituted form.
      - Do NOT lose original sentences; processed_doc["sentences"] remains unchanged.
      - Output remains fully JSON-serializable.
    """
    sentences: List[Dict[str, Any]] = processed_doc.get("sentences") or []
    coref_entities: List[Dict[str, Any]] = processed_doc.get("coref_entities") or []

    # Build pronoun -> main mention map from coref clusters
    pronoun_map = _build_pronoun_replacement_map(coref_entities)
    if not sentences or not pronoun_map:
        # Nothing to do; still create unified_sentences as a shallow copy of originals
        processed_doc["unified_sentences"] = [dict(s) for s in sentences]
        return processed_doc

    unified_sentences: List[Dict[str, Any]] = []

    for idx, sent in enumerate(sentences):
        original_text = sent.get("text", "")
        unified_text = _replace_pronouns_in_text(original_text, pronoun_map)
        new_sent = dict(sent)
        new_sent["text"] = unified_text
        # Optionally annotate origin index for downstream consumers
        new_sent.setdefault("sentence_id", idx)
        unified_sentences.append(new_sent)

    processed_doc["unified_sentences"] = unified_sentences
    return processed_doc