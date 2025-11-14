from __future__ import annotations

from typing import Dict, Any, List, Optional, DefaultDict
from collections import defaultdict

import spacy
from spacy.tokens import Doc, Token

# Lazy-loaded spaCy pipeline with dependency parser enabled
_NLP: Optional["spacy.language.Language"] = None

# Heuristic thresholds and preferences
MAX_ARG_TOKENS = 15  # drop relations with very long subject/object

# Labels we consider most informative for objects
PREFERRED_OBJECT_ENTITY_LABELS = {"PERSON", "ORG", "EVIDENCE"}


def _get_nlp() -> "spacy.language.Language":
    """Return a spaCy pipeline with a dependency parser.

    Uses the same transformer model name as the NER/coref stages for consistency.
    """
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_trf")
    return _NLP


def _iter_sentences_for_relations(processed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of sentences to use for relation extraction.

    Preference order:
      1) processed_doc["unified_sentences"] (after pronoun substitution)
      2) processed_doc["sentences"]
    """
    unified = processed_doc.get("unified_sentences") or []
    if unified:
        return unified
    return processed_doc.get("sentences") or []


def _build_sentence_entity_index(
    processed_doc: Dict[str, Any]
) -> DefaultDict[int, List[Dict[str, Any]]]:
    """Index entities by global sentence_id.

    We reconstruct the mapping from (segment_index, sentence_index) to the
    global sentence_id by iterating segments in order, to mirror the flattening
    used when processed_doc["sentences"] was built.
    """
    segments = processed_doc.get("segments") or []
    entities = processed_doc.get("entities") or []

    # Build mapping (segment_idx, sentence_idx) -> global sentence_id
    seg_sent_to_global: Dict[tuple[int, int], int] = {}
    global_id = 0
    for seg_idx, seg in enumerate(segments):
        for sent_idx, _sent in enumerate(seg.get("sentences", [])):
            seg_sent_to_global[(seg_idx, sent_idx)] = global_id
            global_id += 1

    by_sentence: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ent in entities:
        seg_idx = ent.get("segment_index")
        sent_idx = ent.get("sentence_index")
        if seg_idx is None or sent_idx is None:
            continue
        key = (seg_idx, sent_idx)
        if key not in seg_sent_to_global:
            continue
        sid = seg_sent_to_global[key]
        by_sentence[sid].append(ent)

    return by_sentence


def _overlaps_entity_text(
    text: str,
    entities: List[Dict[str, Any]],
    preferred_only: bool = False,
) -> bool:
    """Check whether text contains (substring match) any entity mention.

    If preferred_only is True, only entities with labels in
    PREFERRED_OBJECT_ENTITY_LABELS are considered.
    """
    if not text or not entities:
        return False

    low = text.lower()
    for ent in entities:
        label = ent.get("label")
        if preferred_only and label not in PREFERRED_OBJECT_ENTITY_LABELS:
            continue
        etext = ent.get("text")
        if not etext:
            continue
        if etext.lower() in low:
            return True
    return False


def _extract_triples_from_doc(doc: Doc, sentence_id: int) -> List[Dict[str, Any]]:
    """Extract simple (subject, predicate, object) triples from a spaCy Doc.

    Rules:
      - Take each VERB/AUX token that is a ROOT.
      - Subject: any child with dep_ in {"nsubj", "nsubjpass"}.
      - Object: any child with dep_ in {"dobj", "pobj", "attr", "dative"}, or
                the pobj of a PREP child if no direct object found.
      - Use subtree text for subject and object, and verb lemma as predicate.
    """
    relations: List[Dict[str, Any]] = []

    for token in doc:
        if token.dep_ != "ROOT" or token.pos_ not in {"VERB", "AUX"}:
            continue

        verb = token
        subject: Optional[Token] = None
        obj: Optional[Token] = None

        # First pass: direct children
        for child in verb.children:
            if child.dep_ in {"nsubj", "nsubjpass"} and subject is None:
                subject = child
            elif child.dep_ in {"dobj", "pobj", "attr", "dative"} and obj is None:
                obj = child

        # Second pass: look into prepositions for pobj if we still have no object
        if obj is None:
            for child in verb.children:
                if child.dep_ == "prep":
                    for gc in child.children:
                        if gc.dep_ == "pobj":
                            obj = gc
                            break
                if obj is not None:
                    break

        if subject is None or obj is None:
            continue

        subject_text = " ".join(t.text for t in subject.subtree).strip()
        object_text = " ".join(t.text for t in obj.subtree).strip()
        predicate_text = (verb.lemma_ or verb.text).strip()

        if not subject_text or not predicate_text or not object_text:
            continue

        relations.append(
            {
                "subject": subject_text,
                "predicate": predicate_text,
                "object": object_text,
                "sentence_id": sentence_id,
            }
        )

    return relations


def _filter_triples_for_sentence(
    triples: List[Dict[str, Any]],
    sentence_id: int,
    entities_by_sentence: Dict[int, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Apply heuristic filters to improve relation quality.

    Filters:
      - Drop triples where subject or object is very long (> MAX_ARG_TOKENS).
      - Drop triples where neither subject nor object overlaps any recognized
        entity in that sentence.
      - For predicate == "be", drop unless the object overlaps a preferred
        entity type (PERSON/ORG/EVIDENCE).
    """
    sent_entities = entities_by_sentence.get(sentence_id, [])
    filtered: List[Dict[str, Any]] = []

    for triple in triples:
        subj = triple.get("subject", "") or ""
        obj = triple.get("object", "") or ""
        pred = triple.get("predicate", "") or ""

        subj_len = len(subj.split())
        obj_len = len(obj.split())

        # 1) Length-based filter
        if subj_len > MAX_ARG_TOKENS or obj_len > MAX_ARG_TOKENS:
            continue

        # 2) Entity-overlap filter
        has_ent_overlap = _overlaps_entity_text(subj, sent_entities) or _overlaps_entity_text(
            obj, sent_entities
        )
        if sent_entities and not has_ent_overlap:
            # If there are entities in the sentence but neither arg touches them, drop.
            continue

        # 3) Drop trivial "be" relations unless object overlaps an entity of
        #    a preferred type (PERSON/ORG/EVIDENCE).
        if pred == "be":
            if not _overlaps_entity_text(obj, sent_entities, preferred_only=True):
                continue

        filtered.append(triple)

    return filtered


def extract_relations(processed_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Populate processed_doc['relations'] with simple SPO triples.

    Behavior:
      - Operates primarily on processed_doc['unified_sentences'] if present,
        otherwise falls back to processed_doc['sentences'].
      - For each sentence:
          * Run spaCy dependency parsing.
          * Extract simple subject-verb-object triples using dependency labels.
          * Filter triples with heuristics that prefer overlap with named entities.
      - Output schema:

            processed_doc["relations"] = [
                {
                   "subject": "...",
                   "predicate": "...",
                   "object": "...",
                   "sentence_id": 3
                },
                ...
            ]

    This function is intentionally minimal and rule-based, with no LLM logic.
    """
    sentences = _iter_sentences_for_relations(processed_doc)
    if not sentences:
        processed_doc["relations"] = []
        return processed_doc

    # Build an index of entities by sentence_id to support filtering
    entities_by_sentence = _build_sentence_entity_index(processed_doc)

    nlp = _get_nlp()
    relations: List[Dict[str, Any]] = []

    for idx, sent in enumerate(sentences):
        text = (sent.get("text") or "").strip()
        if not text:
            continue

        doc = nlp(text)
        sentence_id = sent.get("sentence_id", idx)
        raw_triples = _extract_triples_from_doc(doc, sentence_id=sentence_id)
        good_triples = _filter_triples_for_sentence(
            raw_triples, sentence_id=sentence_id, entities_by_sentence=entities_by_sentence
        )
        relations.extend(good_triples)

    processed_doc["relations"] = relations
    return processed_doc