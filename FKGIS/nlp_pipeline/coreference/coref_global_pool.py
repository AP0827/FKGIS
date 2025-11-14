import json
import argparse
import spacy
from typing import Dict, List, Any
import re

# Simple normalization and pronoun lists (no gender semantics in the pool)
PRONOUNS = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}

def normalize_text(text: str) -> str:
    """Create a deterministic normalization key for mentions."""
    t = text.strip().lower()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.replace(" ", "_")

# Updated SQL Schema as strings (no gender column; one unified KG per case)
CREATE_GLOBAL_ENTITIES = """
-- Global entity pool and unified case knowledge graph
-- For each case_id, the combination of:
--   - entities (local mentions)
--   - global_entities (canonical nodes)
--   - relations (edges)
--   - events (timeline)
-- together form ONE unified knowledge graph for that case.

CREATE TABLE global_entities (
    global_id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    canonical_name TEXT NOT NULL,
    entity_type VARCHAR(50),
    source_docs TEXT[]
);
"""

ALTER_ENTITIES = """
ALTER TABLE entities
ADD COLUMN global_id INT REFERENCES global_entities(global_id) ON DELETE SET NULL;
"""

ALTER_RELATIONS = """
ALTER TABLE relations
ADD COLUMN subject_global_id INT REFERENCES global_entities(global_id),
ADD COLUMN object_global_id INT REFERENCES global_entities(global_id);
"""

ALTER_EVENTS = """
ALTER TABLE events
ADD COLUMN actor_global_id INT REFERENCES global_entities(global_id);
"""

def create_coref_pipeline():
    """
    Create spaCy pipeline with NER and EntityRuler for basic processing.

    Uses the transformer model `en_core_web_trf`, which you have installed.
    If needed, this can later be swapped for a dedicated coref model, but the
    rest of this module already includes a heuristic pronoun resolver and can
    operate without `doc._.coref_clusters`.
    """
    nlp = spacy.load("en_core_web_trf")

    # EntityRuler patterns
    patterns = [
        {"label": "ROLE", "pattern": "Reporting Officer"},
        {"label": "ROLE", "pattern": "Reporting Investigator"},
        {"label": "ORG", "pattern": "Crime Scene Unit"},
        {"label": "ROLE", "pattern": "Coroner's Inspector"},
        {"label": "ROLE", "pattern": "Deputy"},
        {"label": "PERSON", "pattern": "Witness"},
        {"label": "PERSON", "pattern": "Victim"},
        {"label": "EVIDENCE", "pattern": "Body"},
        {"label": "EVIDENCE", "pattern": "Dog"},
        {"label": "LOC", "pattern": "Residence"},
        {"label": "LOC", "pattern": "Scene"}
    ]

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(patterns)

    return nlp

class GlobalEntityPool:
    def __init__(self, case_id: str):
        self.case_id = case_id
        self.pool: Dict[str, Dict[str, Any]] = {}
        self.global_counter = 1

    def add_mention(self, doc_name: str, mention_text: str, label: str, norm: str):
        """
        Add a mention to the pool.
        """
        if norm in self.pool:
            entity = self.pool[norm]
            if mention_text not in entity["mentions"]:
                entity["mentions"].append(mention_text)
            if doc_name not in entity["source_docs"]:
                entity["source_docs"].append(doc_name)
        else:
            self.pool[norm] = {
                "global_id": self.global_counter,
                "canonical_name": norm,
                "mentions": [mention_text],
                "entity_type": label,
                "source_docs": [doc_name],
            }
            self.global_counter += 1


    def resolve_pronouns_in_doc(self, doc, doc_name: str):
        """Resolve simple pronouns by linking them to the nearest preceding entity.

        This is a lightweight heuristic fallback when a full coref component is
        not available in the pipeline. It does NOT attempt to infer or store
        gender information.
        """
        # Build list of candidate entities (token index -> normalized key)
        candidates = []  # list of tuples (end_token_idx, norm, label)
        for ent in doc.ents:
            norm = normalize_text(ent.text)
            candidates.append((ent.end, norm, ent.label_))
            # Ensure the pool includes this entity
            self.add_mention(doc_name, ent.text, ent.label_, norm)

        # Iterate tokens, look for pronouns
        for token in doc:
            if token.pos_ == "PRON":
                low = token.lower_
                if low in PRONOUNS:
                    # find nearest candidate with end <= token.i
                    chosen = None
                    for end_idx, norm, label in reversed(candidates):
                        if end_idx - 1 <= token.i - 1:
                            chosen = (norm, label)
                            break
                    if chosen:
                        norm, label = chosen
                        self.add_mention(doc_name, token.text, label or "PRON", norm)

    def merge_entities(self, norm_a: str, norm_b: str):
        """
        Merge two entity clusters.
        """
        if norm_a not in self.pool or norm_b not in self.pool:
            return
        entity_a = self.pool[norm_a]
        entity_b = self.pool[norm_b]
        # Merge into a, keep global_id of a
        entity_a["mentions"].extend(entity_b["mentions"])
        entity_a["source_docs"].extend(entity_b["source_docs"])
        entity_a["source_docs"] = list(set(entity_a["source_docs"]))
        del self.pool[norm_b]

    def apply_constraints(self):
        """
        Apply forensic constraints to merge/split clusters.
        """
        # Constraint A: Only one victim
        victim_norms = [k for k, v in self.pool.items() if "victim" in k.lower() or "deceased" in k.lower() or "body" in k.lower()]
        if len(victim_norms) > 1:
            for norm in victim_norms[1:]:
                self.merge_entities(victim_norms[0], norm)

        # Constraint B: (previously gender consistency) intentionally disabled; no gender logic in pool

        # Constraint C: Officer role consistency
        officer_norms = [k for k, v in self.pool.items() if "officer" in k.lower() or "inspector" in k.lower() or "deputy" in k.lower()]
        # Merge if similar names, but simplified

        # Constraint D: Unique objects
        unique_objects = ["dog", "body", "scene", "residence"]
        for obj in unique_objects:
            obj_norms = [k for k in self.pool if obj in k.lower()]
            if len(obj_norms) > 1:
                for norm in obj_norms[1:]:
                    self.merge_entities(obj_norms[0], norm)

        # Constraint E: Role label fixes (simplified)

    def integrate_coref_clusters(self, doc, doc_name: str):
        """
        Integrate coreference clusters from a document.
        """
        if not hasattr(doc._, 'coref_clusters'):
            return
        for cluster in doc._.coref_clusters:
            canonical = cluster.main.text if cluster.main else cluster.mentions[0].text
            norm = canonical.lower().replace(' ', '_')
            label = "UNKNOWN"  # Simplified, could lookup from ents
            for mention in cluster.mentions:
                self.add_mention(doc_name, mention.text, label, norm)

    def export(self) -> List[Dict[str, Any]]:
        """
        Export the pool as a list of dicts.
        """
        return list(self.pool.values())


def _build_spacy_doc_from_processed(nlp, processed_doc: Dict[str, Any]):
    """Helper to obtain a spaCy Doc from a processed document structure."""
    raw_text = processed_doc.get("raw_text")
    if not raw_text:
        # Fallback: concatenate segment texts if raw_text is not present
        segments = processed_doc.get("segments") or []
        raw_text = " ".join(seg.get("text", "") for seg in segments)
    return nlp(raw_text)


def run_coref_on_document(nlp, processed_doc: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
    """Run coreference for a single processed document.

    This function is document-type agnostic (narrative/interview/biography).
    It expects a structure of the form:

        {
            "raw_text": "...",
            "segments": [...],
            "sentences": [...],
            "meta": {...},
            "doc_type": "narrative" | "interview" | "biography",
            "entities": [...]  # optional, added by NER
        }

    It augments the document with a "coref_entities" key describing
    canonical entities and their mentions in this document only.
    """
    doc_type = processed_doc.get("doc_type") or doc_type
    meta = processed_doc.get("meta") or {}
    case_id = meta.get("case_id", "UNKNOWN_CASE")
    doc_name = meta.get("doc_id") or meta.get("source") or doc_type

    local_pool = GlobalEntityPool(case_id)

    # Integrate explicit NER entities first (if present)
    for ent in processed_doc.get("entities", []):
        text = ent.get("text", "")
        if not text:
            continue
        label = ent.get("label", "UNKNOWN")
        norm = ent.get("norm") or normalize_text(text)
        local_pool.add_mention(doc_name, text, label, norm)

    # Build spaCy Doc and integrate any model-provided entities + pronouns
    doc = _build_spacy_doc_from_processed(nlp, processed_doc)
    local_pool.integrate_coref_clusters(doc, doc_name)
    local_pool.resolve_pronouns_in_doc(doc, doc_name)

    # Apply domain constraints (victim uniqueness, unique objects, etc.)
    local_pool.apply_constraints()

    processed_doc["coref_entities"] = local_pool.export()
    return processed_doc


def integrate_processed_doc(
    pool: GlobalEntityPool,
    processed_doc: Dict[str, Any],
    doc_name: str,
    doc_type: str,
) -> None:
    """Merge a single processed document's coref info into the global pool.

    The preferred source is processed_doc["coref_entities"], if present.
    As a fallback, it will use processed_doc["entities"] only.
    """
    # Prefer document-level coreference entities if available
    coref_entities = processed_doc.get("coref_entities")
    if coref_entities:
        for cluster in coref_entities:
            canonical_name = cluster.get("canonical_name")
            label = cluster.get("entity_type", "UNKNOWN")
            mentions = cluster.get("mentions") or []
            for m in mentions:
                # m may be a raw string or a dict with "text"
                if isinstance(m, str):
                    mention_text = m
                else:
                    mention_text = m.get("text", "")
                if mention_text:
                    pool.add_mention(doc_name, mention_text, label, canonical_name)
        return

    # Fallback: only NER entities are available
    for ent in processed_doc.get("entities", []):
        text = ent.get("text", "")
        if not text:
            continue
        label = ent.get("label", "UNKNOWN")
        norm = ent.get("norm") or normalize_text(text)
        pool.add_mention(doc_name, text, label, norm)


def _ner_json_to_processed_doc(data: Dict[str, Any], doc_type: str = "narrative") -> Dict[str, Any]:
    """Legacy helper: convert existing narrative NER JSON into processed_doc shape.

    This allows the coreference pipeline to operate on older narrative JSON
    of the form:

        {"segments": [ { "text": ..., "sentences": [ { "entities": [...] }, ... ] }, ... ]}
    """
    segments = data.get("segments") or []
    raw_text = " ".join(seg.get("text", "") for seg in segments)

    # Flatten sentences while keeping original structure in segments
    sentences = []
    all_entities = []
    for seg in segments:
        for sent in seg.get("sentences", []):
            sentences.append(sent)
            for ent in sent.get("entities", []):
                text = ent.get("text", "")
                if not text:
                    continue
                label = ent.get("label", "UNKNOWN")
                norm = ent.get("norm") or normalize_text(text)
                all_entities.append(
                    {
                        "text": text,
                        "label": label,
                        "norm": norm,
                    }
                )

    processed_doc: Dict[str, Any] = {
        "raw_text": raw_text,
        "segments": segments,
        "sentences": sentences,
        "meta": {},
        "doc_type": doc_type,
    }
    if all_entities:
        processed_doc["entities"] = all_entities

    return processed_doc


def process_ner_json(case_id: str, ner_json_path: str, doc_name: str = "narrative") -> List[Dict[str, Any]]:
    """
    Legacy entry point: process a single NER JSON (narrative-style) to build
    a global entity pool with coreference.

    This function is kept for backward compatibility with existing pipelines.
    New code should prefer working with processed_doc dictionaries and the
    run_coref_on_document / integrate_processed_doc APIs.
    """
    nlp = create_coref_pipeline()
    pool = GlobalEntityPool(case_id)

    with open(ner_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_doc = _ner_json_to_processed_doc(data, doc_type="narrative")
    # Run document-level coref to populate processed_doc["coref_entities"]
    processed_doc = run_coref_on_document(nlp, processed_doc, doc_type="narrative")

    # Merge this document into the case-level global pool
    integrate_processed_doc(pool, processed_doc, doc_name=doc_name, doc_type="narrative")

    return pool.export()

def main():
    parser = argparse.ArgumentParser(description='Build global entity pool with coreference from NER JSON.')
    parser.add_argument('--case', required=True, help='Case ID')
    parser.add_argument('--ner_json', required=True, help='Path to NER-processed JSON file')

    args = parser.parse_args()

    entities = process_ner_json(args.case, args.ner_json)

    output_file = f"{args.case}_global_entities.json"
    with open(output_file, 'w') as f:
        json.dump(entities, f, indent=4)

    print(f"Global entities saved to {output_file}")

if __name__ == '__main__':
    main()