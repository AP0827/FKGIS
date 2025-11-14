import json
import argparse
import spacy
from typing import Dict, List, Any
import re

# Simple normalization and pronoun lists
PRONOUNS_MALE = {"he", "him", "his"}
PRONOUNS_FEMALE = {"she", "her", "hers"}
PRONOUNS_NEUTRAL = {"they", "them", "their", "theirs"}

def normalize_text(text: str) -> str:
    """Create a deterministic normalization key for mentions."""
    t = text.strip().lower()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.replace(" ", "_")

# Updated SQL Schema as strings
CREATE_GLOBAL_ENTITIES = """
CREATE TABLE global_entities (
    global_id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    canonical_name TEXT NOT NULL,
    entity_type VARCHAR(50),
    gender VARCHAR(10),
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
                "gender": None,
                "source_docs": [doc_name]
            }
            self.global_counter += 1

    def infer_gender_from_mention(self, mention_text: str) -> str:
        """Simple heuristic to infer gender from mention text (titles/pronouns)."""
        t = mention_text.lower()
        if any(p in t for p in ["mr ", "mr.", "sir", "him", "his"]):
            return "male"
        if any(p in t for p in ["ms ", "mrs", "ms.", "miss", "ma'am", "her", "she"]):
            return "female"
        return "unknown"

    def resolve_pronouns_in_doc(self, doc, doc_name: str):
        """Resolve simple pronouns by linking them to the nearest preceding entity.

        This is a lightweight heuristic fallback when a full coref component is
        not available in the pipeline.
        """
        # Build list of candidate entities (token index -> normalized key)
        candidates = []  # list of tuples (end_token_idx, norm)
        for ent in doc.ents:
            norm = normalize_text(ent.text)
            candidates.append((ent.end, norm, ent.label_))
            # Ensure the pool includes this entity
            self.add_mention(doc_name, ent.text, ent.label_, norm)

        # Iterate tokens, look for pronouns
        for i, token in enumerate(doc):
            if token.pos_ == "PRON":
                low = token.lower_
                if low in PRONOUNS_MALE | PRONOUNS_FEMALE | PRONOUNS_NEUTRAL:
                    # find nearest candidate with end <= token.i
                    chosen = None
                    for end_idx, norm, label in reversed(candidates):
                        if end_idx - 1 <= token.i - 1:
                            chosen = (norm, label)
                            break
                    if chosen:
                        norm, label = chosen
                        self.add_mention(doc_name, token.text, label or "PRON", norm)
                        # try to set gender if unknown
                        g = None
                        if low in PRONOUNS_MALE:
                            g = "male"
                        elif low in PRONOUNS_FEMALE:
                            g = "female"
                        elif low in PRONOUNS_NEUTRAL:
                            g = "unknown"
                        if g and self.pool.get(norm) and (not self.pool[norm]["gender"] or self.pool[norm]["gender"] == "unknown"):
                            self.pool[norm]["gender"] = g

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
        if entity_b["gender"]:
            entity_a["gender"] = entity_b["gender"]
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

        # Constraint B: Gender consistency (simplified)
        # Assume no bio for now

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

def process_ner_json(case_id: str, ner_json_path: str, doc_name: str = "narrative") -> List[Dict[str, Any]]:
    """
    Process NER JSON to build global entity pool with coreference.
    """
    nlp = create_coref_pipeline()
    pool = GlobalEntityPool(case_id)
    
    with open(ner_json_path, 'r') as f:
        data = json.load(f)
    
    # Extract full text from segments
    full_text = " ".join(seg['text'] for seg in data['segments'])
    doc = nlp(full_text)
    # Integrate NER entities first so basic mentions exist in the pool
    for seg in data['segments']:
        for sent in seg['sentences']:
            if 'entities' in sent:
                for ent in sent['entities']:
                    norm = ent.get('norm') or normalize_text(ent.get('text', ''))
                    label = ent.get('label', 'UNKNOWN')
                    mention = ent.get('text', '')
                    pool.add_mention(doc_name, mention, label, norm)

    # Integrate any coref clusters from the doc (if pipeline provides them)
    pool.integrate_coref_clusters(doc, doc_name)

    # Attempt lightweight pronoun resolution to link pronouns to nearby entities
    pool.resolve_pronouns_in_doc(doc, doc_name)

    pool.apply_constraints()
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