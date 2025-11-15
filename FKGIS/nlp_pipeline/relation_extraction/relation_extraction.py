import spacy
from spacy.util import load_model_from_config
import json
import spacy_llm
from pathlib import Path
import os

def create_relation_pipeline():
    """
    Create the spaCy pipeline for relation extraction using your config_relations.cfg file.
    Loads NER, EntityRuler, and LLM-based relation extraction.
    """
    import os
    import spacy
    from spacy.util import load_model_from_config

    # Path to config_relations.cfg relative to this file
    config_path = os.path.join(os.path.dirname(__file__), "config_relations.cfg")

    # Load config object
    config = spacy.util.load_config(config_path)

    # Load full pipeline from config
    nlp = load_model_from_config(config)

        # EntityRuler patterns (add if not in config)
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

    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(patterns)

    return nlp



nlp = create_relation_pipeline()
# -----------------------------
# Utility: Format entities for prompt
# -----------------------------
def format_entities(global_entities):
    """
    Convert your global_entities list into a clean block such as:
    cheryl_weston (PERSON)
    officer_willits (ROLE)
    the_scene (LOC)
    """
    lines = []
    for ent in global_entities:
        lines.append(f"- {ent['canonical_name']} ({ent['entity_type']})")
    return "\n".join(lines)


def extract_relations(case_json):
    """
    Inputs your JSON like:
    {
      "case_id": "...",
      "global_entities": [...],
      "documents": {"narrative": {"sentences": [...]}}
    }

    Returns:
      list of dict:
      [
        {"head": "...", "relation": "...", "tail": "..."}
      ]
    """
    global_entities = case_json["global_entities"]

    # Join all narrative sentences into one text block
    narrative_sentences = case_json["documents"]["narrative"]["sentences"]
    full_text = " ".join([s["text"] for s in narrative_sentences])

    # Create formatted entity list
    entity_block = format_entities(global_entities)

    doc = nlp.make_doc(full_text)
    doc.user_data["entities"] = entity_block
    doc.user_data["text"] = full_text


    doc = nlp(doc)

    llm_output = doc._.llm

    if "relations" not in llm_output:
        raise ValueError("LLM output missing 'relations' field.")

    return llm_output["relations"]


if __name__ == "__main__":

    sample_input = {
      "case_id": "CASE_1029",
      "global_entities": [
        {
          "global_id": 1,
          "canonical_name": "cheryl_weston",
          "entity_type": "PERSON",
          "mentions": ["Cheryl Weston", "the caller", "she"],
          "source_docs": ["narrative", "dispatch_log"]
        },
        {
          "global_id": 2,
          "canonical_name": "officer_willits",
          "entity_type": "ROLE",
          "mentions": ["R/O Willits", "the officer", "he"],
          "source_docs": ["narrative"]
        },
        {
          "global_id": 3,
          "canonical_name": "the_scene",
          "entity_type": "LOC",
          "mentions": ["the scene", "residence", "123 Maple Street"],
          "source_docs": ["narrative", "dispatch_log"]
        }
      ],
      "documents": {
        "narrative": {
          "sentences": [
            {
              "text": "At 11:06 a.m., Dispatch received a 911 call from Cheryl Weston, who reported a disturbance at 123 Maple Street.",
              "timestamp": "11:06 a.m."
            },
            {
              "text": "Officer Willits was dispatched to the scene.",
              "timestamp": None
            }
          ]
        }
      }
    }

    print("\nRunning Relation Extraction...\n")
    relations = extract_relations(sample_input)
    print(json.dumps(relations, indent=2))
