import json
import argparse
from typing import Dict, Any, List

import spacy
from spacy.lang.en import English

def normalize_entity(text):
    """
    Normalize entity text into a machine-friendly identifier.
    - Lowercase
    - Replace non-alphanumeric with underscores
    - Collapse repeated underscores
    - Strip trailing underscores
    """
    import re
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = re.sub(r'_+', '_', text)
    text = text.strip('_')
    return text

def setup_nlp_pipeline():
    """
    Load spaCy transformer NER model and add EntityRuler with forensic patterns.
    """
    nlp = spacy.load("en_core_web_trf")

    # Patterns for EntityRuler
    patterns = [
        {"label": "ROLE", "pattern": "Reporting Officer"},
        {"label": "ROLE", "pattern": "Reporting Investigator"},
        {"label": "ORG", "pattern": "Crime Scene Unit"},
        {"label": "ROLE", "pattern": "Coroner's Inspector"},
        {"label": "ROLE", "pattern": "Deputy"},
        {"label": "ORG", "pattern": "Animal Control"},
        {"label": "ORG", "pattern": "Dispatch"},
        {"label": "ORG", "pattern": [{"LOWER": "unit"}, {"SHAPE": "d+"}]},  # Unit #304
        {"label": "EVIDENCE", "pattern": "body"},
        {"label": "EVIDENCE", "pattern": [{"LOWER": "deceased"}, {"LOWER": "body"}]},
        {"label": "EVIDENCE", "pattern": [{"LOWER": "human"}, {"LOWER": "remains"}]},
        {"label": "EVIDENCE", "pattern": "dog"},
        {"label": "EVIDENCE", "pattern": "handbag"},
        {"label": "EVIDENCE", "pattern": "briefcase"},
        {"label": "LOC", "pattern": "residence"},
        {"label": "PERSON", "pattern": "victim"}
    ]

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(patterns)

    return nlp

def extract_entities_for_segment(nlp, segment):
    """
    Extract entities for each sentence in the segment.

    The function populates a `entities` list on each sentence, with entries:

        {
            "text": str,
            "label": str,
            "norm": str,
            "start": int,  # character offset within the sentence text
            "end": int     # character offset within the sentence text
        }
    """
    for sentence in segment["sentences"]:
        doc = nlp(sentence["text"])
        entities: List[Dict[str, Any]] = []
        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "norm": normalize_entity(ent.text),
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )
        sentence["entities"] = entities

def run_ner_on_document(processed_doc: Dict[str, Any], nlp=None) -> Dict[str, Any]:
    """
    Run NER on a single processed document.

    Expected input shape (for any doc_type: narrative/interview/biography):

        {
          "raw_text": "...",          # optional, not required by this function
          "segments": [
             {
               "time": ...,
               "text": "...",
               "sentences": [
                 {"text": "...", ...},
                 ...
               ],
               "events": [...]
             },
             ...
          ],
          "sentences": [...],         # optional flattened sentences
          "meta": {...},              # optional metadata
          "doc_type": "narrative" | "interview" | "biography"
        }

    This function:
      - runs NER per sentence (per segment),
      - populates `sentence["entities"]`,
      - and adds a top-level `processed_doc["entities"]` list with flattened entities.
    """
    if nlp is None:
        nlp = setup_nlp_pipeline()

    segments = processed_doc.get("segments") or []
    all_entities: List[Dict[str, Any]] = []

    for seg_idx, seg in enumerate(segments):
        extract_entities_for_segment(nlp, seg)
        for sent_idx, sentence in enumerate(seg.get("sentences", [])):
            for ent in sentence.get("entities", []):
                # Copy and annotate with location within the document
                ent_record = {
                    "text": ent.get("text", ""),
                    "label": ent.get("label", ""),
                    "norm": ent.get("norm"),
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                    "segment_index": seg_idx,
                    "sentence_index": sent_idx,
                }
                all_entities.append(ent_record)

    processed_doc["entities"] = all_entities
    return processed_doc


def run_ner_on_narrative(nlp, narrative_json):
    """
    Backwards-compatible wrapper for legacy narrative JSON.

    Operates in-place on the given JSON (which is structurally similar to
    a processed_doc), enriches sentences with `entities`, and returns it.
    """
    processed_doc: Dict[str, Any] = narrative_json
    processed_doc = run_ner_on_document(processed_doc, nlp)
    return processed_doc

def process_json_file(input_path, nlp):
    """
    Load JSON, run NER, save enriched JSON.

    This function is compatible with both:
      - legacy narrative JSON (only `segments` key), and
      - new `processed_doc` JSON (with raw_text/meta/doc_type fields).
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data: Dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {input_path}: {e}")

    segments = data.get("segments") or []
    print(f"Processing {input_path} with {len(segments)} segments.")
    total_sentences = sum(len(seg.get('sentences', [])) for seg in segments)
    print(f"Enriching {total_sentences} sentences with NER.")

    enriched_doc = run_ner_on_document(data, nlp)

    output_path = input_path.replace('.json', '_ner.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_doc, f, indent=4, ensure_ascii=False)

    print(f"Output saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Run NER on preprocessed narrative JSON.')
    parser.add_argument('--input', required=True, help='Path to the input JSON file')
    args = parser.parse_args()

    nlp = setup_nlp_pipeline()
    process_json_file(args.input, nlp)

if __name__ == '__main__':
    main()