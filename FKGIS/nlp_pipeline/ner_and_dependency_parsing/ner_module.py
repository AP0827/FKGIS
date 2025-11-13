import json
import argparse
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
    """
    for sentence in segment["sentences"]:
        doc = nlp(sentence["text"])
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "norm": normalize_entity(ent.text)
            })
        sentence["entities"] = entities

def run_ner_on_narrative(nlp, narrative_json):
    """
    Run NER on all segments in the narrative JSON.
    """
    for seg in narrative_json["segments"]:
        extract_entities_for_segment(nlp, seg)
    return narrative_json

def process_json_file(input_path, nlp):
    """
    Load JSON, run NER, save enriched JSON.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {input_path}: {e}")

    print(f"Processing {input_path} with {len(data['segments'])} segments.")
    total_sentences = sum(len(seg['sentences']) for seg in data['segments'])
    print(f"Enriching {total_sentences} sentences with NER.")

    enriched_data = run_ner_on_narrative(nlp, data)

    output_path = input_path.replace('.json', '_ner.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=4, ensure_ascii=False)

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