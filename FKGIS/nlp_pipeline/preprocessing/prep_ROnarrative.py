import re
import json
import os
import argparse
import spacy
from spacy.lang.en import English

# Abbreviation dictionary
ABBREVIATIONS = {
    r'\bR\s*/\s*O\b': 'Reporting Officer',
    r'\bR\s*/\s*Is\b': 'Reporting Investigators',
    r'\bR\s*/\s*I\b': 'Reporting Investigator',
    r'\bCSU\b': 'Crime Scene Unit',
    r'\bETA\b': 'Estimated Time of Arrival'
}

# Regex patterns
TIME_PATTERN = re.compile(r'\b\d{1,2}[:.]\d{2}\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)\b', re.IGNORECASE)
ACTOR_PATTERN = re.compile(r'\b(?:Reporting Officer|Reporting Investigator|Officer|Detective|Inspector|Deputy)s?\b', re.IGNORECASE)
ENV_PATTERN = re.compile(r'\b(Weather|Temperature|Humidity)\b', re.IGNORECASE)
SEGMENT_PATTERN = re.compile(r'(At \d{1,2}:\d{2} (?:a\.m\.|p\.m\.)|Reporting Officer \w+|Reporting Investigator \w+|Inspector \w+|Deputy \w+)', re.IGNORECASE)
EVENT_VERBS = re.compile(r'\b(reported|arrived|observed|pronounced|called|responded|notified|dispatched|assigned|proceeded)\b', re.IGNORECASE)

def normalize_text(text):
    # Convert to UTF-8 (assuming input is string)
    text = text.encode('utf-8').decode('utf-8')
    # Clean line breaks and extra spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove header-like lines
    text = re.sub(r'^REPORTING OFFICER’S NARRATIVE.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Correct common OCR artifacts
    text = text.replace('EPORTING', 'REPORTING')
    # Normalize degree symbols
    text = text.replace('º', '°')
    # Unify slash spacing
    text = re.sub(r'\s*/\s*', ' / ', text)
    return text

def expand_abbreviations(text):
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(abbr, full, text)
    return text

def segment_text(text):
    segments = []
    parts = SEGMENT_PATTERN.split(text)
    current_segment = ""
    current_time = None
    for part in parts:
        if SEGMENT_PATTERN.match(part.strip()):
            if current_segment:
                segments.append({'time': current_time, 'text': current_segment.strip()})
            current_time = TIME_PATTERN.search(part)
            current_time = current_time.group(0) if current_time else None
            current_segment = part
        else:
            current_segment += part
    if current_segment:
        segments.append({'time': current_time, 'text': current_segment.strip()})
    return segments

def merge_incomplete_sentences(sentences):
    merged = []
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        if len(sent['text'].split()) < 3 and not any(sent['text'].endswith(p) for p in ('.', '!', '?')) and i + 1 < len(sentences):
            # Merge with next
            next_sent = sentences[i+1]
            merged_sent = {
                'text': sent['text'] + ' ' + next_sent['text'],
                'has_time': sent['has_time'] or next_sent['has_time'],
                'has_actor': sent['has_actor'] or next_sent['has_actor'],
                'has_env': sent['has_env'] or next_sent['has_env']
            }
            merged.append(merged_sent)
            i += 2
        else:
            merged.append(sent)
            i += 1
    return merged

def process_sentences(segment_text, nlp):
    doc = nlp(segment_text)
    sentences = []
    for sent in doc.sents:
        text = sent.text.strip()
        has_time = bool(TIME_PATTERN.search(text))
        has_actor = bool(ACTOR_PATTERN.search(text))
        has_env = bool(ENV_PATTERN.search(text))
        sentences.append({
            'text': text,
            'has_time': has_time,
            'has_actor': has_actor,
            'has_env': has_env
        })
    sentences = merge_incomplete_sentences(sentences)
    return sentences

def extract_events(sentences):
    events = []
    for sent in sentences:
        if sent['has_time'] or EVENT_VERBS.search(sent['text']):
            timestamp = TIME_PATTERN.search(sent['text'])
            timestamp = timestamp.group(0) if timestamp else None
            category = 'unknown'
            if 'arrived' in sent['text'].lower():
                category = 'arrival'
            elif 'reported' in sent['text'].lower():
                category = 'report'
            elif 'observed' in sent['text'].lower():
                category = 'observation'
            elif 'pronounced' in sent['text'].lower():
                category = 'pronouncement'
            events.append({
                'text': sent['text'],
                'timestamp': timestamp,
                'category': category
            })
    return events

def process_file(input_path):
    """Preprocess a narrative text file into a uniform processed_doc JSON.

    Output shape:

        {
          "raw_text": "...",          # normalized narrative text
          "segments": [...],          # list of segment dicts with sentences/events
          "sentences": [...],         # flat list of sentence dicts
          "meta": {
              "source_path": "...",   # original file path
          },
          "doc_type": "narrative"
        }
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Step 1: Normalization
    text = normalize_text(text)

    # Step 2: Abbreviation Expansion
    text = expand_abbreviations(text)

    # Step 3: Section and Time Segmentation
    segments = segment_text(text)

    # Load spaCy
    nlp = English()
    nlp.add_pipe('sentencizer')

    processed_segments = []
    all_sentences = []

    for seg in segments:
        # Step 4: Sentence Segmentation
        sentences = process_sentences(seg['text'], nlp)

        # Step 5: Event Candidate Extraction
        events = extract_events(sentences)

        segment_obj = {
            'time': seg['time'],
            'text': seg['text'],
            'sentences': sentences,
            'events': events,
        }
        processed_segments.append(segment_obj)
        all_sentences.extend(sentences)

    processed_doc = {
        'raw_text': text,
        'segments': processed_segments,
        'sentences': all_sentences,
        'meta': {
            'source_path': input_path,
        },
        'doc_type': 'narrative',
    }

    # Output JSON
    output_path = input_path.replace('.txt', '_ROnarrative_processed.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_doc, f, indent=4, ensure_ascii=False)

    return output_path

def main():
    parser = argparse.ArgumentParser(description='Preprocess Reporting Officer’s Narrative text files.')
    parser.add_argument('input_file', help='Path to the input .txt file')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: File {args.input_file} does not exist.")
        return

    output_path = process_file(args.input_file)
    print(f"Processed file saved to: {output_path}")

if __name__ == '__main__':
    main()