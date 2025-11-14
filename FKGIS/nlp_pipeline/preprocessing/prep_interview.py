import re
import json
import os
import argparse

import spacy
from spacy.lang.en import English


TIME_INLINE_PATTERN = re.compile(r"\b\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.|am|pm)\b", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Basic normalization: strip, collapse whitespace."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_header(lines):
    """Extract simple header info from the top of an interview.

    Assumes something like:
        Becky Pace interview
        Sunday, October 16, 2022 – 12:03 p.m.
    """
    title = None
    header_time = None
    cleaned_lines = [ln.strip() for ln in lines if ln.strip()]

    if cleaned_lines:
        title = cleaned_lines[0]
    if len(cleaned_lines) > 1:
        header_time = cleaned_lines[1]

    return title, header_time


def segment_by_speaker(body_text: str):
    """Segment interview body by speaker turns.

    Lines that start with 'Name:' are treated as speaker turns.
    Other lines are attached to the current speaker; lines before
    the first 'Speaker:' are attributed to 'NARRATOR'.
    """
    speaker_pattern = re.compile(r"^([^:]+):\s*(.*)$")

    segments = []
    current_speaker = "NARRATOR"
    current_lines = []

    for raw_line in body_text.split("\n"):
        line = raw_line.strip()
        if not line:
            # Preserve paragraph breaks inside a speaker turn
            if current_lines:
                current_lines.append("")
            continue

        m = speaker_pattern.match(line)
        if m:
            # Flush previous segment
            if current_lines:
                segments.append(
                    {
                        "speaker": current_speaker,
                        "text": " ".join(l for l in current_lines if l).strip(),
                    }
                )
                current_lines = []
            current_speaker = m.group(1).strip()
            first_utterance = m.group(2).strip()
            if first_utterance:
                current_lines.append(first_utterance)
        else:
            current_lines.append(line)

    if current_lines:
        segments.append(
            {
                "speaker": current_speaker,
                "text": " ".join(l for l in current_lines if l).strip(),
            }
        )

    return segments


def annotate_sentences(segment_text: str, speaker: str, nlp):
    """Split a segment into sentences and add simple flags."""
    doc = nlp(segment_text)
    sentences = []
    for sent in doc.sents:
        text = sent.text.strip()
        if not text:
            continue
        has_time = bool(TIME_INLINE_PATTERN.search(text))
        # Simple actor heuristic: presence of proper noun or first-person pronoun
        has_actor = any(t.pos_ in {"PROPN", "PRON"} for t in sent)
        has_env = False  # interviews are not strongly environment-focused by default
        sentences.append(
            {
                "text": text,
                "has_time": has_time,
                "has_actor": has_actor,
                "has_env": has_env,
                "speaker": speaker,
            }
        )
    return sentences


def process_file(input_path: str) -> str:
    """Preprocess an interview text file into a uniform processed_doc JSON.

    Output shape:

        {
          "raw_text": "...",
          "segments": [
             {
               "time": "Sunday, October 16, 2022 – 12:03 p.m.",  # from header if present
               "speaker": "Detective Murphy",
               "text": "...",
               "sentences": [ { "text": "...", "speaker": "...", ... }, ... ],
               "events": []
             },
             ...
          ],
          "sentences": [...],   # flat list of all sentence dicts
          "meta": {
              "source_path": "...",
              "doc_title": "...",
          },
          "doc_type": "interview"
        }
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    raw_text = normalize_text(raw_text)
    lines = raw_text.split("\n")

    # Header extraction
    title, header_time = parse_header(lines)

    # Body text after header (skip first 1-2 non-empty lines)
    non_empty_indices = [i for i, ln in enumerate(lines) if ln.strip()]
    body_start_idx = 0
    if non_empty_indices:
        body_start_idx = non_empty_indices[0] + 1
        if len(non_empty_indices) > 1 and non_empty_indices[1] == body_start_idx:
            body_start_idx = non_empty_indices[1] + 1
    body_text = "\n".join(lines[body_start_idx:]).strip()

    # Speaker segmentation
    speaker_segments = segment_by_speaker(body_text)

    # spaCy for sentence segmentation and simple POS tags
    nlp = English()
    nlp.add_pipe("sentencizer")

    processed_segments = []
    all_sentences = []
    for seg in speaker_segments:
        speaker = seg["speaker"]
        seg_text = seg["text"]
        sentences = annotate_sentences(seg_text, speaker, nlp)
        segment_obj = {
            "time": header_time,  # same header time for all turns, if present
            "speaker": speaker,
            "text": seg_text,
            "sentences": sentences,
            "events": [],
        }
        processed_segments.append(segment_obj)
        all_sentences.extend(sentences)

    processed_doc = {
        "raw_text": raw_text,
        "segments": processed_segments,
        "sentences": all_sentences,
        "meta": {
            "source_path": input_path,
            "doc_title": title,
        },
        "doc_type": "interview",
    }

    output_path = input_path.replace(".txt", "_interview_processed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_doc, f, indent=4, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess interview text files.")
    parser.add_argument("input_file", help="Path to the input .txt interview file")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: File {args.input_file} does not exist.")
        return

    output_path = process_file(args.input_file)
    print(f"Processed interview saved to: {output_path}")


if __name__ == "__main__":
    main()