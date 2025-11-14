import re
import json
import os
import argparse

from spacy.lang.en import English


DATE_INLINE_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Basic normalization for biography text."""
    text = text.replace("\r\n", "\n")
    # Collapse multiple blank lines but keep paragraph boundaries
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_paragraphs(text: str):
    """Split text into paragraphs separated by blank lines."""
    paragraphs = []
    current_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current_lines:
                paragraphs.append(" ".join(current_lines).strip())
                current_lines = []
        else:
            current_lines.append(stripped)

    if current_lines:
        paragraphs.append(" ".join(current_lines).strip())

    return paragraphs


def annotate_sentences(paragraph_text: str, nlp):
    """Split a paragraph into sentences with simple flags."""
    doc = nlp(paragraph_text)
    sentences = []
    for sent in doc.sents:
        text = sent.text.strip()
        if not text:
            continue
        has_time = bool(DATE_INLINE_PATTERN.search(text))
        # Actor heuristic: any proper noun or first-person pronoun
        has_actor = any(t.pos_ in {"PROPN", "PRON"} for t in sent)
        has_env = False
        sentences.append(
            {
                "text": text,
                "has_time": has_time,
                "has_actor": has_actor,
                "has_env": has_env,
            }
        )
    return sentences


def process_file(input_path: str) -> str:
    """Preprocess a biography text file into a uniform processed_doc JSON.

    Output shape:

        {
          "raw_text": "...",
          "segments": [
             {
               "time": null,
               "text": "... paragraph ...",
               "sentences": [ { "text": "...", ... }, ... ],
               "events": []
             },
             ...
          ],
          "sentences": [...],   # flat list of all sentence dicts
          "meta": {
              "source_path": "...",
              "doc_title": "..."
          },
          "doc_type": "biography"
        }
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    raw_text = normalize_text(raw_text)
    paragraphs = split_paragraphs(raw_text)

    # Use the first non-empty line as a simple title, if present
    first_line = raw_text.split("\n", 1)[0].strip() if raw_text else None
    title = first_line or os.path.basename(input_path)

    # spaCy for sentence segmentation
    nlp = English()
    nlp.add_pipe("sentencizer")

    processed_segments = []
    all_sentences = []

    for para in paragraphs:
        sentences = annotate_sentences(para, nlp)
        seg_obj = {
            "time": None,
            "text": para,
            "sentences": sentences,
            "events": [],
        }
        processed_segments.append(seg_obj)
        all_sentences.extend(sentences)

    processed_doc = {
        "raw_text": raw_text,
        "segments": processed_segments,
        "sentences": all_sentences,
        "meta": {
            "source_path": input_path,
            "doc_title": title,
        },
        "doc_type": "biography",
    }

    output_path = input_path.replace(".txt", "_bio_processed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_doc, f, indent=4, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess biography text files.")
    parser.add_argument("input_file", help="Path to the input .txt biography file")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: File {args.input_file} does not exist.")
        return

    output_path = process_file(args.input_file)
    print(f"Processed biography saved to: {output_path}")


if __name__ == "__main__":
    main()