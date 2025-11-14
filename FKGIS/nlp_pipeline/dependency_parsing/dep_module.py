from __future__ import annotations

from typing import Dict, Any, List, Optional

import json
import os
import argparse

import spacy
from spacy.tokens import Doc


# Lazy-loaded spaCy pipeline with dependency parser enabled
_NLP: Optional["spacy.language.Language"] = None


def _get_nlp() -> "spacy.language.Language":
    """Return a spaCy pipeline with a dependency parser.

    Uses the same transformer model name as the NER/coref stages for consistency.
    """
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_trf")
    return _NLP


def _iter_sentences_for_deps(processed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of sentences to parse for dependencies.

    Preference order:
      1) processed_doc["unified_sentences"] (after pronoun substitution)
      2) processed_doc["sentences"]
    """
    unified = processed_doc.get("unified_sentences") or []
    if unified:
        return unified
    return processed_doc.get("sentences") or []


def _attach_deps_to_sentence(sent_dict: Dict[str, Any], doc: Doc) -> None:
    """Attach a JSON-serializable dependency representation to a sentence dict.

    We store, for each token in the sentence:

        {
          "text": str,
          "lemma": str,
          "pos": str,
          "tag": str,
          "dep": str,
          "head": int,   # index of head token within this sentence (0-based),
                         # or -1 if the token is the root
        }

    The resulting list is stored under sent_dict["tokens"], and the index of the
    root token (if any) is stored under sent_dict["root_index"].

    Note: we assume the Doc was created from a single sentence string, so we
    treat the entire Doc as the sentence span.
    """
    tokens: List[Dict[str, Any]] = []
    span = doc
    root_index = -1

    for i, token in enumerate(span):
        # In this usage, the Doc is just this sentence, so token indices are
        # already local. If the token is its own head, mark as root.
        if token.head.i == token.i:
            head_in_span = -1
            root_index = i
        else:
            head_in_span = token.head.i

        tokens.append(
            {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "head": head_in_span,
            }
        )

    sent_dict["tokens"] = tokens
    if root_index != -1:
        sent_dict["root_index"] = root_index


def attach_dependencies(processed_doc: Dict[str, Any], nlp: Optional["spacy.language.Language"] = None) -> Dict[str, Any]:
    """Attach dependency parses to each sentence in a processed_doc.

    Behavior:
      - Operates primarily on processed_doc["unified_sentences"] if present
        (since pronoun substitution yields cleaner arguments), otherwise
        falls back to processed_doc["sentences"].
      - For each sentence:
          * Run spaCy with a model that has a dependency parser.
          * Attach a JSON-serializable "tokens" list with dep/pos/heads.
      - Sets processed_doc["has_dependencies"] = True.

    The resulting structure is suitable for downstream consumers like
    relation extraction or custom rule-based logic, without needing to
    store spaCy Doc objects directly.
    """
    sentences = _iter_sentences_for_deps(processed_doc)
    if not sentences:
        processed_doc["has_dependencies"] = False
        return processed_doc

    if nlp is None:
        nlp = _get_nlp()

    for idx, sent in enumerate(sentences):
        text = (sent.get("text") or "").strip()
        if not text:
            continue
        doc = nlp(text)
        _attach_deps_to_sentence(sent, doc)

    processed_doc["has_dependencies"] = True
    return processed_doc


def process_json_file(input_path: str, nlp: Optional["spacy.language.Language"] = None) -> str:
    """Load a processed_doc JSON, attach dependency parses, and save.

    This function is compatible with:
      - narrative processed JSON from prep_ROnarrative,
      - biography processed JSON from prep_biography,
      - interview processed JSON from prep_interview,
      - or any other document that follows the same processed_doc schema.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {input_path}: {e}")

    doc_type = data.get("doc_type", "unknown")
    print(f"Attaching dependencies for {input_path} (doc_type={doc_type}).")

    enriched = attach_dependencies(data, nlp)

    output_path = input_path.replace(".json", "_dep.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=4, ensure_ascii=False)

    print(f"Dependency-enriched document saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Attach dependency parses to a processed_doc JSON."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the processed_doc JSON (output of preprocessing stage).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: File {args.input} does not exist.")
        return

    nlp = _get_nlp()
    process_json_file(args.input, nlp)


if __name__ == "__main__":
    main()