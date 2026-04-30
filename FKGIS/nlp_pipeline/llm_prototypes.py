"""
Prototype spaCy-LLM pipelines for:
- NER (LLM-assisted)
- Relation extraction (LLM-assisted)
- Coref-style clustering (LLM-assisted, generic)

This file is meant to be used on the `spacy-llm` branch in a clean
Python 3.11 environment with:

    pip install "spacy>=3.7,<3.8" "spacy-llm>=0.7,<0.8" "openai>=1.0.0"
    python -m spacy download en_core_web_sm

You will need a working OPENAI_API_KEY in your environment.

NOTE:
- The exact registry keys for @llm_models depend on the spacy-llm
  version and provider adapters you install. Adjust "@llm_models"
  below to match the docs for your version (e.g., "spacy.OpenAI.v1").
"""

from __future__ import annotations

from typing import List, Dict, Any

import spacy
from spacy.tokens import Doc


def make_llm_ner_nlp() -> "spacy.language.Language":
    """Create a spaCy-LLM NER pipeline.

    This uses an LLM "llm" component for NER on top of a blank English
    pipeline. In practice you may want to prepend a tok2vec/transformer
    or combine with your existing en_core_web_trf model.
    """
    nlp = spacy.blank("en")

    # LLM wrapper for NER.
    # IMPORTANT: use one of the registered OpenAI adapters listed in the error:
    # e.g. "spacy.GPT-3-5.v3" or "spacy.GPT-4.v3".
    nlp.add_pipe(
        "llm",
        config={
            "task": {
                "@llm_tasks": "spacy.NER.v1",
                # Optional: restrict labels to those you care about
                # "labels": ["PERSON", "ORG", "LOC", "EVIDENCE", "ROLE"],
            },
            "model": {
                "@llm_models": "spacy.GPT-3-5.v3",
                "name": "gpt-3.5-turbo",
            },
        },
    )

    return nlp


def make_llm_rel_nlp() -> "spacy.language.Language":
    """Create a spaCy-LLM relation extraction pipeline.

    This uses the spacy.REL.v1 task if available in your spacy-llm
    version. You must define your relation labels and patterns.

    Example relation labels for a forensic setting:
      - "INVESTIGATED"
      - "RELATED_TO"
      - "EMPLOYED_BY"
      - "FAMILY_OF"
    """
    nlp = spacy.blank("en")

    nlp.add_pipe(
        "llm",
        config={
            "task": {
                "@llm_tasks": "spacy.REL.v1",
                # Define your relation labels here
                "labels": ["INVESTIGATED", "RELATED_TO", "EMPLOYED_BY", "FAMILY_OF"],
                # Optional: you can also pass hints or examples in the task config.
            },
            "model": {
                "@llm_models": "spacy.GPT-3-5.v3",
                "name": "gpt-3.5-turbo",
            },
        },
    )

    return nlp


def make_llm_coref_nlp() -> "spacy.language.Language":
    """Create a generic spaCy-LLM coref-like pipeline.

    spaCy-LLM does not (yet) ship a fully opinionated coref task,
    so we use a Generic task that prompts the LLM to return JSON
    with clusters.

    Expected JSON schema (example):

        {
          "clusters": [
            {
              "canonical": "Officer Willits",
              "mentions": ["Officer Willits", "he", "his"]
            },
            ...
          ]
        }

    You can later map these clusters into your GlobalEntityPool.
    """
    nlp = spacy.blank("en")

    # Generic task with JSON parser
    nlp.add_pipe(
        "llm",
        config={
            "task": {
                "@llm_tasks": "spacy.Generic.v1",
                "template": (
                    "You are a coreference resolver for short forensic texts.\n"
                    "Text: {text}\n\n"
                    "Identify coreference clusters of entity mentions.\n"
                    "Return ONLY JSON with the following schema:\n"
                    "{{\"clusters\": [{{\"canonical\": str, \"mentions\": [str, ...]}}, ...]}}\n"
                ),
                "output_parser": {
                    "@llm_parsers": "spacy.Json.v1",
                },
            },
            "model": {
                "@llm_models": "spacy.GPT-3-5.v3",
                "name": "gpt-3.5-turbo",
            },
        },
    )

    return nlp


def demo_ner():
    nlp = make_llm_ner_nlp()
    text = "Detective Armstrong interviewed witness Rebecca Pace at 12:03 p.m. in Oxford."
    doc = nlp(text)
    print("LLM-NER demo:")
    for ent in doc.ents:
        print(f"  {ent.text!r} ({ent.label_})")
    print()


def demo_rel():
    nlp = make_llm_rel_nlp()
    text = "Detective Armstrong interviewed witness Rebecca Pace at the scene."
    doc = nlp(text)
    print("LLM-REL demo:")
    # Depending on implementation of spacy.REL.v1, relations may be stored on doc._.rel or similar.
    # Here we just print the raw llm response if exposed.
    if hasattr(doc._, "llm_response"):
        print(doc._.llm_response)
    else:
        print("  (Inspect doc._ attributes for relation output.)")
    print()


def demo_coref():
    nlp = make_llm_coref_nlp()
    text = (
        "Officer Willits arrived at the scene. He secured the residence and "
        "later said his actions were according to protocol."
    )
    doc = nlp(text)
    print("LLM-COREF demo:")
    if hasattr(doc._, "llm_response"):
        # Expected to be a dict with "clusters"
        print(doc._.llm_response)
    else:
        print("  (Inspect doc._ attributes for coref output.)")
    print()


if __name__ == "__main__":
    # Run all three demos when called directly
    demo_ner()
    demo_rel()
    demo_coref()