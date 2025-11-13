# NER Module

This module contains `ner_module.py`, a script for Named Entity Recognition on preprocessed narrative JSON.

## Purpose
- Loads spaCy transformer NER model.
- Adds custom EntityRuler for forensic roles and objects.
- Normalizes entity strings.
- Extracts entities for each sentence in the JSON.

## Usage
```bash
python ner_module.py --input narrative_processed.json
```
Output: `narrative_processed_ner.json`

## Dependencies
- spacy
- en_core_web_trf model
- json, argparse

## Output Format
Input JSON enriched with `entities` key in each sentence.