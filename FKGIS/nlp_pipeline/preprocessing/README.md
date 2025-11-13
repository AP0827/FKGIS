# Preprocessing Module

This module contains `prep_ROnarrative.py`, a script for preprocessing "Reporting Officer's Narrative" crime scene reports.

## Purpose
- Normalizes and cleans raw text files.
- Expands abbreviations (e.g., R/O → Reporting Officer).
- Segments text into temporal/logical sections.
- Splits sections into sentences with metadata (time, actor, environment).
- Extracts potential event candidates.

## Usage
```bash
python prep_ROnarrative.py input.txt
```
Output: `input_ROnarrative_processed.json`

## Dependencies
- spacy
- re, json, os, argparse

## Output Format
JSON with segments containing sentences and events.