# Forensic NLP Pipeline (Kimberly Pace Case)

Deterministic, rule-based NLP pipeline tailored to the Kimberly Pace case documents under:

- `nlp_pipeline/case_docs/IncidentReport_RO.txt`
- `nlp_pipeline/case_docs/BIO_Kimberley_Pace.txt`
- `nlp_pipeline/case_docs/Interview_Kimberley_*.txt`

No LLMs are used here. This is the **baseline** model; spaCy-LLM integration will be added later on a separate branch.

---

## 1. Overall pipeline order

The case-level pipeline (run via `pipeline.py`) executes in this order:

1. **Document Loader**  
   - [`document_loader/load_docs.py`](document_loader/load_docs.py:1)  
   - Discovers raw `.txt` documents under `nlp_pipeline/case_docs/`.
   - Detects `doc_type`:
     - `IncidentReport_RO.txt` → `narrative`
     - `BIO_*.txt` → `biography`
     - `Interview_*.txt` → `interview`

2. **Preprocessing (type-specific)**  
   - Narrative: [`preprocessing/prep_ROnarrative.py`](preprocessing/prep_ROnarrative.py:1)  
   - Biography: [`preprocessing/prep_biography.py`](preprocessing/prep_biography.py:1)  
   - Interview: [`preprocessing/prep_interview.py`](preprocessing/prep_interview.py:1)  

   All produce a uniform `processed_doc` JSON:

   ```json
   {
     "raw_text": "...",
     "segments": [...],
     "sentences": [...],
     "meta": {...},
     "doc_type": "narrative" | "biography" | "interview"
   }
   ```

3. **NER**  
   - [`ner_and_dependency_parsing/ner_module.py`](ner_and_dependency_parsing/ner_module.py:1)  
   - `run_ner_on_document()` attaches:
     - `sentence["entities"]` (per sentence),
     - flattened `processed_doc["entities"]` with `(segment_index, sentence_index, start, end)`.

4. **Dependency parsing**  
   - [`dependency_parsing/dep_module.py`](dependency_parsing/dep_module.py:1)  
   - `attach_dependencies()` adds for each sentence:

   ```json
   "tokens": [
     {
       "text": "...",
       "lemma": "...",
       "pos": "...",
       "tag": "...",
       "dep": "...",
       "head": 3
     },
     ...
   ],
   "root_index": 0
   ```

5. **Coreference (document-level)**  
   - [`coreference/coref_global_pool.py`](coreference/coref_global_pool.py:1)  
   - `run_coref_on_document()`:
     - uses spaCy NER + simple pronoun rules,
     - builds `processed_doc["coref_entities"]` and alias `processed_doc["coref"]`.

6. **Mention unification (pronoun + abbrev)**  
   - Base: [`mention_unification/mention_unifier.py`](mention_unification/mention_unifier.py:1)  
   - Case wrapper: [`mention_unification/unify_mentions.py`](mention_unification/unify_mentions.py:1)  

   `unify_mentions()`:

   - Uses `coref_entities` to map pronouns (he, she, they, etc.) → main mention.
   - Builds `processed_doc["unified_sentences"]` with:
     - pronoun substitution,
     - case-specific expansions: `R/O` → `Reporting Officer`, `R/Is` → `Reporting Investigators`, `CSU` → `Crime Scene Unit`, `ETA` → `Estimated Time of Arrival`.

7. **Relation extraction (forensic, rule-based)**  
   - Base: [`relation_extraction/basic_rel_extractor.py`](relation_extraction/basic_rel_extractor.py:1)  
   - Case refinement: [`relation_extraction/basic_spacy_rel_extractor.py`](relation_extraction/basic_spacy_rel_extractor.py:1)  

   Flow:

   - `basic_rel_extractor.extract_relations(processed_doc)`:
     - parses each sentence with `en_core_web_trf`,
     - extracts subject–verb–object triples using deps (`nsubj`, `dobj`, `pobj`, `attr`, `dative`, `prep`→`pobj`),
     - filters by argument length and entity overlap.

   - `refine_case_relations(processed_doc)`:
     - keeps only relations whose verb lemma is in a **case lexicon**:
       - `arrive, secure, observe, remove, pronounce, respond, identify, meet, direct, interview, examine, report, say, indicate, call, dispatch`
     - backstop: retains relations mentioning key case actors/objects:
       - officers (Willits, Harding, Armstrong, Murphy, Johnson),
       - Animal Control (Lukens, Sanchez),
       - witnesses (Cheryl Weston, Rebecca/Becky Pace, Jeremy Gladwell),
       - victim (Kimberly / Kim Pace),
       - dog (Thoreau),
       - body/victim.
     - Annotates relations with `relation_type` where known:
       - e.g., `arrive` → `ARRIVED_AT`, `secure` → `SECURED`, `interview` → `INTERVIEWED`, `report/say/indicate` → `CLAIMED`, `call/dispatch` → `DISPATCHED`.

   Output:

   ```json
   "relations": [
     {
       "subject": "Reporting Officer Willits",
       "predicate": "secure",
       "object": "the scene",
       "sentence_id": 19,
       "relation_type": "SECURED"
     },
     ...
   ]
   ```

8. **Event extraction**  
   - Base: [`event_construction/event_builder.py`](event_construction/event_builder.py:1)  
   - Case wrapper: [`event_extraction/build_events.py`](event_extraction/build_events.py:1)  

   Steps:

   - Base builder converts relations → events:

     ```json
     {
       "description": "Reporting Officer Willits secured the scene...",
       "subject": "Reporting Officer Willits",
       "predicate": "secure",
       "object": "the scene",
       "sentence_id": 19,
       "time_raw": "11:27 a.m",
       "time_resolved": null
     }
     ```

   - Case wrapper:
     - `_collect_sentence_entities()` uses `(segment_index, sentence_index)` to map entities to `sentence_id`.
     - `_attach_participants_and_locations()`:
       - `participants`: entities with labels in `{"PERSON", "ROLE", "ORG", "EVIDENCE"}` found in the sentence (officers, witnesses, dog, body, etc.).
       - `locations`: from LOC entities or keywords (residence, basement, bedroom, Big Bad Breakfast, The Lucky Café, C'est Belle, etc.).
     - `_assign_event_ids()` assigns `EV1`, `EV2`, … for stable IDs.

   Final events:

   ```json
   "events": [
     {
       "event_id": "EV10",
       "description": "...",
       "subject": "...",
       "predicate": "...",
       "object": "...",
       "sentence_id": 19,
       "time_raw": "...",
       "time_resolved": null,
       "participants": ["Reporting Officer Willits", "Crime Scene Unit"],
       "locations": ["residence", "basement"]
     },
     ...
   ]
   ```

9. **Timeline builder**  
   - [`timeline/timeline_builder.py`](timeline/timeline_builder.py:1)  

   Behaviors:

   - `normalize_event_times_for_doc(processed_doc)`:
     - Parses `time_raw` like `"11:06 a.m."`, `"11:42 a.m."`, `"12:53 p.m."`, `"4:35 p.m."`, `"12:03 p.m."`, `"12:44 p.m."`, `"9:30 p.m."`.
     - Anchors them to **incident date** `"2022-10-16"`:
       - `"11:06 a.m."` → `"2022-10-16T11:06:00"`.
   - `build_before_after_edges(case_docs)`:
     - Collects all events with `time_resolved` across narrative + interviews.
     - Sorts chronologically.
     - For each consecutive pair, adds a `"BEFORE"` edge (and later an explicit `"AFTER"` reverse in the KG layer).

   Output of:

   ```python
   build_timeline_for_case(case_id, case_docs_entries)
   ```

   ```json
   {
     "case_id": "CASE123",
     "timeline_edges": [
       {
         "source_event_id": "EV1",
         "target_event_id": "EV2",
         "relation": "BEFORE",
         "source_doc": "IncidentReport_RO",
         "target_doc": "IncidentReport_RO",
         "source_doc_type": "narrative",
         "target_doc_type": "narrative"
       },
       ...
     ]
   }
   ```

10. **Knowledge graph builder**  
    - [`kg/graph_builder.py`](kg/graph_builder.py:1)  

    `build_nodes_and_edges_for_case(case_id, case_docs_entries, timeline_info)`:

    - Node types:
      - `Entity`: persons, officers, witnesses, dog, objects (briefcase, handbag, body, residence).
      - `Event`: one per `event_id`.
      - `Time`: one per unique `time_resolved`.
      - `Location`: location strings from events/keywords.

    - Edge types:
      - `PARTICIPATED_IN` (Entity → Event)
      - `LOCATED_AT` (Event → Location)
      - `HAS_TIME` (Event → Time)
      - `BEFORE` / `AFTER` (Event ↔ Event) from timeline edges
      - `CLAIMED` (Entity → Entity for claim/statement)
      - `OBSERVED` (Entity → Entity or object for `OBSERVED` relations)
      - `OWNS` (Entity → Entity for ownership semantics, e.g., Kim → Thoreau, Kimberly → residence/briefcase/handbag)

    `save_graph_json(graph, base_dir)` writes:

    - `graph_nodes.json` (list of nodes)
    - `graph_edges.json` (list of edges)

    into `nlp_pipeline/output/` (or the specified directory).

11. **Verification**  
    - [`verification/verify_outputs.py`](verification/verify_outputs.py:1)  

    `verify_case_outputs(...)` checks:

    - At least one non-empty `entities` list across all docs.
    - `relations_count ≥ min_relations` (default 5).
    - `events_count > 0`.
    - `events_with_subject_and_predicate > 0`.
    - Timeline has at least some `BEFORE` edges.
    - No obvious malformed entries:
      - entities with `null` text,
      - relations missing subject/object,
      - events with `null` `event_id`.
    - Graph nodes/edges non-empty.

    Returns a verification report embedded into `case_output.json`.

---

## 2. Orchestrator (`pipeline.py`)

[`pipeline.py`](pipeline.py:1) supports two main flows:

### 2.1. Narrative-only pipeline (legacy)

```bash
python -m FKGIS.nlp_pipeline.pipeline \
  --case CASE123 \
  --input FKGIS/nlp_pipeline/sampleRO.txt \
  --step full
```

Outputs:

- narrative preprocessed JSON
- narrative NER JSON
- `CASE123_global_entities.json`
- narrative enriched events JSON

### 2.2. Full multi-document case pipeline (recommended)

```bash
python -m FKGIS.nlp_pipeline.pipeline \
  --case CASE123 \
  --step case
```

This:

1. Discovers docs under `nlp_pipeline/case_docs/`.
2. Runs the full stack (1–11 above) for:
   - Incident report narrative,
   - Kimberly’s biography,
   - Interviews (sister, friend, neighbor).
3. Writes per-doc enriched JSONs into `nlp_pipeline/output/`:
   - `IncidentReport_RO_enriched.json`
   - `BIO_Kimberley_Pace_enriched.json`
   - `Interview_Kimberley_*.json`
4. Builds:
   - `timeline_edges` (BEFORE relationships).
   - `graph_nodes.json` and `graph_edges.json` (ready to ingest into Neo4j).
5. Produces final:

   ```text
   nlp_pipeline/output/case_output.json
   ```

   with:

   - `case_id`
   - document list
   - embedded timeline info
   - paths to `graph_nodes.json` and `graph_edges.json`
   - verification report (`passed`, `checks`, `issues`).

---

## 3. Running the pipeline

From project root (`/home/aayush/Desktop/Projects/FKGIS`):

### 3.1. Full case pipeline

```bash
python3 -m FKGIS.nlp_pipeline.pipeline --case CASE123 --step case
```

Artifacts in `nlp_pipeline/output/`:

- `case_output.json`
- `graph_nodes.json`
- `graph_edges.json`
- `*_enriched.json` for each source document.

### 3.2. Narrative-only (for testing incremental changes)

```bash
python3 -m FKGIS.nlp_pipeline.pipeline \
  --case CASE123 \
  --input FKGIS/nlp_pipeline/sampleRO.txt \
  --step full
```

---

## 4. JSON structures

### 4.1. `case_output.json`

```json
{
  "case_id": "CASE123",
  "documents": [
    { "doc_name": "IncidentReport_RO", "doc_type": "narrative" },
    { "doc_name": "BIO_Kimberley_Pace", "doc_type": "biography" },
    { "doc_name": "Interview_Kimberley_Sister", "doc_type": "interview" },
    ...
  ],
  "timeline": {
    "case_id": "CASE123",
    "timeline_edges": [ ... ]
  },
  "graph_nodes_path": "nlp_pipeline/output/graph_nodes.json",
  "graph_edges_path": "nlp_pipeline/output/graph_edges.json",
  "verification": {
    "case_id": "CASE123",
    "passed": true,
    "checks": {
      "entities_non_empty": true,
      "relations_count": 42,
      "relations_count_ok": true,
      "events_count": 30,
      "events_with_subject_and_predicate": 30,
      "timeline_has_before_after": true,
      "graph_nodes_non_empty": true,
      "graph_edges_non_empty": true,
      "issues": []
    }
  }
}
```

### 4.2. `graph_nodes.json`

Each node has:

- `id`: unique string (`ENT_*`, `EV*`, `TIME_*`, `LOC_*`)
- `type`: one of `Entity`, `Event`, `Time`, `Location`
- plus type-specific fields (`text`, `norm`, `label`, `description`, `value`, etc.).

### 4.3. `graph_edges.json`

Each edge has:

- `source`
- `target`
- `edge_type`:
  - `PARTICIPATED_IN`, `BEFORE`, `AFTER`, `LOCATED_AT`, `CLAIMED`, `OBSERVED`, `OWNS`, `HAS_TIME`
- `case_id`
- optionally `doc_name`, `doc_type` for relation-based edges.

---

## 5. Notes

- Everything is deterministic and rule-based—no spaCy-LLM in this baseline.
- All components are JSON-serializable.
- All heuristics are tailored specifically to the Kimberly Pace case documents:
  - Recognizing key names (officers, witnesses, victim, dog),
  - Case-specific verbs and abbreviations,
  - Time expressions and locations mentioned in the narrative and interviews.

This baseline provides:

- Refined entities,
- Coreference-resolved sentences (through unification),
- Clean relations,
- Clean events,
- Timeline edges,
- Graph node/edge exports for Neo4j,
- A verification report to guard against broken runs.

Future work: build LLM wrappers on top of these outputs for:
- improved NER,
- richer relation types,
- nuanced temporal reasoning and stance detection in interviews.