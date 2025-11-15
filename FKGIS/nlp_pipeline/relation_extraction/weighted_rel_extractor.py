#!/usr/bin/env python3
"""
relation_extraction.py
Phase 1: Extract candidate relations from segmented JSON input.

Usage:
  python relation_extraction.py input_case.json relations.json
"""

import json, math, re, sys
from collections import defaultdict
import spacy

# ------------------------------
# Config (weights / lexical lists)
# ------------------------------
WEIGHTS = {
    "f_role": 0.25,
    "f_event": 0.20,
    "f_dep": 0.15,
    "f_temporal": 0.10,
    "f_verb": 0.10,
    "f_scene": 0.08,
    "f_type": 0.07,
    "f_dist": 0.03,
    "f_mod": 0.02
}

HIGH_VERBS = {"discover", "discovere", "report", "reported", "secure", "secured", "detain", "detained", "pronounce", "pronounced", "dispatch", "dispatched", "arrive", "arrived", "find", "found", "observe", "observed"}
NEUTRAL_VERBS = {"say","said","note","noted","see","saw","meet","met","proceed","proceeded"}

MODAL_RE = re.compile(r"\b(may|might|could|possibly|allegedly|suspect|suspected|to the best of his knowledge)\b", re.I)

# Basic type compat matrix (extend for your domain)
TYPE_MATRIX = defaultdict(lambda: defaultdict(lambda: 0.3))
TYPE_MATRIX["PERSON"]["PERSON"] = 0.9
TYPE_MATRIX["PERSON"]["LOC"] = 0.9
TYPE_MATRIX["PERSON"]["EVIDENCE"] = 0.7
TYPE_MATRIX["ROLE"]["LOC"] = 0.9
TYPE_MATRIX["ROLE"]["EVIDENCE"] = 0.8
TYPE_MATRIX["LOC"]["EVIDENCE"] = 0.6

# Role mapping matrix (coarse)
ROLE_MATRIX = defaultdict(lambda: defaultdict(lambda: 0.4))
ROLE_MATRIX["OFFICER"]["SCENE"] = 1.0
ROLE_MATRIX["WITNESS"]["VICTIM"] = 0.9
ROLE_MATRIX["WITNESS"]["SCENE"] = 0.9
ROLE_MATRIX["CORONER"]["VICTIM"] = 1.0

# ------------------------------
# Helpers
# ------------------------------
def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))

def infer_role(label, text):
    lab = (label or "").upper()
    t = (text or "").lower()
    if lab == "ROLE" or "officer" in t or "deputy" in t:
        return "OFFICER"
    if lab == "PERSON":
        if "victim" in t or "deceased" in t: return "VICTIM"
        if "caller" in t or "witness" in t: return "WITNESS"
        return "PERSON"
    if lab == "LOC": return "SCENE"
    if lab == "EVIDENCE": return "EVIDENCE"
    return "OTHER"

# Dependency path length: ascend to root sets and compute distance via head chain
def dependency_path_len(tok1, tok2):
    # build ancestor sets (by index)
    a1, cur = set(), tok1
    while True:
        a1.add(cur.i)
        if cur.head == cur: break
        cur = cur.head
    a2, cur = set(), tok2
    while True:
        a2.add(cur.i)
        if cur.head == cur: break
        cur = cur.head
    # find LCA by walking up tok1
    cur = tok1
    steps1 = 0
    lca = None
    while True:
        if cur.i in a2:
            lca = cur
            break
        if cur.head == cur: break
        cur = cur.head
        steps1 += 1
    if lca is None:
        return 20
    cur = tok2
    steps2 = 0
    while cur.i != lca.i:
        if cur.head == cur: break
        cur = cur.head
        steps2 += 1
    return steps1 + steps2

# compute features for a pair given spaCy sentence doc and JSON sentence
def compute_features(e1, e2, sent_doc, sent_json, all_segments):
    # identify token spans
    span1 = sent_doc.char_span(e1.get("start",0), e1.get("end",0), alignment_mode="expand")
    span2 = sent_doc.char_span(e2.get("start",0), e2.get("end",0), alignment_mode="expand")
    if span1 is None: span1 = sent_doc[0:1]
    if span2 is None: span2 = sent_doc[0:1]
    t1 = span1.root
    t2 = span2.root

    # f_dep
    path_len = dependency_path_len(t1,t2)
    f_dep = 1.0 / (1.0 + path_len)

    # f_dist
    f_dist = 1.0 / (1.0 + abs(t1.i - t2.i))

    # f_verb (nearest verb to midpoint)
    mid = (t1.i + t2.i)//2
    nearest_verb = None
    for d in range(0, max(mid, len(sent_doc)-mid)+1):
        for idx in (mid-d, mid+d):
            if 0 <= idx < len(sent_doc) and sent_doc[idx].pos_ == "VERB":
                nearest_verb = sent_doc[idx]
                break
        if nearest_verb: break
    verb_lemma = nearest_verb.lemma_.lower() if nearest_verb else ""
    if verb_lemma in HIGH_VERBS: f_verb = 1.0
    elif verb_lemma in NEUTRAL_VERBS: f_verb = 0.5
    else: f_verb = 0.0

    # f_type
    lab1 = e1.get("label","").upper(); lab2 = e2.get("label","").upper()
    f_type = TYPE_MATRIX[lab1].get(lab2, 0.3)

    # f_mod
    penalty = 0.5 if MODAL_RE.search(sent_json.get("text","")) else 0.0
    f_mod = 1.0 - penalty

    # f_role
    r1 = infer_role(lab1, e1.get("text","")); r2 = infer_role(lab2, e2.get("text",""))
    f_role = ROLE_MATRIX[r1].get(r2, 0.4)

    # f_temporal: fraction of segments where both norms occur
    norm1 = (e1.get("norm") or e1.get("text","")).lower()
    norm2 = (e2.get("norm") or e2.get("text","")).lower()
    total = max(1, len(all_segments)); count = 0
    for seg in all_segments:
        txt = seg.get("text","").lower()
        if norm1 in txt and norm2 in txt: count += 1
    f_temporal = min(1.0, count / total)

    # f_event: heuristic
    txt = sent_json.get("text","").lower()
    if "pronounced" in txt or "pronounce" in txt: f_event = 1.0
    elif "911" in txt or "caller" in txt: f_event = 0.95
    elif "dispatched" in txt or "dispatch" in txt: f_event = 0.9
    elif "removed from the scene" in txt or "transported" in txt: f_event = 0.8
    else: f_event = 0.0

    # f_scene: boost if either looks like body/scene/evidence
    f_scene = 1.0 if ("body" in e1.get("text","").lower() or "body" in e2.get("text","").lower() or lab1 in ("EVIDENCE","VICTIM") or lab2 in ("EVIDENCE","VICTIM")) else 0.0

    features = {
        "f_dep": round(f_dep,4),
        "f_dist": round(f_dist,4),
        "f_verb": round(f_verb,4),
        "f_type": round(f_type,4),
        "f_mod": round(f_mod,4),
        "f_role": round(f_role,4),
        "f_temporal": round(f_temporal,4),
        "f_event": round(f_event,4),
        "f_scene": round(f_scene,4)
    }
    return features

def linear_score(features, weights=WEIGHTS):
    s = 0.0
    for k,w in weights.items():
        s += w * features.get(k,0.0)
    return s

# ------------------------------
# Main runner
# ------------------------------
def run(input_json_path, output_json_path, nlp_model="en_core_web_trf"):
    print("Loading input:", input_json_path)
    data = json.load(open(input_json_path, "r", encoding="utf8"))
    segments = data.get("segments", [])
    nlp = spacy.load(nlp_model, disable=["ner"])  # do not override your pre-annotated entities
    candidates = []

    for seg_idx, seg in enumerate(segments):
        for sent_idx, sent in enumerate(seg.get("sentences", [])):
            text = sent.get("text","")
            if not text.strip(): continue
            doc = nlp(text)
            ents = sent.get("entities", [])
            # pairwise unordered pairs
            for i in range(len(ents)):
                for j in range(i+1,len(ents)):
                    e1 = ents[i]; e2 = ents[j]
                    features = compute_features(e1,e2,doc,sent,segments)
                    raw = linear_score(features)
                    # map raw to probability (toy mapping; replace with trained logistic when available)
                    prob = sigmoid(3*(raw-0.5))
                    cand = {
                        "head": (e1.get("norm") or e1.get("text","")).lower(),
                        "tail": (e2.get("norm") or e2.get("text","")).lower(),
                        "head_text": e1.get("text",""),
                        "tail_text": e2.get("text",""),
                        "segment": seg_idx,
                        "sentence_index": sent_idx,
                        "sentence_text": text,
                        "features": features,
                        "raw_score": round(raw,4),
                        "prob": round(prob,4)
                    }
                    candidates.append(cand)

    # save all candidates (downstream graph module will filter threshold)
    with open(output_json_path, "w", encoding="utf8") as f:
        json.dump(candidates, f, indent=2)
    print("Wrote", len(candidates), "candidates to", output_json_path)

# CLI
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python relation_extraction.py <input_json> <output_relations_json>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
