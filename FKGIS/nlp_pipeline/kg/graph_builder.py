from __future__ import annotations

from typing import Dict, Any, List, Tuple, Set, Optional
from pathlib import Path
import json
import networkx as nx # Import networkx
import time


# Node / edge types required by the spec
NODE_TYPE_ENTITY = "Entity"
NODE_TYPE_EVENT = "Event"
NODE_TYPE_TIME = "Time"
NODE_TYPE_LOCATION = "Location"

EDGE_TYPE_PARTICIPATED_IN = "PARTICIPATED_IN"
EDGE_TYPE_BEFORE = "BEFORE"
EDGE_TYPE_AFTER = "AFTER"
EDGE_TYPE_LOCATED_AT = "LOCATED_AT"
EDGE_TYPE_CLAIMED = "CLAIMED"
EDGE_TYPE_OBSERVED = "OBSERVED"
EDGE_TYPE_OWNS = "OWNS"

# New Edge Types for richer relations
EDGE_TYPE_HAS_PARENT = "HAS_PARENT"
EDGE_TYPE_HAS_CHILD = "HAS_CHILD"
EDGE_TYPE_HAS_SIBLING = "HAS_SIBLING"
EDGE_TYPE_PARTNER_OF = "PARTNER_OF"
EDGE_TYPE_MARRIED_TO = "MARRIED_TO"
EDGE_TYPE_WORKS_AT = "WORKS_AT"
EDGE_TYPE_STUDIES_AT = "STUDIES_AT"
EDGE_TYPE_OWNS_BUSINESS = "OWNS_BUSINESS"
EDGE_TYPE_REPORTED_TO = "REPORTED_TO"
EDGE_TYPE_CONTACTED_ABOUT = "CONTACTED_ABOUT"
EDGE_TYPE_NOTIFIED_ABOUT = "NOTIFIED_ABOUT"
EDGE_TYPE_LIVES_AT = "LIVES_AT"
EDGE_TYPE_FREQUENTS = "FREQUENTS"
EDGE_TYPE_DISCOVERED = "DISCOVERED"
EDGE_TYPE_HEARD = "HEARD"
EDGE_TYPE_STATED = "STATED"
EDGE_TYPE_ASKED = "ASKED"
EDGE_TYPE_BELIEVED = "BELIEVED"
EDGE_TYPE_KNEW = "KNEW"
EDGE_TYPE_WITNESSED = "WITNESSED"
EDGE_TYPE_LOCATED = "LOCATED"
EDGE_TYPE_TOOK = "TOOK"
EDGE_TYPE_GAVE = "GAVE"
EDGE_TYPE_RECEIVED = "RECEIVED"
EDGE_TYPE_PROVIDED = "PROVIDED"
EDGE_TYPE_REQUESTED = "REQUESTED"
EDGE_TYPE_CONFIRMED = "CONFIRMED"
EDGE_TYPE_DETERMINED = "DETERMINED"
EDGE_TYPE_INVESTIGATED = "INVESTIGATED"
EDGE_TYPE_DOCUMENTED = "DOCUMENTED"
EDGE_TYPE_COLLECTED = "COLLECTED"
EDGE_TYPE_SEIZED = "SEIZED"
EDGE_TYPE_TRANSPORTED = "TRANSPORTED"
EDGE_TYPE_PROCESSED = "PROCESSED"
EDGE_TYPE_REVIEWED = "REVIEWED"
EDGE_TYPE_NOTED = "NOTED"
EDGE_TYPE_DESCRIBED = "DESCRIBED"
EDGE_TYPE_MENTIONED = "MENTIONED"


# Mapping from high-level relation_type (produced by the case-specific
# relation extractor) to concrete edge_type values in the KG.
RELATION_TYPE_TO_EDGE = {
    # Direct 1:1 mappings for incident-process actions
    "ARRIVED_AT": "ARRIVED_AT",
    "SECURED": "SECURED",
    "INTERVIEWED": "INTERVIEWED",
    "EXAMINED": "EXAMINED",
    "RESPONDED": "RESPONDED",
    "DISPATCHED": "DISPATCHED",
    "PRONOUNCED": "PRONOUNCED",
    "MET": "MET",
    "DIRECTED": "DIRECTED",
    "IDENTIFIED": "IDENTIFIED",
    # Narrative / evidentiary statements
    "CLAIMED": EDGE_TYPE_CLAIMED,
    "OBSERVED": EDGE_TYPE_OBSERVED,
    # REMOVED is modeled as an observation about evidence/scene state
    "REMOVED": EDGE_TYPE_OBSERVED,
    # New relations
    "HAS_PARENT": EDGE_TYPE_HAS_PARENT,
    "HAS_CHILD": EDGE_TYPE_HAS_CHILD,
    "HAS_SIBLING": EDGE_TYPE_HAS_SIBLING,
    "PARTNER_OF": EDGE_TYPE_PARTNER_OF,
    "MARRIED_TO": EDGE_TYPE_MARRIED_TO,
    "WORKS_AT": EDGE_TYPE_WORKS_AT,
    "STUDIES_AT": EDGE_TYPE_STUDIES_AT,
    "OWNS_BUSINESS": EDGE_TYPE_OWNS_BUSINESS,
    "REPORTED_TO": EDGE_TYPE_REPORTED_TO,
    "CONTACTED_ABOUT": EDGE_TYPE_CONTACTED_ABOUT,
    "NOTIFIED_ABOUT": EDGE_TYPE_NOTIFIED_ABOUT,
    "LIVES_AT": EDGE_TYPE_LIVES_AT,
    "FREQUENTS": EDGE_TYPE_FREQUENTS,
    "DISCOVERED": EDGE_TYPE_DISCOVERED,
    "HEARD": EDGE_TYPE_HEARD,
    "STATED": EDGE_TYPE_STATED,
    "ASKED": EDGE_TYPE_ASKED,
    "BELIEVED": EDGE_TYPE_BELIEVED,
    "KNEW": EDGE_TYPE_KNEW,
    "WITNESSED": EDGE_TYPE_WITNESSED,
    "LOCATED": EDGE_TYPE_LOCATED,
    "TOOK": EDGE_TYPE_TOOK,
    "GAVE": EDGE_TYPE_GAVE,
    "RECEIVED": EDGE_TYPE_RECEIVED,
    "PROVIDED": EDGE_TYPE_PROVIDED,
    "REQUESTED": EDGE_TYPE_REQUESTED,
    "CONFIRMED": EDGE_TYPE_CONFIRMED,
    "DETERMINED": EDGE_TYPE_DETERMINED,
    "INVESTIGATED": EDGE_TYPE_INVESTIGATED,
    "DOCUMENTED": EDGE_TYPE_DOCUMENTED,
    "COLLECTED": EDGE_TYPE_COLLECTED,
    "SEIZED": EDGE_TYPE_SEIZED,
    "TRANSPORTED": EDGE_TYPE_TRANSPORTED,
    "PROCESSED": EDGE_TYPE_PROCESSED,
    "REVIEWED": EDGE_TYPE_REVIEWED,
    "NOTED": EDGE_TYPE_NOTED,
    "DESCRIBED": EDGE_TYPE_DESCRIBED,
    "MENTIONED": EDGE_TYPE_MENTIONED,
}


def _normalize_name(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _make_entity_id(norm: str) -> str:
    return f"ENT_{norm.replace(' ', '_')}"


def _make_time_id(value: str) -> str:
    return f"TIME_{value.replace(':', '').replace('-', '').replace('T', '')}"


def _make_location_id(norm: str) -> str:
    return f"LOC_{norm.replace(' ', '_')}"


def _build_mention_to_canonical(case_docs: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map every known mention text to its canonical name, using each document's
    coref_entities (produced by GlobalEntityPool during coreference resolution).

    This lets node identity be driven by the case's actual coreference/mention
    unification output instead of a hardcoded, case-specific list of names.
    """
    mention_to_canonical: Dict[str, str] = {}
    for entry in case_docs:
        processed_doc = entry.get("processed_doc") or {}
        for cluster in processed_doc.get("coref_entities") or []:
            canonical = cluster.get("canonical_name")
            if not canonical:
                continue
            canonical_norm = _normalize_name(canonical.replace("_", " "))
            for m in cluster.get("mentions") or []:
                mention_text = m if isinstance(m, str) else (m.get("text") if isinstance(m, dict) else "")
                if mention_text:
                    mention_to_canonical[_normalize_name(mention_text)] = canonical_norm
    return mention_to_canonical


def _canonical_id(norm: str, mention_to_canonical: Dict[str, str]) -> str:
    """Resolve a normalized mention to its canonical entity id, falling back to
    the mention itself when no coref cluster covers it."""
    canonical_norm = mention_to_canonical.get(norm, norm)
    return _make_entity_id(canonical_norm)


def _build_mention_lookup(
    entity_nodes: Dict[str, Dict[str, Any]],
    mention_to_canonical: Dict[str, str],
) -> List[Tuple[str, str]]:
    """List of (mention_norm, canonical_norm), longest mention first.

    A relation's subject/object is often a full dependency-parse subtree
    ("Cheryl Weston regarding the whereabouts of Ms. Pace"), not a clean
    entity string, so resolving it requires checking whether a known mention
    is *contained in* the text -- not that the text *equals* a known mention.
    Longest-first ensures "cheryl weston" matches before a shorter, looser
    substring like "cheryl" would.
    """
    lookup: Dict[str, str] = dict(mention_to_canonical)
    for node in entity_nodes.values():
        norm = node.get("norm")
        if norm and norm not in lookup:
            lookup[norm] = norm
    return sorted(lookup.items(), key=lambda kv: len(kv[0]), reverse=True)


def _resolve_entity_mention(text_norm: str, mention_lookup: List[Tuple[str, str]]) -> Optional[str]:
    """Return the canonical norm of the longest known entity mention contained
    in text_norm, or None if text_norm doesn't reference any known entity."""
    for mention_norm, canonical_norm in mention_lookup:
        if mention_norm and mention_norm in text_norm:
            return canonical_norm
    return None


def _resolve_location_alias(text_norm: str, canonical_locations: Dict[str, str]) -> Optional[str]:
    """Return the location id for the longest known location alias contained
    in text_norm, or None."""
    best_alias: Optional[str] = None
    best_id: Optional[str] = None
    for alias, loc_id in canonical_locations.items():
        if alias in text_norm and (best_alias is None or len(alias) > len(best_alias)):
            best_alias, best_id = alias, loc_id
    return best_id


def build_nodes_and_edges_for_case(
    case_id: str,
    case_docs: List[Dict[str, Any]],
    timeline_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Build graph_nodes and graph_edges for a case.

    Args:
        case_id: Case identifier (e.g., "CASE123").
        case_docs: List of dicts of the form:
            {
              "doc_name": str,
              "doc_type": str,
              "processed_doc": { ... }
            }
        timeline_info: Dict as returned by timeline_builder.build_timeline_for_case,
            containing "timeline_edges": [ { source_event_id, target_event_id, relation="BEFORE", ... } ]

    Returns:
        {
          "case_id": ...,
          "nodes": [...],
          "edges": [...],
          "timing": { ... }  # Timing information
        }
    """
    kg_start_time = time.time()
    timing_info = {}

    mention_to_canonical = _build_mention_to_canonical(case_docs)

    # Maps for deduplication
    entity_nodes: Dict[str, Dict[str, Any]] = {}
    time_nodes: Dict[str, Dict[str, Any]] = {}
    location_nodes: Dict[str, Dict[str, Any]] = {}
    event_nodes: Dict[str, Dict[str, Any]] = {}

    edges: List[Dict[str, Any]] = []

    # Collect nodes and edges from each document
    for entry in case_docs:
        doc_name = entry.get("doc_name") or entry.get("name") or "unknown_doc"
        doc_type = entry.get("doc_type") or "unknown"
        processed_doc = entry.get("processed_doc") or {}

        entities = processed_doc.get("entities") or []
        events = processed_doc.get("events") or []

        # --- Entity nodes ---
        for ent in entities:
            text = ent.get("text")
            label = ent.get("label")
            if not text:
                continue
            norm = _normalize_name(ent.get("norm") or text)
            ent_id = _canonical_id(norm, mention_to_canonical)
            if ent_id not in entity_nodes:
                entity_nodes[ent_id] = {
                    "id": ent_id,
                    "type": NODE_TYPE_ENTITY,
                    "text": text,
                    "norm": norm,
                    "label": label,
                    "case_id": case_id,
                }

        # --- Event nodes + edges PARTICIPATED_IN / LOCATED_AT ---
        for ev in events:
            ev_id = ev.get("event_id")
            if not ev_id:
                # skip events without IDs; they should have been assigned earlier
                continue
            if ev_id not in event_nodes:
                event_nodes[ev_id] = {
                    "id": ev_id,
                    "type": NODE_TYPE_EVENT,
                    "description": ev.get("description"),
                    "subject": ev.get("subject"),
                    "predicate": ev.get("predicate"),
                    "object": ev.get("object"),
                    "doc_name": doc_name,
                    "doc_type": doc_type,
                    "case_id": case_id,
                    "time_raw": ev.get("time_raw"),
                    "time_resolved": ev.get("time_resolved"),
                }

            # PARTICIPATED_IN: link entity nodes to event
            participants: List[str] = ev.get("participants") or []
            for p in participants:
                norm_p = _normalize_name(p)
                ent_id = _canonical_id(norm_p, mention_to_canonical)
                if ent_id not in entity_nodes:
                    entity_nodes[ent_id] = {
                        "id": ent_id,
                        "type": NODE_TYPE_ENTITY,
                        "text": p,
                        "norm": norm_p,
                        "label": None,
                        "case_id": case_id,
                    }
                edges.append(
                    {
                        "source": ent_id,
                        "target": ev_id,
                        "edge_type": EDGE_TYPE_PARTICIPATED_IN,
                        "case_id": case_id,
                    }
                )

            # LOCATED_AT: link events to location nodes
            locations: List[str] = ev.get("locations") or []
            for loc in locations:
                norm_loc = _normalize_name(loc)
                loc_id = _make_location_id(norm_loc)
                if loc_id not in location_nodes:
                    location_nodes[loc_id] = {
                        "id": loc_id,
                        "type": NODE_TYPE_LOCATION,
                        "text": loc,
                        "norm": norm_loc,
                        "case_id": case_id,
                    }
                edges.append(
                    {
                        "source": ev_id,
                        "target": loc_id,
                        "edge_type": EDGE_TYPE_LOCATED_AT,
                        "case_id": case_id,
                    }
                )

            # TIME nodes at event-level
            time_iso = ev.get("time_resolved")
            if time_iso:
                time_id = _make_time_id(time_iso)
                if time_id not in time_nodes:
                    time_nodes[time_id] = {
                        "id": time_id,
                        "type": NODE_TYPE_TIME,
                        "value": time_iso,
                        "case_id": case_id,
                    }
                edges.append(
                    {
                        "source": ev_id,
                        "target": time_id,
                        "edge_type": "HAS_TIME",
                        "case_id": case_id,
                    }
                )

    # --- BEFORE / AFTER edges from timeline ---
    timeline_edges: List[Dict[str, Any]] = timeline_info.get("timeline_edges") or []
    for e in timeline_edges:
        src_ev = e.get("source_event_id")
        tgt_ev = e.get("target_event_id")
        if not src_ev or not tgt_ev:
            continue
        if src_ev in event_nodes and tgt_ev in event_nodes:
            edges.append(
                {
                    "source": src_ev,
                    "target": tgt_ev,
                    "edge_type": EDGE_TYPE_BEFORE,
                    "case_id": case_id,
                }
            )
            # Add explicit AFTER edge as inverse
            edges.append(
                {
                    "source": tgt_ev,
                    "target": src_ev,
                    "edge_type": EDGE_TYPE_AFTER,
                    "case_id": case_id,
                }
            )

    # --- CLAIMED / OBSERVED / OWNS edges from relations ---
    # We rely on refined relations attached to each processed_doc (if present).
    # relation_type is produced by the case-specific extractor.
    KEY_OBJECT_OWNERSHIP = {"dog", "thoreau", "briefcase", "handbag", "residence", "house"}
    CANONICAL_LOCATIONS = {
        "the scene": "LOC_home",
        "the residence": "LOC_home",
        "the house": "LOC_home",
        "the basement": "LOC_basement",
        "the bedroom": "LOC_bedroom",
        "the gallery": "LOC_gallery",
        "the kitchen": "LOC_kitchen",
        "the porch": "LOC_porch",
        "the stairs": "LOC_stairs",
    }

    # Only relations where BOTH sides reference a genuine, recognized entity
    # or location become graph nodes/edges. A dependency-parsed relation's
    # subject/object is often a full subtree phrase ("Cheryl Weston regarding
    # the whereabouts of Ms. Pace"), not a clean entity string -- so resolution
    # looks for a known mention *contained in* the text, not an exact match.
    # A phrase with no such mention at all (e.g. "an approximate 30 degree
    # angle to the body") is real narrative detail, but not a case actor, and
    # promoting it to its own node would flood the graph with one-off junk.
    # Such relations remain available in processed_doc["relations"]/events for
    # narrative detail; they just don't get graph nodes.
    mention_lookup = _build_mention_lookup(entity_nodes, mention_to_canonical)

    for entry in case_docs:
        doc_name = entry.get("doc_name") or entry.get("name") or "unknown_doc"
        doc_type = entry.get("doc_type") or "unknown"
        processed_doc = entry.get("processed_doc") or {}
        relations: List[Dict[str, Any]] = processed_doc.get("relations") or []

        for rel in relations:
            subj = (rel.get("subject") or "").strip()
            obj = (rel.get("object") or "").strip()
            if not subj or not obj:
                continue

            pred = (rel.get("predicate") or "").strip().lower()
            rel_type = rel.get("relation_type")

            subj_norm = _normalize_name(subj)
            obj_norm = _normalize_name(obj)

            # --- Subject: resolve to a location alias, else a known entity
            # mention contained in the phrase. Drop the relation if neither
            # side resolves -- it's not a grounded entity-to-entity fact. ---
            subj_loc_id = _resolve_location_alias(subj_norm, CANONICAL_LOCATIONS)
            subj_canonical = None if subj_loc_id else _resolve_entity_mention(subj_norm, mention_lookup)
            if not subj_loc_id and not subj_canonical:
                continue

            obj_loc_id = _resolve_location_alias(obj_norm, CANONICAL_LOCATIONS)
            obj_canonical = None if obj_loc_id else _resolve_entity_mention(obj_norm, mention_lookup)
            if not obj_loc_id and not obj_canonical:
                continue

            if subj_loc_id:
                subj_id = subj_loc_id
                if subj_id not in location_nodes:
                    location_nodes[subj_id] = {
                        "id": subj_id,
                        "type": NODE_TYPE_LOCATION,
                        "text": subj,
                        "norm": subj_norm,
                        "case_id": case_id,
                    }
            else:
                subj_id = _make_entity_id(subj_canonical)
                if subj_id not in entity_nodes:
                    entity_nodes[subj_id] = {
                        "id": subj_id,
                        "type": NODE_TYPE_ENTITY,
                        "text": subj,
                        "norm": subj_canonical,
                        "label": None,
                        "case_id": case_id,
                    }

            if obj_loc_id:
                obj_id = obj_loc_id
                if obj_id not in location_nodes:
                    location_nodes[obj_id] = {
                        "id": obj_id,
                        "type": NODE_TYPE_LOCATION,
                        "text": obj,
                        "norm": obj_norm,
                        "case_id": case_id,
                    }
            else:
                obj_id = _make_entity_id(obj_canonical)
                if obj_id not in entity_nodes:
                    entity_nodes[obj_id] = {
                        "id": obj_id,
                        "type": NODE_TYPE_ENTITY,
                        "text": obj,
                        "norm": obj_canonical,
                        "label": None,
                        "case_id": case_id,
                    }

            edge_type: str | None = None

            # Primary: map to a curated edge type when we have one; otherwise
            # fall back to the relation_type itself (e.g. an uppercased verb
            # lemma from the extractor) rather than dropping the edge.
            if rel_type:
                edge_type = RELATION_TYPE_TO_EDGE.get(rel_type, rel_type)

            # Secondary: simple ownership heuristic when there's no relation_type at all
            if not edge_type and any(k in obj_norm for k in KEY_OBJECT_OWNERSHIP):
                # subject OWNS object if object is in ownership set
                edge_type = EDGE_TYPE_OWNS

            if edge_type:
                edges.append(
                    {
                        "source": subj_id,
                        "target": obj_id,
                        "edge_type": edge_type,
                        "case_id": case_id,
                        "doc_name": doc_name,
                        "doc_type": doc_type,
                    }
                )

    # Assemble final list of nodes
    nodes: List[Dict[str, Any]] = []
    nodes.extend(entity_nodes.values())
    nodes.extend(event_nodes.values())
    nodes.extend(time_nodes.values())
    nodes.extend(location_nodes.values())

    # Calculate PageRank scores for entities
    pagerank_start = time.time()
    G = nx.DiGraph()
    for node in nodes:
        if node["type"] == NODE_TYPE_ENTITY:
            G.add_node(node["id"])

    for edge in edges:
        # Only consider edges between entities for PageRank calculation
        # Or, if you want to include events/locations, you'd add them as nodes too
        if edge["source"].startswith("ENT_") and edge["target"].startswith("ENT_"):
            G.add_edge(edge["source"], edge["target"])
        elif edge["source"].startswith("ENT_") and edge["target"].startswith("EV_"):
            G.add_edge(edge["source"], edge["target"])
        elif edge["source"].startswith("EV_") and edge["target"].startswith("LOC_"):
            G.add_edge(edge["source"], edge["target"])
        # Add other relevant edges for PageRank calculation

    if G.nodes(): # Ensure graph is not empty
        pagerank_scores = nx.pagerank(G)
        for node in nodes:
            if node["id"] in pagerank_scores:
                node["pagerank_score"] = pagerank_scores[node["id"]]

    timing_info["pagerank_calculation"] = time.time() - pagerank_start
    timing_info["total_kg_creation"] = time.time() - kg_start_time

    return {
        "case_id": case_id,
        "nodes": nodes,
        "edges": edges,
        "timing": timing_info,
    }


def save_graph_json(
    graph: Dict[str, Any],
    base_dir: str,
    nodes_filename: str = "graph_nodes.json",
    edges_filename: str = "graph_edges.json",
) -> Tuple[str, str]:
    """Save graph node and edge lists as separate JSON files.

    Returns:
        (nodes_path, edges_path)
    """
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = out_dir / nodes_filename
    edges_path = out_dir / edges_filename

    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    with nodes_path.open("w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=4, ensure_ascii=False)

    with edges_path.open("w", encoding="utf-8") as f:
        json.dump(edges, f, indent=4, ensure_ascii=False)

    return str(nodes_path), str(edges_path)