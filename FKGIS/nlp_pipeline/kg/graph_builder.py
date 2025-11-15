from __future__ import annotations

from typing import Dict, Any, List, Tuple, Set
from pathlib import Path
import json


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
}


def _normalize_name(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _make_entity_id(norm: str) -> str:
    return f"ENT_{norm.replace(' ', '_')}"


def _make_time_id(value: str) -> str:
    return f"TIME_{value.replace(':', '').replace('-', '').replace('T', '')}"


def _make_location_id(norm: str) -> str:
    return f"LOC_{norm.replace(' ', '_')}"


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
          "edges": [...]
        }
    """
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
            ent_id = _make_entity_id(norm)
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
                ent_id = _make_entity_id(norm_p)
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

            subj_id = _make_entity_id(subj_norm)
            if subj_id not in entity_nodes:
                entity_nodes[subj_id] = {
                    "id": subj_id,
                    "type": NODE_TYPE_ENTITY,
                    "text": subj,
                    "norm": subj_norm,
                    "label": None,
                    "case_id": case_id,
                }

            obj_id = _make_entity_id(obj_norm)
            if obj_id not in entity_nodes:
                entity_nodes[obj_id] = {
                    "id": obj_id,
                    "type": NODE_TYPE_ENTITY,
                    "text": obj,
                    "norm": obj_norm,
                    "label": None,
                    "case_id": case_id,
                }

            edge_type: str | None = None

            # Primary: use explicit relation_type from the case-specific extractor
            if rel_type:
                edge_type = RELATION_TYPE_TO_EDGE.get(rel_type)

            # Secondary: simple ownership heuristic when no explicit mapping
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

    return {
        "case_id": case_id,
        "nodes": nodes,
        "edges": edges,
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