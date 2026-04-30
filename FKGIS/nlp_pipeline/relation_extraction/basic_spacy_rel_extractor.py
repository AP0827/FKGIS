from __future__ import annotations

from typing import Dict, Any, List
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.matcher import Matcher

from .basic_rel_extractor import extract_relations as _base_extract_relations


# Case-specific verb lexicon (lemmas) for high-precision relations in the
# Kimberly Pace incident:
#
#   arrived, secured, observed, removed, pronounced, responded, identified,
#   met, directed, interviewed, examined, reported, said, indicated, called,
#   dispatched, found, saw, heard, told, asked, stated, believed, knew,
#   witnessed, located, took, gave, received, provided, requested, confirmed,
#   determined, investigated, documented, collected, seized, transported,
#   processed, reviewed, noted, discovered, described, mentioned, claimed,
#   reported, observed, found, saw, heard, told, asked, stated, believed, knew,
#   witnessed, located, took, gave, received, provided, requested, confirmed,
#   determined, investigated, documented, collected, seized, transported,
#   processed, reviewed, noted, discovered, described, mentioned, born, sister,
#   married, boyfriend, work, teach, graduate, own, live, socialize, notify,
#   contact
#
CASE_VERB_LEXICON = {
    "arrive", "secure", "observe", "remove", "pronounce", "respond", "identify",
    "meet", "direct", "interview", "examine", "report", "say", "indicate",
    "call", "dispatch", "find", "see", "hear", "tell", "ask", "state",
    "believe", "know", "witness", "locate", "take", "give", "receive",
    "provide", "request", "confirm", "determine", "investigate", "document",
    "collect", "seize", "transport", "process", "review", "note", "discover",
    "describe", "mention", "claim",
    "bear", # for "born to"
    "sister", # for "is my sister"
    "marry", # for "married to"
    "boyfriend", # for "is boyfriend of"
    "work", # for "works at"
    "teach", # for "teaches at"
    "graduate", # for "graduated from"
    "own", # for "owns" (business)
    "live", # for "lives at"
    "socialize", # for "frequents"
    "notify", # for "notified"
    "contact", # for "contacted"
}


VERB_TO_RELATION_TYPE = {
    "arrive": "ARRIVED_AT", "secure": "SECURED", "observe": "OBSERVED",
    "remove": "REMOVED", "pronounce": "PRONOUNCED", "respond": "RESPONDED",
    "identify": "IDENTIFIED", "meet": "MET", "direct": "DIRECTED",
    "interview": "INTERVIEWED", "examine": "EXAMINED", "report": "CLAIMED",
    "say": "CLAIMED", "indicate": "CLAIMED", "call": "DISPATCHED",
    "dispatch": "DISPATCHED", "find": "DISCOVERED", "see": "OBSERVED",
    "hear": "HEARD", "tell": "STATED", "ask": "ASKED", "state": "STATED",
    "believe": "BELIEVED", "know": "KNEW", "witness": "WITNESSED",
    "locate": "LOCATED", "take": "TOOK", "give": "GAVE", "receive": "RECEIVED",
    "provide": "PROVIDED", "request": "REQUESTED", "confirm": "CONFIRMED",
    "determine": "DETERMINED", "investigate": "INVESTIGATED",
    "document": "DOCUMENTED", "collect": "COLLECTED", "seize": "SEIZED",
    "transport": "TRANSPORTED", "process": "PROCESSED", "review": "REVIEWED",
    "note": "NOTED", "discover": "DISCOVERED", "describe": "DESCRIBED",
    "mention": "MENTIONED", "claim": "CLAIMED",
    "bear": "HAS_PARENT", # "was born to"
    "sister": "HAS_SIBLING", # "is my sister"
    "marry": "MARRIED_TO", # "married to"
    "boyfriend": "PARTNER_OF", # "is boyfriend of"
    "work": "WORKS_AT", # "works at"
    "teach": "WORKS_AT", # "teaches at"
    "graduate": "STUDIES_AT", # "graduated from"
    "own": "OWNS_BUSINESS", # "owns" (business)
    "live": "LIVES_AT", # "lives at"
    "socialize": "FREQUENTS", # "socializing with"
    "notify": "NOTIFIED_ABOUT", # "was notified"
    "contact": "CONTACTED_ABOUT", # "contacted"
}


def _normalize_predicate(pred: str) -> str:
    """Normalize predicate text to a lemma-like form for lexicon matching.

    The base extractor already uses verb.lemma_ when available, but we
    defensively lowercase and strip here.
    """
    return (pred or "").strip().lower()


def _get_entity_text(token_or_span: Token | Span) -> str:
    """Helper to get text of a token or span, prioritizing entity text if available."""
    # If it's a Span, and it's an entity, use its text. Otherwise, just use the text.
    if isinstance(token_or_span, Span):
        if token_or_span.label_: # Use label_ for Span entity type
            return token_or_span.text
        return token_or_span.text
    # If it's a Token, and it's part of an entity, use the entity text. Otherwise, just use the token text.
    elif isinstance(token_or_span, Token):
        if token_or_span.ent_type_: # Use ent_type_ for Token entity type
            return token_or_span.ent_kb_id_ if token_or_span.ent_kb_id_ else token_or_span.text
        return token_or_span.text
    return str(token_or_span)


def _extract_relations_with_matcher(doc: Doc, nlp: Language) -> List[Dict[str, Any]]:
    """Extract relations using spaCy's Matcher for pattern-based extraction."""
    matcher = Matcher(nlp.vocab)
    extracted_relations: List[Dict[str, Any]] = []

    # --- HAS_PARENT patterns ---
    # Pattern 1: [PERSON] was born to [PERSON1] and [PERSON2]
    matcher.add("HAS_PARENT_PATTERN1", [[
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}}, # Subject (e.g., Kimberly)
        {"LEMMA": "be"},
        {"LEMMA": "bear"}, # predicate "born"
        {"LOWER": "to"},
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}, "OP": "+"}, # Object (e.g., Valerie and Robert Pace)
    ]])

    # Pattern 2: [PERSON]'s parents were [PERSON1] and [PERSON2]
    matcher.add("HAS_PARENT_PATTERN2", [[
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}}, # Subject (e.g., Kimberly)
        {"POS": "PART", "LOWER": "'s"},
        {"LOWER": "parent", "OP": "+"}, # parents
        {"LEMMA": "be"},
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}, "OP": "+"}, # Object (e.g., Valerie and Robert Pace)
    ]])

    # --- HAS_SIBLING patterns ---
    # Pattern 1: [PERSON1] and [PERSON2] got along well
    matcher.add("HAS_SIBLING_PATTERN1", [[
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}}, # Subject (e.g., Kimberly)
        {"LOWER": "and"},
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}}, # Object (e.g., Becky)
        {"POS": "VERB", "LEMMA": "get"},
        {"LOWER": "along"},
        {"LOWER": "well"},
    ]])

    # Pattern 2: [PERSON1] and [PERSON2] are sisters
    matcher.add("HAS_SIBLING_PATTERN2_ALT", [[
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}},
        {"LOWER": "and"},
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}},
        {"LEMMA": "be"},
        {"LOWER": "sister", "OP": "+"},
    ]])

    # --- INTERVIEWED patterns ---
    # Pattern 1: [PERSON/ROLE] interviewed [PERSON/ROLE]
    matcher.add("INTERVIEWED_PATTERN1", [[
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}}, # Interviewer (e.g., Detectives Armstrong and Murphy)
        {"LEMMA": "interview"}, # predicate
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}, "OP": "+"}, # Interviewee (e.g., her, Cheryl Weston)
    ]])

    # --- ASKED patterns ---
    # Pattern 1: [PERSON/ROLE] asked [PERSON/ROLE]
    matcher.add("ASKED_PATTERN1", [[
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}}, # Asker
        {"LEMMA": "ask"}, # predicate
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}, "OP": "?"}, # Asked person (optional)
    ]])

    # --- STATED/CLAIMED/REPORTED patterns ---
    # Pattern 1: [PERSON/ROLE] said/stated/reported/claimed
    matcher.add("STATED_PATTERN1", [[
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}}, # Speaker
        {"LEMMA": {"IN": ["say", "state", "report", "claim"]}}, # predicate
    ]])

    # --- TOLD patterns ---
    # Pattern 1: [PERSON/ROLE] told [PERSON/ROLE]
    matcher.add("TOLD_PATTERN1", [[
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}}, # Teller
        {"LEMMA": "tell"}, # predicate
        {"ENT_TYPE": {"IN": ["PERSON", "ROLE"]}}, # Told person
    ]])


    matches = matcher(doc)

    for match_id, start, end in matches:
        span = doc[start:end]
        rule_id = nlp.vocab.strings[match_id]

        # Extract subject, predicate, object based on rule_id
        subj_text = ""
        obj_text = ""
        pred_text = ""
        relation_type = ""

        if rule_id == "HAS_PARENT_PATTERN1":
            # Example: "Kimberly was born to Valerie and Robert Pace"
            # Subject: first PERSON entity
            # Predicate: "born to"
            # Object: subsequent PERSON entities
            subj_token = [token for token in span if token.ent_type_ in ["PERSON", "ROLE"]][0]
            obj_tokens = [token for token in span if token.ent_type_ in ["PERSON", "ROLE"]][1:]
            pred_token = [token for token in span if token.lemma_ == "bear"][0]

            subj_text = _get_entity_text(subj_token)
            obj_text = " and ".join([_get_entity_text(t) for t in obj_tokens])
            pred_text = pred_token.text
            relation_type = "HAS_PARENT"

            # Also add inverse HAS_CHILD relations
            for o_token in obj_tokens:
                extracted_relations.append({
                    "subject": _get_entity_text(o_token),
                    "predicate": "has child",
                    "object": subj_text,
                    "relation_type": "HAS_CHILD",
                })

        elif rule_id == "HAS_PARENT_PATTERN2":
            # Example: "Kimberly's parents were Valerie and Robert Pace"
            # Subject: first PERSON entity
            # Predicate: "parents were"
            # Object: subsequent PERSON entities
            subj_token = [token for token in span if token.ent_type_ in ["PERSON", "ROLE"]][0]
            obj_tokens = [token for token in span if token.ent_type_ in ["PERSON", "ROLE"]][1:]
            pred_text = "parents were" # Simplified predicate for this pattern

            subj_text = _get_entity_text(subj_token)
            obj_text = " and ".join([_get_entity_text(t) for t in obj_tokens])
            relation_type = "HAS_PARENT"

            # Also add inverse HAS_CHILD relations
            for o_token in obj_tokens:
                extracted_relations.append({
                    "subject": _get_entity_text(o_token),
                    "predicate": "has child",
                    "object": subj_text,
                    "relation_type": "HAS_CHILD",
                })

        elif rule_id == "HAS_SIBLING_PATTERN1":
            # Example: "Kimberly and Becky got along well"
            # Subject: first PERSON entity
            # Object: second PERSON entity
            # Predicate: "got along well"
            person_tokens = [token for token in span if token.ent_type_ in ["PERSON", "ROLE"]]
            if len(person_tokens) >= 2:
                subj_text = _get_entity_text(person_tokens[0])
                obj_text = _get_entity_text(person_tokens[1])
                pred_text = "got along well" # Simplified predicate
                relation_type = "HAS_SIBLING"
                # Add inverse
                extracted_relations.append({
                    "subject": obj_text,
                    "predicate": "has sibling",
                    "object": subj_text,
                    "relation_type": "HAS_SIBLING",
                })

        elif rule_id == "HAS_SIBLING_PATTERN2_ALT":
            # Example: "X and Y are sisters"
            person_tokens = [token for token in span if token.ent_type_ in ["PERSON", "ROLE"]]
            if len(person_tokens) >= 2:
                subj_text = _get_entity_text(person_tokens[0])
                obj_text = _get_entity_text(person_tokens[1])
                pred_text = "are sisters"
                relation_type = "HAS_SIBLING"
                # Add inverse
                extracted_relations.append({
                    "subject": obj_text,
                    "predicate": "has sibling",
                    "object": subj_text,
                    "relation_type": "HAS_SIBLING",
                })
        
        elif rule_id == "INTERVIEWED_PATTERN1":
            # Example: "Detectives Armstrong and Murphy interviewed her"
            # Subject: Interviewer(s)
            # Predicate: "interviewed"
            # Object: Interviewee(s)
            interviewer_tokens = [token for token in span if token.ent_type_ in ["PERSON", "ROLE"] and token.dep_ == "nsubj"]
            interviewee_tokens = [token for token in span if token.ent_type_ in ["PERSON", "ROLE"] and token.dep_ == "dobj"]
            pred_token = [token for token in span if token.lemma_ == "interview"][0]

            if interviewer_tokens and interviewee_tokens:
                subj_text = " and ".join([_get_entity_text(t) for t in interviewer_tokens])
                obj_text = " and ".join([_get_entity_text(t) for t in interviewee_tokens])
                pred_text = pred_token.text
                relation_type = "INTERVIEWED"

        elif rule_id == "ASKED_PATTERN1":
            # Example: "Detective Murphy: For the record, could you please state your name and address?"
            # Subject: Asker
            # Predicate: "asked"
            # Object: Question (simplified to just the text for now)
            asker_tokens = [token for token in span if token.ent_type_ in ["PERSON", "ROLE"] and token.dep_ == "nsubj"]
            pred_token = [token for token in span if token.lemma_ == "ask"][0]

            if asker_tokens:
                subj_text = " and ".join([_get_entity_text(t) for t in asker_tokens])
                obj_text = span.text # The entire question as the object
                pred_text = pred_token.text
                relation_type = "ASKED"

        elif rule_id == "STATED_PATTERN1":
            # Example: "Cheryl Weston: My name is Cheryl Weston."
            # Subject: Speaker
            # Predicate: "stated"
            # Object: Statement (simplified to just the text for now)
            speaker_tokens = [token for token in span if token.ent_type_ in ["PERSON", "ROLE"] and token.dep_ == "nsubj"]
            pred_token = [token for token in span if token.lemma_ in ["say", "state", "report", "claim"]][0]

            if speaker_tokens:
                subj_text = " and ".join([_get_entity_text(t) for t in speaker_tokens])
                obj_text = span.text # The entire statement as the object
                pred_text = pred_token.text
                relation_type = "STATED" # Can be refined to CLAIMED/REPORTED based on lemma

        elif rule_id == "TOLD_PATTERN1":
            # Example: "Jeremy Gladwell told me that Thoreau seemed a little low energy"
            # Subject: Teller
            # Predicate: "told"
            # Object: Told person
            teller_tokens = [token for token in span if token.ent_type_ in ["PERSON", "ROLE"] and token.dep_ == "nsubj"]
            told_person_tokens = [token for token in span if token.ent_type_ in ["PERSON", "ROLE"] and token.dep_ == "dobj"]
            pred_token = [token for token in span if token.lemma_ == "tell"][0]

            if teller_tokens and told_person_tokens:
                subj_text = " and ".join([_get_entity_text(t) for t in teller_tokens])
                obj_text = " and ".join([_get_entity_text(t) for t in told_person_tokens])
                pred_text = pred_token.text
                relation_type = "TOLD"


        if relation_type and subj_text and obj_text:
            extracted_relations.append({
                "subject": subj_text,
                "predicate": pred_text, # Use the extracted predicate text
                "object": obj_text,
                "relation_type": relation_type,
            })

    return extracted_relations


def refine_case_relations(processed_doc: Dict[str, Any], nlp: Language) -> Dict[str, Any]:
    """Refine relations for the Kimberly Pace case using a verb lexicon and spaCy Matcher patterns.

    Steps:
      1) Call the generic basic_rel_extractor.extract_relations() to populate
         processed_doc["relations"] with high-recall SPO triples.
      2) Extract additional relations using spaCy Matcher patterns.
      3) Filter relations to keep only those whose predicate lemma is in the
         case-specific verb lexicon, or whose subject/object clearly involve
         core case actors (officers, witnesses, victim, dog, etc.).
      4) For retained relations, attach a "relation_type" field using
         VERB_TO_RELATION_TYPE, when known.

    This function overwrites processed_doc["relations"] with the refined list.
    """
    # Create the spaCy Doc object locally within this function
    doc = nlp(processed_doc["raw_text"])

    # 1) Get base relations
    processed_doc = _base_extract_relations(processed_doc)
    raw_relations: List[Dict[str, Any]] = processed_doc.get("relations") or []

    # 2) Extract relations using Matcher
    matcher_relations = _extract_relations_with_matcher(doc, nlp)
    raw_relations.extend(matcher_relations)

    refined: List[Dict[str, Any]] = []

    # Key case actors / objects for backstop when predicate isn't in lexicon
    KEY_ACTORS = {
        "willits", "harding", "armstrong", "murphy", "johnson", "lukens",
        "sanchez", "rebecca pace", "becky pace", "cheryl weston",
        "jeremy gladwell", "kimberly pace", "kim", "thoreau", "dog", "body",
        "victim", "reporting officer", "reporting investigator", "deputy",
        "animal control", "crime scene unit", "coroner", "dispatch",
        "valerie", "robert", "paul evans", "miguel ochoa",
        "university of mississippi", "yokanapatawpha high school",
        "university elementary school", "the lucky café", "lane properties",
        "the library", "the downtown grill", "ajax diner", "c'est belle gallery",
    }

    for rel in raw_relations:
        subj = (rel.get("subject") or "").strip()
        obj = (rel.get("object") or "").strip()
        pred = (rel.get("predicate") or "").strip()

        if not subj or not obj or not pred:
            continue

        pred_norm = _normalize_predicate(pred)

        keep = False

        # 1) Primary: verb in case lexicon or relation_type already assigned by matcher
        if pred_norm in CASE_VERB_LEXICON or "relation_type" in rel:
            keep = True
        else:
            # 2) Secondary: subject or object mention key case actors / objects
            s_low = subj.lower()
            o_low = obj.lower()
            if any(name in s_low for name in KEY_ACTORS) or any(
                name in o_low for name in KEY_ACTORS
            ):
                keep = True

        if not keep:
            continue

        rel_out = dict(rel)
        if "relation_type" not in rel_out: # Only assign if not already set by matcher
            rel_out["predicate"] = pred_norm  # store normalized predicate
            rel_type = VERB_TO_RELATION_TYPE.get(pred_norm)
            if rel_type:
                rel_out["relation_type"] = rel_type

        if "relation_type" in rel_out: # Only add if a relation_type is assigned
            refined.append(rel_out)

    # Remove the temporary spacy_doc before returning (no longer needed as it's local)
    # if "spacy_doc" in processed_doc:
    #     del processed_doc["spacy_doc"]

    processed_doc["relations"] = refined
    return processed_doc