import json
import time
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

CHUNK_SIZE = 100
# API key and model are read from the environment so secrets are never
# committed to the repository. Provide GEMINI_API_KEY and optionally
# override the model with GEMINI_MODEL (defaults to the model used by the
# current implementation, gemini-2.0-flash).
GENAI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GENAI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_WORKERS = 3

_client = None


def get_client():
    """Return a lazily created Gemini client, raising if no API key is set."""
    global _client
    if not GENAI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in the environment (or .env) to "
            "enable LLM refinement."
        )
    if _client is None:
        from google import genai  # imported lazily so the app works without the LLM SDK

        _client = genai.Client(api_key=GENAI_API_KEY)
    return _client

def chunk_list(lst, chunk_size=100):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def load_json(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def normalize_text(text):
    if not text or not isinstance(text, str):
        return "unknown"
    return text.lower().replace(" ", "_")

def validate_node(node):
    if not isinstance(node, dict):
        return None
    if "id" not in node:
        return None
    if "text" not in node or not node["text"]:
        node["text"] = node.get("id", "unknown").replace("ENT_", "").replace("_", " ")
    if "label" not in node:
        node["label"] = "OTHER"
    if "norm" not in node or not node["norm"]:
        node["norm"] = normalize_text(node["text"])
    node["text"] = str(node["text"]).strip()
    return node

def validate_edge(edge):
    if not isinstance(edge, dict):
        return None
    required_fields = ["source", "target", "edge_type"]
    for field in required_fields:
        if field not in edge:
            return None
    return edge

def extract_json_from_text(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    json_match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    return None

def call_gemini(prompt):
    for attempt in range(MAX_RETRIES):
        try:
            response = get_client().models.generate_content(
                model=GENAI_MODEL,
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    raise RuntimeError("Failed to get a valid response from Gemini after multiple retries.")

def process_node_chunk(chunk_data):
    """Process a single chunk of nodes with FIXED prompting"""
    chunk_idx, nodes_chunk = chunk_data
    
    # Validate and pre-process chunk
    validated_chunk = []
    for node in nodes_chunk:
        validated_node = validate_node(node)
        if validated_node:
            validated_chunk.append(validated_node)
    
    if not validated_chunk:
        return []
    
    # FIXED: Use properly escaped JSON examples
    prompt = f"""You are an expert knowledge graph engineer specializing in entity resolution and data quality.

## YOUR TASK
Refine the following {len(validated_chunk)} knowledge graph nodes to improve quality and consistency.

## INPUT DATA
{json.dumps(validated_chunk, indent=2)}

## REFINEMENT RULES

### 1. Entity Type Classification
Correct any misclassified 'label' fields using these standard types:
- PERSON: Individual humans
- ORGANIZATION: Companies, institutions, agencies  
- LOCATION: Geographic places
- DATE: Temporal references
- EVENT: Named occurrences
- CONCEPT: Abstract ideas
- PRODUCT: Goods or services
- OTHER: Anything that doesn't fit above categories

### 2. Within-Chunk Duplicate Merging
Merge nodes that refer to the SAME real-world entity:
- "Apple Inc.", "Apple Corporation", "Apple" (in business context) -> merge to one node
- "Dr. John Smith", "John Smith", "J. Smith" -> merge if clearly same person

### 3. Text Normalization
Ensure 'norm' field is correctly normalized:
- Lowercase all characters
- Replace spaces with underscores
- Remove special characters except underscores

### 4. Data Preservation
- Keep ALL original fields: 'id', 'type', 'case_id', etc.
- Never delete fields, even if they seem unused

## OUTPUT FORMAT
Return ONLY a valid JSON array with this structure:
[
  {{
    "id": "original_or_primary_id",
    "text": "cleaned entity text", 
    "label": "ENTITY_TYPE",
    "norm": "normalized_text",
    "merged_ids": ["id1", "id2"],
    "refinement_notes": "Brief note about changes",
    ... (all other original fields)
  }}
]

## IMPORTANT CONSTRAINTS
- Output MUST be valid JSON (no markdown, no explanations)
- Do NOT merge entities across different case_ids
- When uncertain about merging, keep entities separate
- Preserve all original data

Begin refinement:
"""
    
    try:
        response_text = call_gemini(prompt)
        
        # Try direct parsing first
        try:
            refined_nodes = json.loads(response_text)
        except json.JSONDecodeError:
            # Extract from markdown
            refined_nodes = extract_json_from_text(response_text)
            if refined_nodes is None:
                print(f"[Warning] Failed to parse JSON for chunk {chunk_idx + 1}, using original")
                refined_nodes = validated_chunk
        
        # Validate refined nodes
        validated_refined = []
        for node in refined_nodes:
            valid_node = validate_node(node)
            if valid_node:
                validated_refined.append(valid_node)
        
        print(f"Completed node chunk {chunk_idx + 1}: {len(validated_chunk)} -> {len(validated_refined)} nodes")
        return validated_refined
        
    except Exception as e:
        print(f"[Error] Chunk {chunk_idx + 1} failed: {e}")
        return validated_chunk

def process_edge_chunk(chunk_data):
    """Process a single chunk of edges with FIXED prompting"""
    chunk_idx, edges_chunk = chunk_data
    
    # Remove duplicates within chunk and validate
    seen = set()
    validated_chunk = []
    for edge in edges_chunk:
        validated_edge = validate_edge(edge)
        if validated_edge:
            key = (validated_edge["source"], validated_edge["target"], validated_edge["edge_type"])
            if key not in seen:
                seen.add(key)
                validated_chunk.append(validated_edge)
    
    if not validated_chunk:
        return []
    
    # FIXED: Use properly escaped JSON examples
    prompt = f"""You are an expert knowledge graph engineer specializing in relationship validation and semantic analysis.

## YOUR TASK
Refine the following {len(validated_chunk)} knowledge graph edges (relationships) to ensure quality and consistency.

## INPUT EDGES
{json.dumps(validated_chunk, indent=2)}

## REFINEMENT RULES

### 1. Duplicate Edge Removal
Remove exact duplicates where source, target, and edge_type are identical

### 2. Relationship Type Standardization
Ensure 'edge_type' uses consistent, clear naming:

Common relationship types:
- WORKS_FOR: person -> organization employment
- LOCATED_IN: entity -> location  
- PART_OF: entity -> parent entity
- ASSOCIATED_WITH: general relationship
- OCCURRED_ON: event -> date
- RELATED_TO: generic connection

### 3. Semantic Validation
Flag or remove edges that don't make logical sense

## OUTPUT FORMAT
Return ONLY a valid JSON array:
[
  {{
    "source": "entity_id_1",
    "target": "entity_id_2", 
    "edge_type": "STANDARDIZED_TYPE",
    "refinement_notes": "what changed (optional)",
    ... (all other original fields)
  }}
]

## IMPORTANT CONSTRAINTS
- Output MUST be valid JSON (no markdown, no explanations)
- Do NOT remove edges unless they're exact duplicates
- When uncertain about semantics, keep the edge
- Preserve all original metadata fields

Begin refinement:
"""
    
    try:
        response_text = call_gemini(prompt)
        
        try:
            refined_edges = json.loads(response_text)
        except json.JSONDecodeError:
            refined_edges = extract_json_from_text(response_text)
            if refined_edges is None:
                print(f"[Warning] Failed to parse JSON for edge chunk {chunk_idx + 1}, using original")
                refined_edges = validated_chunk
        
        print(f"Completed edge chunk {chunk_idx + 1}: {len(validated_chunk)} -> {len(refined_edges)} edges")
        return refined_edges
        
    except Exception as e:
        print(f"[Error] Edge chunk {chunk_idx + 1} failed: {e}")
        return validated_chunk

def refine_nodes_parallel(nodes_file, output_file, use_llm=False):
    nodes = load_json(nodes_file)
    print(f"Loaded {len(nodes)} nodes from {nodes_file}")
    
    # First, validate all nodes
    validated_nodes = []
    for i, node in enumerate(nodes):
        validated_node = validate_node(node)
        if validated_node:
            validated_nodes.append(validated_node)
        else:
            print(f"[Warning] Skipped invalid node at index {i}: {node}")
    
    print(f"After validation: {len(validated_nodes)} nodes")
    
    if not use_llm:
        seen_ids = set()
        refined_nodes = []
        for node in validated_nodes:
            if node["id"] not in seen_ids:
                seen_ids.add(node["id"])
                refined_nodes.append(node)
        save_json(refined_nodes, output_file)
        print(f"Processed {len(refined_nodes)} nodes (no LLM)")
        return

    chunks = list(enumerate(chunk_list(validated_nodes, CHUNK_SIZE)))
    refined_nodes = []
    
    print(f"Processing {len(chunks)} node chunks in parallel with {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {
            executor.submit(process_node_chunk, chunk): chunk 
            for chunk in chunks
        }
        
        for future in as_completed(future_to_chunk):
            try:
                result = future.result()
                refined_nodes.extend(result)
                if len(refined_nodes) % (CHUNK_SIZE * 2) == 0:
                    save_json(refined_nodes, output_file + ".partial")
                    print(f"Saved progress: {len(refined_nodes)} nodes so far")
            except Exception as e:
                chunk_idx, failed_chunk = future_to_chunk[future]
                print(f"Chunk {chunk_idx + 1} failed: {e}")
                refined_nodes.extend(failed_chunk)
    
    seen_ids = set()
    unique_nodes = []
    for node in refined_nodes:
        if node["id"] not in seen_ids:
            seen_ids.add(node["id"])
            unique_nodes.append(node)
    
    save_json(unique_nodes, output_file)
    print(f"Refined {len(unique_nodes)} nodes saved to {output_file}")

def refine_edges_parallel(edges_file, output_file, use_llm=False):
    edges = load_json(edges_file)
    print(f"Loaded {len(edges)} edges from {edges_file}")
    
    validated_edges = []
    for i, edge in enumerate(edges):
        validated_edge = validate_edge(edge)
        if validated_edge:
            validated_edges.append(validated_edge)
        else:
            print(f"[Warning] Skipped invalid edge at index {i}: {edge}")
    
    print(f"After validation: {len(validated_edges)} edges")
    
    if not use_llm:
        seen_edges = set()
        refined_edges = []
        for edge in validated_edges:
            key = (edge["source"], edge["target"], edge["edge_type"])
            if key not in seen_edges:
                seen_edges.add(key)
                refined_edges.append(edge)
        save_json(refined_edges, output_file)
        print(f"Processed {len(refined_edges)} edges (no LLM)")
        return
    
    chunks = list(enumerate(chunk_list(validated_edges, CHUNK_SIZE)))
    refined_edges = []
    
    print(f"Processing {len(chunks)} edge chunks in parallel with {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {
            executor.submit(process_edge_chunk, chunk): chunk 
            for chunk in chunks
        }
        
        for future in as_completed(future_to_chunk):
            try:
                result = future.result()
                refined_edges.extend(result)
                if len(refined_edges) % (CHUNK_SIZE * 2) == 0:
                    save_json(refined_edges, output_file + ".partial")
                    print(f"Saved progress: {len(refined_edges)} edges so far")
            except Exception as e:
                chunk_idx, failed_chunk = future_to_chunk[future]
                print(f"Chunk {chunk_idx + 1} failed: {e}")
                refined_edges.extend(failed_chunk)
    
    seen_edges = set()
    unique_edges = []
    for edge in refined_edges:
        key = (edge["source"], edge["target"], edge["edge_type"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(edge)
    
    save_json(unique_edges, output_file)
    print(f"Refined {len(unique_edges)} edges saved to {output_file}")

if __name__ == "__main__":
    try:
        start_time = time.time()
        
        print("Starting parallel node refinement...")
        refine_nodes_parallel("graph_nodes.json", "graph_nodes_refined.json", use_llm=True)
        
        node_time = time.time()
        print(f"Node refinement completed in {node_time - start_time:.2f} seconds")
        
        print("Starting parallel edge refinement...")
        refine_edges_parallel("graph_edges.json", "graph_edges_refined.json", use_llm=True)
        
        end_time = time.time()
        print(f"Edge refinement completed in {end_time - node_time:.2f} seconds")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print("Refinement completed successfully!")
        
    except KeyboardInterrupt:
        print("\n[Info] Process interrupted by user")
    except Exception as e:
        print(f"[Fatal] An error occurred: {e}")
        import traceback
        traceback.print_exc()