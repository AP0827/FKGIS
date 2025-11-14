CREATE TABLE cases(
    case_id VARCHAR(50) PRIMARY KEY,
    title TEXT NOT NULL,
    reporting_officer TEXT,
    investigating_officer TEXT,
    status VARCHAR(50) DEFAULT 'Open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE rawfiles(
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    filename TEXT,
    file_desc TEXT
);

CREATE TABLE documents(
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    doc_type VARCHAR(50),  -- narrative, biography, interview, dispatch
    title TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Legacy tables, can be migrated to documents
CREATE TABLE narratives(
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    type VARCHAR(50),
    text TEXT
);

CREATE TABLE victim_biographies(
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    victim_name TEXT,
    biography TEXT
);

CREATE TABLE suspect_interviews(
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    suspect_name TEXT,
    interview_text TEXT,
    interview_date TIMESTAMP
);

CREATE TABLE entities(
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    entity_type VARCHAR(50),
    entity_name TEXT,
    normalized_name TEXT
);

CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    description TEXT,
    timestamp TIMESTAMP,
    location TEXT,
    entities_involved INTEGER[]
);

CREATE TABLE relations (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    subject_id INT REFERENCES entities(id) ON DELETE CASCADE,
    predicate VARCHAR(100),
    object_id INT REFERENCES entities(id) ON DELETE CASCADE
);

-- Global entity pool and unified case knowledge graph
-- For each case_id, the combination of:
--   - entities (local mentions)
--   - global_entities (canonical nodes)
--   - relations (edges)
--   - events (timeline)
-- together form ONE unified knowledge graph for that case.
 
CREATE TABLE global_entities (
    global_id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    canonical_name TEXT NOT NULL,
    entity_type VARCHAR(50),
    source_docs TEXT[]
);
 
ALTER TABLE entities
ADD COLUMN global_id INT REFERENCES global_entities(global_id) ON DELETE SET NULL;
 
ALTER TABLE relations
ADD COLUMN subject_global_id INT REFERENCES global_entities(global_id),
ADD COLUMN object_global_id INT REFERENCES global_entities(global_id);
 
ALTER TABLE events
ADD COLUMN actor_global_id INT REFERENCES global_entities(global_id);