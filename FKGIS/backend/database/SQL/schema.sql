CREATE TABLE cases(
    case_id VARCHAR(50) PRIMARY KEY,
    title TEXT NOT NULL,
    reporting_officer TEXT,
    investigating_officer TEXT,
    status VARCHAR(50) DEFAULT 'Open',
);

CREATE TABLE rawfiles(
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    filename TEXT,
    file_desc TEXT
)

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
    entities_involved INTEGER[],
);

CREATE TABLE relations (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id) ON DELETE CASCADE,
    subject_id INT REFERENCES entities(id) ON DELETE CASCADE,
    predicate VARCHAR(100),
    object_id INT REFERENCES entities(id) ON DELETE CASCADE,
);