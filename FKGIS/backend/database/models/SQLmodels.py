import psycopg2
from collections import namedtuple

# Named tuples for data structures
Case = namedtuple('Case', ['case_id', 'title', 'reporting_officer', 'investigating_officer', 'status'])
RawFile = namedtuple('RawFile', ['id', 'case_id', 'filename'])
Entity = namedtuple('Entity', ['id', 'case_id', 'entity_type', 'entity_name', 'normalized_name'])
Event = namedtuple('Event', ['id', 'case_id', 'description', 'timestamp', 'location', 'entities_involved'])
Relation = namedtuple('Relation', ['id', 'case_id', 'subject_id', 'predicate', 'object_id'])

def get_connection(dbname='your_db', user='your_user', password='your_pass', host='localhost', port='5432'):
    return psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)

class CaseModel:
    @staticmethod
    def get_all(conn):
        with conn.cursor() as cur:
            cur.execute("SELECT case_id, title, reporting_officer, investigating_officer, status FROM cases")
            rows = cur.fetchall()
            return [Case(*row) for row in rows]

    @staticmethod
    def get_by_id(conn, case_id):
        with conn.cursor() as cur:
            cur.execute("SELECT case_id, title, reporting_officer, investigating_officer, status FROM cases WHERE case_id = %s", (case_id,))
            row = cur.fetchone()
            return Case(*row) if row else None

class RawFileModel:
    @staticmethod
    def get_all(conn):
        with conn.cursor() as cur:
            cur.execute("SELECT id, case_id, filename FROM rawfiles")
            rows = cur.fetchall()
            return [RawFile(*row) for row in rows]

    @staticmethod
    def get_by_id(conn, rawfile_id):
        with conn.cursor() as cur:
            cur.execute("SELECT id, case_id, filename FROM rawfiles WHERE id = %s", (rawfile_id,))
            row = cur.fetchone()
            return RawFile(*row) if row else None

    @staticmethod
    def get_by_case_id(conn, case_id):
        with conn.cursor() as cur:
            cur.execute("SELECT id, case_id, filename FROM rawfiles WHERE case_id = %s", (case_id,))
            rows = cur.fetchall()
            return [RawFile(*row) for row in rows]

class EntityModel:
    @staticmethod
    def get_all(conn):
        with conn.cursor() as cur:
            cur.execute("SELECT id, case_id, entity_type, entity_name, normalized_name FROM entities")
            rows = cur.fetchall()
            return [Entity(*row) for row in rows]

    @staticmethod
    def get_by_id(conn, entity_id):
        with conn.cursor() as cur:
            cur.execute("SELECT id, case_id, entity_type, entity_name, normalized_name FROM entities WHERE id = %s", (entity_id,))
            row = cur.fetchone()
            return Entity(*row) if row else None

    @staticmethod
    def get_by_case_id(conn, case_id):
        with conn.cursor() as cur:
            cur.execute("SELECT id, case_id, entity_type, entity_name, normalized_name FROM entities WHERE case_id = %s", (case_id,))
            rows = cur.fetchall()
            return [Entity(*row) for row in rows]

class EventModel:
    @staticmethod
    def get_all(conn):
        with conn.cursor() as cur:
            cur.execute("SELECT id, case_id, description, timestamp, location, entities_involved FROM events")
            rows = cur.fetchall()
            return [Event(*row) for row in rows]

    @staticmethod
    def get_by_id(conn, event_id):
        with conn.cursor() as cur:
            cur.execute("SELECT id, case_id, description, timestamp, location, entities_involved FROM events WHERE id = %s", (event_id,))
            row = cur.fetchone()
            return Event(*row) if row else None

    @staticmethod
    def get_by_case_id(conn, case_id):
        with conn.cursor() as cur:
            cur.execute("SELECT id, case_id, description, timestamp, location, entities_involved FROM events WHERE case_id = %s", (case_id,))
            rows = cur.fetchall()
            return [Event(*row) for row in rows]

class RelationModel:
    @staticmethod
    def get_all(conn):
        with conn.cursor() as cur:
            cur.execute("SELECT id, case_id, subject_id, predicate, object_id FROM relations")
            rows = cur.fetchall()
            return [Relation(*row) for row in rows]

    @staticmethod
    def get_by_id(conn, relation_id):
        with conn.cursor() as cur:
            cur.execute("SELECT id, case_id, subject_id, predicate, object_id FROM relations WHERE id = %s", (relation_id,))
            row = cur.fetchone()
            return Relation(*row) if row else None

    @staticmethod
    def get_by_case_id(conn, case_id):
        with conn.cursor() as cur:
            cur.execute("SELECT id, case_id, subject_id, predicate, object_id FROM relations WHERE case_id = %s", (case_id,))
            rows = cur.fetchall()
            return [Relation(*row) for row in rows]