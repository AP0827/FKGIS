from mongoengine import Document, EmbeddedDocument, StringField, DateTimeField, ListField, EmbeddedDocumentField, DictField
from datetime import datetime

# Embedded documents
class Narrative(EmbeddedDocument):
    type = StringField()  # e.g., 'reporting_officer'
    text = StringField()

class Victim(EmbeddedDocument):
    name = StringField()
    biography = StringField()

class Interview(EmbeddedDocument):
    text = StringField()
    date = DateTimeField()

class Suspect(EmbeddedDocument):
    name = StringField()
    interviews = ListField(EmbeddedDocumentField(Interview))

class Entity(EmbeddedDocument):
    entity_type = StringField()
    entity_name = StringField()
    normalized_name = StringField()

class Event(EmbeddedDocument):
    description = StringField()
    timestamp = DateTimeField()
    location = StringField()
    entities_involved = ListField(StringField())  # list of entity names

class Relation(EmbeddedDocument):
    subject = StringField()  # entity name
    predicate = StringField()
    object = StringField()  # entity name

# Main document
class Case(Document):
    case_id = StringField(primary_key=True)
    title = StringField(required=True)
    reporting_officer = StringField()
    investigating_officer = StringField()
    status = StringField(default='Open')
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    # Raw data storage
    raw_texts = ListField(StringField())  # list of raw text inputs
    raw_files = ListField(DictField())  # list of {'filename': str, 'data': bytes} for images/docs

    # Processed data
    narratives = ListField(EmbeddedDocumentField(Narrative))
    victims = ListField(EmbeddedDocumentField(Victim))
    suspects = ListField(EmbeddedDocumentField(Suspect))
    entities = ListField(EmbeddedDocumentField(Entity))
    events = ListField(EmbeddedDocumentField(Event))
    relations = ListField(EmbeddedDocumentField(Relation))

    meta = {'collection': 'cases'}