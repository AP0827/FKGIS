from neomodel import StructuredNode, StringProperty, DateTimeProperty, RelationshipTo, RelationshipFrom

class EntityNode(StructuredNode):
    name = StringProperty(unique_index=True)
    entity_type = StringProperty()
    normalized_name = StringProperty()

    # Relationships
    related_to = RelationshipTo('EntityNode', 'RELATED')
    involved_in = RelationshipFrom('EventNode', 'INVOLVES')

class EventNode(StructuredNode):
    description = StringProperty()
    timestamp = DateTimeProperty()
    location = StringProperty()

    # Relationships
    involves = RelationshipTo('EntityNode', 'INVOLVES')
    part_of = RelationshipFrom('CaseNode', 'HAS_EVENT')

class CaseNode(StructuredNode):
    case_id = StringProperty(unique_index=True)
    title = StringProperty()
    status = StringProperty()

    # Relationships
    has_events = RelationshipTo('EventNode', 'HAS_EVENT')
    has_entities = RelationshipTo('EntityNode', 'HAS_ENTITY')