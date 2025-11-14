from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, ForeignKey, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Case(Base):
    __tablename__ = 'cases'
    case_id = Column(String(50), primary_key=True)
    title = Column(Text, nullable=False)
    reporting_officer = Column(Text)
    investigating_officer = Column(Text)
    status = Column(String(50), default='Open')
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

class RawFile(Base):
    __tablename__ = 'rawfiles'
    id = Column(Integer, primary_key=True)
    case_id = Column(String(50), ForeignKey('cases.case_id', ondelete='CASCADE'))
    filename = Column(Text)
    file_desc = Column(Text)

class Narrative(Base):
    __tablename__ = 'narratives'
    id = Column(Integer, primary_key=True)
    case_id = Column(String(50), ForeignKey('cases.case_id', ondelete='CASCADE'))
    type = Column(String(50))
    text = Column(Text)

class VictimBiography(Base):
    __tablename__ = 'victim_biographies'
    id = Column(Integer, primary_key=True)
    case_id = Column(String(50), ForeignKey('cases.case_id', ondelete='CASCADE'))
    victim_name = Column(Text)
    biography = Column(Text)

class SuspectInterview(Base):
    __tablename__ = 'suspect_interviews'
    id = Column(Integer, primary_key=True)
    case_id = Column(String(50), ForeignKey('cases.case_id', ondelete='CASCADE'))
    suspect_name = Column(Text)
    interview_text = Column(Text)
    interview_date = Column(TIMESTAMP)

class Entity(Base):
    __tablename__ = 'entities'
    id = Column(Integer, primary_key=True)
    case_id = Column(String(50), ForeignKey('cases.case_id', ondelete='CASCADE'))
    entity_type = Column(String(50))
    entity_name = Column(Text)
    normalized_name = Column(Text)

class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    case_id = Column(String(50), ForeignKey('cases.case_id', ondelete='CASCADE'))
    description = Column(Text)
    timestamp = Column(TIMESTAMP)
    location = Column(Text)
    entities_involved = Column(ARRAY(Integer))

class Relation(Base):
    __tablename__ = 'relations'
    id = Column(Integer, primary_key=True)
    case_id = Column(String(50), ForeignKey('cases.case_id', ondelete='CASCADE'))
    subject_id = Column(Integer, ForeignKey('entities.id', ondelete='CASCADE'))
    predicate = Column(String(100))
    object_id = Column(Integer, ForeignKey('entities.id', ondelete='CASCADE'))