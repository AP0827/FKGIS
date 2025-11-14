from neomodel import config
import os

def connect_to_neo4j():
    # Default to local Neo4j
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
    config.DATABASE_URL = f'{neo4j_uri}://{neo4j_user}:{neo4j_password}'
    print(f"Connected to Neo4j at {neo4j_uri}")