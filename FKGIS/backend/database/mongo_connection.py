from mongoengine import connect
import os

def connect_to_mongo():
    # Default to local MongoDB
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/fkgis')
    connect(host=mongo_uri)
    print(f"Connected to MongoDB at {mongo_uri}")