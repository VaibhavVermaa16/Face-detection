from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv('MONGODB_URI')
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['face_recognition_db']
collection = db['embeddings']

# Function to save embeddings
def save_embedding_to_db(embedding, data):
    avg_embedding = np.mean(embedding, axis=0)
    embedding = avg_embedding.tolist()
  

    # Prepare the document to be inserted into the database
    document = {
        'id': data['id'],
        'embedding': embedding, # Save the embedding (list or array)
        'name': data['name'],
        'age': data['age'],
        'email': data['email'],
        'phone': data['phone'],
        'address': data['address']
    }

    # Insert the document into the MongoDB collection
    collection.insert_one(document)

# Function to retrieve all embeddings from the database
def get_all_embeddings_from_db():
    return list(collection.find({}, {'_id': 0}))  # Exclude MongoDB's default ID

def delete_person_from_db(id):
    collection.delete_one({'id': id})
    return 'Person deleted successfully'
