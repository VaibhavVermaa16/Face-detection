import cv2
import dropbox
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


DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")  # Store in environment for security

# Initialize Dropbox Client
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


try:
    dbx.users_get_current_account()
    print("You successfully connected to Dropbox!")
except Exception as e:
    print(e)

# Function to upload image to Dropbox
def upload_to_dropbox(file, path):
    try:
        file.seek(0)  # Move the cursor to the beginning of the file
        dbx.files_upload(file.read(), path)
        shared_link_metadata = dbx.sharing_create_shared_link(path)
        print(shared_link_metadata.url)
        return shared_link_metadata.url  # Return the shared link for the uploaded file
    except Exception as e:
        print(f"Failed to upload file to Dropbox: {e}")
        return None

# Function to download image from Dropbox to local file
def download_from_dropbox(path):
    try:
        # Download the file content from Dropbox as bytes
        metadata, response = dbx.files_download(path)
        file_bytes = response.content
        if not file_bytes:
            print("File is empty or not downloaded correctly.")
            return None

        # Convert the byte content to a NumPy array
        np_arr = np.frombuffer(file_bytes, np.uint8)

        # Decode the NumPy array into an image (OpenCV format)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            print("Failed to decode image")
            return None

        # Now you can process the image (e.g., extract embeddings)
        return image

    except Exception as e:
        print(f"Failed to process image: {e}")
        return None

# Function to delete image from Dropbox
def delete_from_dropbox(path):
    try:
        dbx.files_delete_v2(path)
        print(f"Deleted {path} from Dropbox")
    except Exception as e:
        print(f"Failed to delete file from Dropbox: {e}")


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
