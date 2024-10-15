import cv2
from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
from app.utils import delete_person_from_db, save_embedding_to_db, get_all_embeddings_from_db
import os
from werkzeug.utils import secure_filename
from app.feature_extractor import feature_extractor
from app.find_best_embedding import find_best_embedding
import uuid
from flask import jsonify

app = Flask(__name__)

# Set upload folder and allowed file types
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def home():
    persons = get_all_embeddings_from_db()
    return jsonify({"users":persons})


# Route to handle image upload and embedding extraction
@app.route('/add_person', methods=['POST'])
def add_person():
    if 'files' not in request.files:
        return jsonify({"message": 'No file part in the request'})

    # generate a unique id for the person
    id = str(uuid.uuid4())
    name = request.form.get('name')
    age = request.form.get('age')
    email = request.form.get('email')
    phone = request.form.get('phone')
    address = request.form.get('address')
    
    # Get the list of uploaded files
    files = request.files.getlist('files')
    
    # Validate that all form data is provided
    if not all([name, age, email, phone, address]):
        return jsonify({'error': 'Missing required form data'}), 400
    
    # Ensure at least 4 files are uploaded
    if len(files) < 4:
        return jsonify({"message": 'Please upload at least 4 images.'}), 200

    embeddings = []   
    for file in files:
        if file.filename == '':
            return jsonify({"message": 'No file selected'}), 200

        # Ensure the file type is allowed
        if file and allowed_file(file.filename):
            file_bytes = np.fromstring(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            embedding = feature_extractor(image)

            if embedding is None:
                return jsonify({"message": 'No face detected in the image %s' % file.filename})
            else:
                embeddings.append(embedding) # Store the embedding for this image

    data = {
        'id': id,
        'name': name,
        'age': age,
        'email': email,
        'phone': phone,
        'address': address,
    }
    # Save all embeddings and user data to the database
    save_embedding_to_db(embeddings, data)

    return jsonify({"message": 'Person added successfully', "data": data}), 200
  


# API route to get all embeddings
@app.route('/get_embeddings', methods=['GET'])
def get_embeddings():
    persons = get_all_embeddings_from_db()
    
    return jsonify({"users":persons})


@app.route('/search', methods=['POST'])
def search():
    # name = request.form.get('name')
    persons = get_all_embeddings_from_db()
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"message": 'No file selected'}), 200
    
    if file and allowed_file(file.filename):
        file_bytes = np.fromstring(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        embeddings = feature_extractor(image)
        
        if embeddings is None:
            return jsonify({"message": 'No face detected in the image %s' % file.filename}), 200
        
        
        best_match, score = find_best_embedding(embeddings, persons)
        return jsonify({'message': 'Person found successfully',
                        "person":best_match, 
                        "score":score}), 200
        
    return jsonify({"message": 'Invalid file type'}), 200
    
def delete_person(id):
    message=delete_person_from_db(id)
    return jsonify({"message": message}), 200

if __name__ == '__main__':
    app.run(debug=True)
