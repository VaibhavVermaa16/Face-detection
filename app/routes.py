from flask import Flask, request, render_template, redirect, url_for, flash
from app.utils import save_embedding_to_db, get_all_embeddings_from_db
import os
from werkzeug.utils import secure_filename
from app.feature_extractor import feature_extractor
from app.find_best_embedding import find_best_embedding
import uuid

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
    return render_template('index.html', persons=persons)

# To route to the upload page
@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('upload.html')


# Route to handle image upload and embedding extraction
@app.route('/add_person', methods=['POST'])
def add_person():
    if 'files' not in request.files:
        return render_template('success.html', message='No file part in the request')

    # Fetch form data
    name = request.form.get('name')
    age = request.form.get('age')
    email = request.form.get('email')
    phone = request.form.get('phone')
    address = request.form.get('address')
    
    # Get the list of uploaded files
    files = request.files.getlist('files')
    
    # Ensure at least 4 files are uploaded
    if len(files) < 4:
        flash('Please upload at least 4 images.')
        return redirect(url_for('upload_page'))

    embeddings = []
    images = []
    for file in files:
        if file.filename == '':
            return render_template('success.html', message='No file selected')

        # Ensure the file type is allowed
        if file and allowed_file(file.filename):
            # Save the file with a unique name to avoid overwriting
            unique_filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            # Extract the face embedding
            embedding = feature_extractor(file_path)

            if embedding is None:
                return render_template('success.html', message='No face detected in the image %s' % file.filename)
            else:
                embeddings.append(embedding) # Store the embedding for this image
                images.append(file_path) # Store the image path

    data = {
        'name': name,
        'age': age,
        'email': email,
        'phone': phone,
        'address': address,
        'image_paths': images
    }
    # Save all embeddings and user data to the database
    save_embedding_to_db(embeddings, data)

    # Redirect to the get_embeddings page after successful submission
    return redirect(url_for('get_embeddings'))


# API route to get all embeddings
@app.route('/get_embeddings', methods=['GET'])
def get_embeddings():
    persons = get_all_embeddings_from_db()
    
    return render_template('all_persons.html', persons=persons)


@app.route('/find_person', methods=['GET'])
def find_person():
    return render_template('query.html')


@app.route('/search', methods=['POST'])
def search():
    name = request.form.get('name')
    persons = get_all_embeddings_from_db()
    file = request.files['file']
    if file.filename == '':
        return render_template('success.html', message='No file selected')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract the face embedding
        embeddings = feature_extractor(file_path)
        
        if embeddings is None:
            return render_template('success.html', message='No face detected in the image %s' % file.filename)
        
        
        best_match, img_path, score = find_best_embedding(embeddings, persons)
        # print(img_path)
        return render_template('person_found.html', person=best_match, img_path=img_path, score=score)
    
    return render_template('success.html', message='Invalid file type')

if __name__ == '__main__':
    app.run(debug=True)
