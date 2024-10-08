# **Face Recognition System**

## **Overview**
This project implements a robust **face recognition system** designed to identify individuals based on their facial features. 
It allows users to upload images of persons, extract their facial embeddings using a pre-trained deep learning model, 
and store them in a database. The system also supports searching for matches based on query images and displaying all 
stored individuals dynamically.

### **Features**
- **Add Person**: Upload an image of a person, extract their facial embeddings, and store the details in a database.
- **Search Person**: Query a new image to find the closest matching face from the stored database.
- **Display All Persons**: View all stored individuals in a user-friendly card format.
- **Facial Embedding Extraction**: Uses the pre-trained FaceNet model to extract facial embeddings.
- **Database Storage**: Store and manage facial embeddings in **MongoDB Atlas**.

## **Project Structure**
```
Face-detection/
├── app/
│   ├── face_detection.py        # Handles face detection
│   ├── feature_extractor.py     # Extract embeddings using FaceNet
│   ├── find_best_embedding.py   # Find the best matching embedding
│   ├── routes.py                # Flask routes (API endpoints)
│   ├── templates/               # Jinja templates for web pages
│   │   ├── all_persons.html     # Displays all persons
│   │   ├── index.html           # Homepage
│   │   ├── person_found.html    # Result page for matched person
│   │   ├── query.html           # Search form page
│   │   ├── success.html         # Success message page
│   │   └── upload.html          # Form to upload a new person
│   └── utils.py                 # Utility functions for MongoDB
├── config.py                    # Configuration file (e.g., MongoDB URI)
├── main.py                      # Main entry point to run Flask app
├── models/
│   └── model.py                 # FaceNet model loading and management
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
```

## **Setup Instructions**

### **Prerequisites**
Ensure the following software is installed:
- Python 3.12
- MongoDB Atlas account and cluster
- Pip (Python package manager)

### Step 1: Clone the Repository
```
git clone https://github.com/your-repository/face-detection.git
cd face-detection
```
### Step 2: Set Up Virtual Environment
```
python3 -m venv .venv
source .venv/bin/activate  # For Linux/macOS
# For Windows use: .venv\Scripts\activate
```
### Step 3: Install Dependencies
```
pip install -r requirements.txt
```
### Step 4: Configure MongoDB Atlas
- Create a MongoDB Atlas cluster and obtain your connection string (MongoDB URI).
- Replace the placeholder in config.py with your actual MongoDB connection URI.
### Step 5: Run the Application
```
python main.py
```

The Flask server should now be running at http://127.0.0.1:5000/.

## **Usage**
### **Adding person**:
1. Navigate to /upload to upload a new person's image and label.

2. The image is processed, and the person's facial embeddings are stored in the database.

### **Searching for a Person**
1. Navigate to /search to upload a query image and search for the closest matching person in the database.

### **Viewing All Persons**
1. Navigate to /all_users to see a list of all persons stored in the system, displayed in card format.

## **API Endpoints**
1. **Add Person** (POST /```add_person```)
Allows users to upload an image and add a new person.
- Request:
    - file: The image of the person to upload.
    - label: The person's name or ID.
- Response:
    - 200 OK: Person successfully added.
    - 400 Error: Error in file upload or embedding extraction.

2. **Search Person** (POST /```search_person```)
Search for a matching face in the database using a query image.
- Request:
    - file: The image of the person to search.
- Response:
    - 200 OK: Match found.
    - 400 Error: No match found or error during embedding extraction.

3. **Display All Persons** (GET /```all_users```)
Show all persons stored in the database.
- Response:
    - HTML page with a list of all stored persons.

## **Future Improvements**
- Real-Time Face Recognition: Expand the system to process video streams and recognize faces in real time.
- Approximate Nearest Neighbors (ANN): Integrate FAISS for faster search and matching in large datasets.
- Security Enhancements: Add user authentication, encrypt sensitive data, and set rate limits on API endpoints.
- Deploy to Cloud: Deploy the project to cloud services like AWS, Heroku, or Google Cloud for global access.
- Mobile App Integration: Develop a mobile app that interacts with the Flask API for real-time face recognition.

## **Contributors**
**Vaibhav Verma** – Lead Developer and Architect

Feel free to reach out with any questions or collaboration requests!
