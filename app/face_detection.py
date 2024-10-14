# Description: This file contains the code to detect faces in an image using MTCNN.
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2

detector=MTCNN()

def preprocess_image(image):
    img = image  # Load the image in color (BGR format by default)
    
    if img is None:
        print(f"Error: Unable to read image")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (160, 160))  # Resize to the model's input size
    return img  # Return as uint8, values in range [0, 255]

def extract_face(image):
    # Load the image using your preprocess function (returns a NumPy array)
    try:
        img = preprocess_image(image)  # Already in NumPy format (RGB)
        if img is None:
            return None
    except Exception as e:
        print(f"Failed to load image: {e}")
        return None

    # Check if the image is empty
    if img.size == 0:
        print(f"Image is empty")
        return None

    # Detect faces in the image
    results = detector.detect_faces(img)

    # Process results
    if results:
        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]
        return face

    print(f"No face found in image")
    return None
