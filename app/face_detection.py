# Description: This file contains the code to detect faces in an image using MTCNN.
from mtcnn import MTCNN
from PIL import Image
import numpy as np

detector=MTCNN()

def extract_face(img_path):

    # Load the image using PIL
    try:
        sample_img = Image.open(img_path)

        # Check if the image has an alpha channel (4 channels - RGBA)
        if sample_img.mode == 'RGBA':
            sample_img = sample_img.convert('RGB')  # Convert RGBA to RGB

        # Convert to NumPy array
        sample_img = np.array(sample_img)
    except Exception as e:
        print(f"Failed to load image with PIL: {e}")
        return None

    # Check if the image is empty
    if sample_img.size == 0:
        print(f"Image is empty: {img_path}")
        return None

    # Detect faces in the image
    results = detector.detect_faces(sample_img)

    # Process results
    if results:
        x, y, width, height = results[0]['box']
        face = sample_img[y:y + height, x:x + width]
        return face

    print(f"No face found in image: {img_path}")
    return None