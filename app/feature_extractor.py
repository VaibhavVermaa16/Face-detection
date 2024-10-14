# This file contains the code for extracting the face embeddings from the image using FaceNet model.
import numpy as np
from tensorflow import keras
from keras_facenet import FaceNet
from app.face_detection import extract_face

model = FaceNet()

def feature_extractor(image,model=model):
    face = extract_face(image)
    if face is None:
        print("No face detected in the image.")
        return None
    expanded_img = np.expand_dims(face,axis=0)
    img = keras.applications.efficientnet.preprocess_input(expanded_img)
    embedding = model.embeddings(img)
    return embedding