# Description: This file contains the function to find the best match for a query embedding in a dataset of embeddings.
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def average_embedding(embeddings):
    return np.mean(embeddings, axis=0)

def find_best_embedding(query_embedding, persons, threshold=0.4):
    best_match = None
    min_distance = float('inf')
    score = 0

    for item in persons:
        embedding = item['embedding']
        
        # Ensure all embeddings are NumPy arrays
        # embeddings = [np.array(embedding) if isinstance(embedding, list) else embedding for embedding in embeddings]

        # Calculate the average embedding for the current person
        # avg_embedding = average_embedding(embeddings)
        
        # Calculate cosine similarity
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))
        distance = 1 - similarity[0, 0]  # Convert similarity to distance
        
        # Check if the distance is below the threshold and if it's the best match so far
        if distance < min_distance and distance < threshold:
            min_distance = distance
            best_match = item
            score = 1 - distance  # Similarity score between 0 and 1

    return best_match, score 
 # Return the best match or None if no match found
