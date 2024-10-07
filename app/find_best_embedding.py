# Description: This file contains the function to find the best match for a query embedding in a dataset of embeddings.
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_best_embedding(query_embedding, dataset_embeddings, threshold=0.45):
    best_match = None
    min_distance = float('inf')

    for item in dataset_embeddings:
        label = item['Name']
        embedding = item['embedding']

        # Calculate cosine similarity
        similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))
        distance = 1 - similarity[0, 0]  # Convert similarity to distance

        # Check if the distance is below the threshold
        if distance < min_distance and distance < threshold:
            min_distance = distance
            best_match = label

    return best_match  # Return the best match or None if no match found
