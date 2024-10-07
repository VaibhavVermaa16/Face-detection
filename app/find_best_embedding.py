# Description: This file contains the function to find the best match for a query embedding in a dataset of embeddings.
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_best_embedding(query_embedding, persons, threshold=0.45):
    best_match = None
    min_distance = float('inf')
    image = None

    for item in persons:
        embeddings = item['embedding']

        # Calculate cosine similarity
        for i in range(len(embeddings)):
            embedding = embeddings[i]
            img = item['image_paths'][i]
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))
            distance = 1 - similarity[0, 0]  # Convert similarity to distance

            # Check if the distance is below the threshold
            if distance < min_distance and distance < threshold:
                min_distance = distance
                best_match = item
                image = img

    return best_match, image  # Return the best match or None if no match found
