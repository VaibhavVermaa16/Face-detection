from app.feature_extractor import feature_extractor
import numpy as np


def add_new_person(img_path,model,embeddings_list, name):
  embedding = feature_extractor(img_path, model)  # Get the embedding
  if embedding is not None:
    embeddings_list.append({
                'Name': name,  # Store the image path
                'embedding': embedding.flatten()  # Flatten the embedding to make it 1D
            })
    # return embeddings_list
  else:
    print(f"Skipping {img_path} due to face extraction failure.")
  return embeddings_list