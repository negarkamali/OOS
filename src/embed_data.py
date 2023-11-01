import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os 

def sentences_to_embeddings(data_path, output_path=None):
    # Load the pre-trained Universal Sentence Encoder model
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    # Load the data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Initialize an empty dictionary to hold the embedded data
    embedded_data = {}
    
    # Convert sentences to embeddings
    for key in data:
        embedded_data[key] = []
        for sentence, label in data[key]:
            # Join the tokens to form a sentence
            sentence = " ".join(sentence)
            embedding = embed([sentence]).numpy().tolist()[0]
            embedded_data[key].append([embedding, label])
    
    # Save the embeddings data (optional)
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(embedded_data, f, indent=4)

    return embedded_data

# Paths
INPUT_PATH = '../data/data_full_tokenized.json'
OUTPUT_PATH = '../data/data_oos_plus_embedded.json'

# Convert sentences to embeddings and save
embedded_data = sentences_to_embeddings(INPUT_PATH, OUTPUT_PATH)
