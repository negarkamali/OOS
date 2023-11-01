import json
import tensorflow_hub as hub
import numpy as np
import os

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def embed_sentences(sentences):
    return embed(sentences)

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_and_save(data, output_file):
    for key in data:
        sentences = [item[0] for item in data[key]]
        embeddings = embed_sentences(sentences)
        for i, item in enumerate(data[key]):
            item[0] = embeddings[i].numpy().tolist()  # Convert embedding to list
    with open(output_file, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    input_file = "../data/data_full.json"
    output_file = "../data/data_oos_plus_embedded.json"
    data = load_data(input_file)
    preprocess_and_save(data, output_file)
