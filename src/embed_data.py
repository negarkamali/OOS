import json
import numpy as np

def load_glove_model(glove_file):
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def tokens_to_embeddings(data_path, glove_path, output_path=None):
    embeddings_index = load_glove_model(glove_path)

    # Load the data
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Convert tokens to embeddings
    for key in data:
        for i, (tokens, label) in enumerate(data[key]):
            embedding_matrix = np.array([embeddings_index.get(token, np.zeros(50)) for token in tokens])
            data[key][i][0] = embedding_matrix.tolist()

    # Save the embeddings data (optional)
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

    return data

# Paths
INPUT_PATH = '/Users/Negar/Library/CloudStorage/GoogleDrive-nkamal5@uic.edu/My Drive/2nd PhD/Research/With Jessica/Conformal Prediction/Conformlab/OOS_Intent_Classification/data/data_full_tokenized.json'
GLOVE_PATH = '/Users/Negar/Library/CloudStorage/GoogleDrive-nkamal5@uic.edu/My Drive/2nd PhD/Research/With Jessica/Conformal Prediction/Conformlab/OOS_Intent_Classification/data/glove.6B.50d.txts'
OUTPUT_PATH = '/Users/Negar/Library/CloudStorage/GoogleDrive-nkamal5@uic.edu/My Drive/2nd PhD/Research/With Jessica/Conformal Prediction/Conformlab/OOS_Intent_Classification/data/data_full_embedded.json'

# Convert tokens to embeddings and save
embedded_data = tokens_to_embeddings(INPUT_PATH, GLOVE_PATH, OUTPUT_PATH)
