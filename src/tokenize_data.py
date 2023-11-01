import json
import nltk
from nltk.tokenize import word_tokenize

# Ensure that you have the punkt tokenizer models downloaded
nltk.download('punkt')

def tokenize_sentence(sentence):
    return word_tokenize(sentence)

def tokenize_data(data_path, output_path=None):
    # Load the data
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Tokenize the data
    for key in data:
        for i, (sentence, label) in enumerate(data[key]):
            tokens = tokenize_sentence(sentence)
            data[key][i][0] = tokens

    # Save the tokenized data (optional)
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

    return data

# Paths
INPUT_PATH = '../data/cleaned_data.json'
OUTPUT_PATH = '../data/data_full_tokenized.json'

# Tokenize and save the data
tokenized_data = tokenize_data(INPUT_PATH, OUTPUT_PATH)
