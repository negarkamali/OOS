import json
import re

INPUT_FILE = '../data/data_full.json'
OUTPUT_FILE = '../data/cleaned_data.json'

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Simple tokenization (splitting by spaces)
    tokens = text.split()
    
    return ' '.join(tokens)

def clean_data():
    with open(INPUT_FILE, 'r') as file:
        data = json.load(file)

    cleaned_data = {}
    for key, value in data.items():
        cleaned_data[key] = [(clean_text(item[0]), item[1]) for item in value]

    with open(OUTPUT_FILE, 'w') as file:
        json.dump(cleaned_data, file, indent=4)

if __name__ == '__main__':
    clean_data()
