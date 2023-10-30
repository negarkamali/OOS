import requests
import os

DATA_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, os.pardir, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "data_full.json")

def download_data():
    response = requests.get(DATA_URL)
    response.raise_for_status()

    with open(OUTPUT_FILE, 'w') as file:
        file.write(response.text)

if __name__ == "__main__":
    download_data()


