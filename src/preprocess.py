import json
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    # Define file path
    file_path = "../data/data_oos_plus_embedded.json"

    # Load data from JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Separate features (X) and labels (y) for in-scope and out-of-scope data
    X_train, y_train = zip(*[(embedding, label) for embedding, label in data["train"]])
    X_val, y_val = zip(*[(embedding, label) for embedding, label in data["val"]])
    X_test, y_test = zip(*[(embedding, label) for embedding, label in data["test"]])

    X_train_oos, y_train_oos = zip(*[(embedding, 'oos') for embedding, label in data["oos_train"]])
    X_val_oos, y_val_oos = zip(*[(embedding, 'oos') for embedding, label in data["oos_val"]])
    X_test_oos, y_test_oos = zip(*[(embedding, 'oos') for embedding, label in data["oos_test"]])

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)

    X_train_oos = np.array(X_train_oos)
    X_val_oos = np.array(X_val_oos)
    X_test_oos = np.array(X_test_oos)

    print("X_train shape:", (X_train).shape)
    print("X_train_oos shape:", (X_train_oos).shape)
    # Combine in-scope and out-of-scope data
    X_train_combined = np.vstack([X_train, X_train_oos])
    X_val_combined = np.vstack([X_val, X_val_oos])
    X_test_combined = np.vstack([X_test, X_test_oos])

    y_train_combined = y_train + y_train_oos
    y_val_combined = y_val + y_val_oos
    y_test_combined = y_test + y_test_oos

    # Convert labels to integers using LabelEncoder, fitting on the combined label set
    label_encoder = LabelEncoder()
    y_train_combined = label_encoder.fit_transform(y_train_combined)
    y_val_combined = label_encoder.transform(y_val_combined)
    y_test_combined = label_encoder.transform(y_test_combined)

    # Convert integer labels to one-hot encoding
    y_train_combined = to_categorical(y_train_combined)
    y_val_combined = to_categorical(y_val_combined)
    y_test_combined = to_categorical(y_test_combined)

    
    return (X_train_combined, y_train_combined), (X_val_combined, y_val_combined), (X_test_combined, y_test_combined), label_encoder

if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder = load_and_preprocess_data()
