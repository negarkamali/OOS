import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Define file path
file_path = "../data/data_oos_plus.json"

# Load data from JSON file
with open(file_path, 'r') as f:
    data = json.load(f)

# Separate features (X) and labels (y)
X_train, y_train = zip(*[(embedding, label) for embedding, label in data["oos_train"]])
X_val, y_val = zip(*[(embedding, label) for embedding, label in data["oos_val"]])
X_test, y_test = zip(*[(embedding, label) for embedding, label in data["oos_test"]])

# Convert lists to numpy arrays
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

# Convert labels to integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

# Convert integer labels to one-hot encoding
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)
