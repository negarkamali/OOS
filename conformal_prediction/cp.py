import os
import sys
from turtle import shape
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from mapie.classification import MapieClassifier
from sklearn.neural_network import MLPClassifier
import json
# Get the absolute path to the src folder
src_dir = os.path.abspath('../src')

# Add the src directory to sys.path
sys.path.insert(0, src_dir)
import preprocess  # Assuming a preprocess.py file


# Load preprocessed data
(X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder = preprocess.load_and_preprocess_data()


# Convert y_test from one-hot encoding to label encoding for stratification
y_test_labels = np.argmax(y_test, axis=1)

# Split test set into calibration and final test set (50-50 split)
X_cal, X_test_final, y_cal, y_test_final = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
)

# Convert back to one-hot encoding
# y_cal = to_categorical(y_cal, num_classes=y_test.shape[1])
# y_test_final = to_categorical(y_test_final, num_classes=y_test.shape[1])

# Define and compile the model
model = Sequential([
    Dense(200, input_shape=(X_train.shape[1],), activation='tanh'),
    Dropout(0.5),
    Dense(200, activation='tanh'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

print(y_cal.shape) #(151)
print(y_test_final.shape) #(151)
print("X_test_final shape:", X_test_final.shape)
print("y_test_final shape:", y_test_final.shape)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)

# Evaluate the model on the final test data
loss, accuracy = model.evaluate(X_test_final, y_test_final)
print("Test Accuracy:", accuracy)


# Predictions for in-scope and out-of-scope data
y_test_final_labels = np.argmax(y_test_final, axis=-1)
is_idx = np.where(y_test_final_labels != label_encoder.transform(['oos'])[0])[0]
oos_idx = np.where(y_test_final_labels == label_encoder.transform(['oos'])[0])[0]

# In-scope predictions
y_pred_is = model.predict(X_test_final[is_idx])
y_true_is = y_test_final[is_idx]

# Out-of-scope predictions
y_pred_oos = model.predict(X_test_final[oos_idx])
y_true_oos = y_test_final[oos_idx]

# Recall for out-of-scope data
oos_recall = recall_score(np.argmax(y_true_oos, axis=-1), np.argmax(y_pred_oos, axis=-1), average='micro')
print("Out-of-scope Recall:", oos_recall)

# # # Save the model
# # model.save("models/cp_model_mlp.keras")

# # # Save the training history
# # history = model.history.history
# # with open('results/cp_training_history.json', 'w') as f:
# #     json.dump(history, f)

# # # Save the evaluation results
# # with open('results/cp_evaluation_results.txt', 'w') as f:
# #     f.write("Test Accuracy: {:.2f}%\n".format(accuracy * 100))
# #     f.write("Out-of-scope Recall: {:.2f}%\n".format(oos_recall * 100))

# # # Save the predictions
# # predictions = model.predict(X_test_final)
# # np.savetxt('results/cp_predictions.csv', predictions, delimiter=',')

# Conformal prediction
# Convert the target values from one-hot encoding to label encoding

y_train_labels = np.argmax(y_train, axis=1)
y_cal_labels = np.argmax(y_cal, axis=1)


# Initialize and train the MLP classifier
clf = MLPClassifier(random_state=42, max_iter=500)  # You may need to adjust max_iter
clf.fit(X_train, y_train_labels)

y_pred_proba = clf.predict_proba(X_train)
# Initialize MAPIE
mapie_score = MapieClassifier(estimator=clf, cv="prefit", method="score")
mapie_score.fit(X_cal, y_cal_labels)

alpha = [0.05]
y_pred_score, y_ps_score = mapie_score.predict(X_train, alpha=alpha)


print("Shape of y_train:", y_train.shape)
print("Data type of y_train:", y_train.dtype)

results = pd.DataFrame(list(map(np.ravel, y_ps_score)))
results['prediction'] = y_pred_score

y_train_labels = np.argmax(y_train, axis=1)
results['label'] = y_train_labels

def set_pandas_display_options() -> None:
    """Set pandas display options."""
    # Ref: https://stackoverflow.com/a/52432757/
    display = pd.options.display

    display.max_columns = 1000
    display.max_rows = 1000
    display.max_colwidth = 199
    display.width = 1000
    # display.precision = 2  # set as needed
    # display.float_format = lambda x: '{:,.2f}'.format(x)  # set as needed
set_pandas_display_options()

filtered_results = results.loc[results['prediction'] != results['label']]
print("Shape of filtered_results:", filtered_results.shape)

print(filtered_results)

y_pred_score_without, y_ps_score_without = mapie_score.predict(X_test, alpha=alpha)
results_without = pd.DataFrame(list(map(np.ravel, y_ps_score_without)))
results_without['prediction'] = y_pred_score_without

y_test_labels = np.argmax(y_test, axis=1)
results_without['label'] = y_test_labels

print("Shape of X_test:", X_test.shape)
print("Data type of X_test:", X_test.dtype)


filtered_results_without = results_without.loc[results_without['prediction'] != results_without['label']]

print(filtered_results_without)