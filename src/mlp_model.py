from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import recall_score
import numpy as np
import preprocess
import os

# Load preprocessed data
(X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder = preprocess.load_and_preprocess_data()

# Define and compile the model
model = Sequential([
    Dense(200, input_shape=(X_train.shape[1],), activation='tanh'),
    Dropout(0.5),
    Dense(200, activation='tanh'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Identify the 'oos' class index
oos_class_index = label_encoder.transform(['oos'])[0]

# Predictions for out-of-scope data 
oos_predictions = model.predict(X_test[y_test[:, oos_class_index] == 1])
oos_predictions_labels = np.argmax(oos_predictions, axis=-1)

# True labels for out-of-scope data
y_test_oos_true_labels = np.full(oos_predictions_labels.shape, oos_class_index)

# Calculate recall for out-of-scope data
oos_recall = recall_score(y_test_oos_true_labels, oos_predictions_labels, average='micro')
print("Out-of-scope Recall:", oos_recall)

model.save("models/model_mlp.keras")

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)

import json
with open('results/training_history.json', 'w') as f:
    json.dump(history.history, f)

loss, accuracy = model.evaluate(X_test, y_test)
with open('results/evaluation_results.txt', 'w') as f:
    f.write("Test Accuracy: {:.2f}%\n".format(accuracy * 100))
    f.write("Out-of-scope Recall: {:.2f}%\n".format(oos_recall * 100))

predictions = model.predict(X_test)
np.savetxt('results/predictions.csv', predictions, delimiter=',')
