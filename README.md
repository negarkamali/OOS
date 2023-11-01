# Model Results

This directory contains the results of our Keras model.

## Files

- `model.h5`: The saved Keras model.
- `training_history.json`: The training history of the model, including loss and accuracy over epochs.
- `evaluation_results.txt`: The evaluation results on the test set.
- `predictions.csv`: The raw predictions of the model on the test set.

## How to Load the Model

You can load the model back into Keras with the following code:

```python
from tensorflow.keras.models import load_model
model = load_model('path/to/your/model.h5')
