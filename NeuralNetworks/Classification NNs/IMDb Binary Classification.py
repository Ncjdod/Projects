import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import models
from keras import layers
import numpy as np
import torch

# Check GPU availability
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load data
(training_data, training_label), (test_data, test_label) = keras.datasets.imdb.load_data(num_words=10000)


# Fast NumPy vectorization (replaces slow TensorFlow version)
def vectorize_sequences(sequences, dimension=10000):
    """Vectorize sequences to multi-hot vectors with word counts."""
    results = np.zeros((len(sequences), dimension), dtype='float32')
    for i, sequence in enumerate(sequences):
        unique, counts = np.unique(sequence, return_counts=True)
        results[i, unique] = counts  # Word frequency counts
    return results


# Vectorize the data
print("Vectorizing training data...")
training_data = vectorize_sequences(training_data)
print("Vectorizing test data...")
test_data = vectorize_sequences(test_data)

# Convert labels to float32
training_label = training_label.astype('float32')
test_label = test_label.astype('float32')

print(f"Training data shape: {training_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Build the model
model = models.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Train
print("\nTraining...")
history = model.fit(
    training_data,
    training_label,
    epochs=4,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_data, test_label)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the model
model.save('imdb_binary_classifier.keras')
print("\nModel saved to 'imdb_binary_classifier.keras'")
