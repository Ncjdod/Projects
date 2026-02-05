import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import models
from keras import layers
import jax

(training_data, training_label), (test_data, test_label) = keras.datasets.mnist.load_data()
training_data = training_data.reshape(60000, 28 * 28).astype('float32') / 255
test_data = test_data.reshape(10000, 28 * 28).astype('float32') / 255
training_label = jax.nn.one_hot(training_label, 10)
test_label = jax.nn.one_hot(test_label, 10)


model = models.Sequential(
    [
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
)

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.categorical_accuracy]
)

model.fit(
    training_data,
    training_label,
    epochs=8,
    batch_size=128,
    validation_data=(test_data, test_label)
)

# Save the trained model
model.save('basic_mnist_model.keras')

