import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Import the custom model class and helpers
from NeuralODE import NeuralODE, integrate_with_scan, fhn_dynamics

# --- Model parameters (must match training) ---
T = 50.0
steps = 200
dt = T / (steps - 1)
n_steps = steps - 1
t_span = jnp.linspace(0.0, T, steps)

# --- Load the trained model ---
print("Loading trained model...")

# Method 1: Direct load (works if model was saved with decorator)
try:
    model = keras.models.load_model('NeuralODE.keras')
    print("Model loaded directly!")
except TypeError:
    # Method 2: Reconstruct model and load weights
    print("Direct load failed. Reconstructing model and loading weights...")
    model = NeuralODE(dt=dt, n_steps=n_steps)
    # Build the model first by passing dummy input
    dummy_input = jnp.zeros((1, 2), dtype=jnp.float32)
    _ = model(dummy_input)
    model.load_weights('NeuralODE.keras')
    print("Weights loaded successfully!")

# --- Create a new initial condition (unseen during training) ---
y0_test = jnp.array([[0.5, 0.2]], dtype=jnp.float32)
print(f"Test initial condition: v={y0_test[0, 0]:.2f}, w={y0_test[0, 1]:.2f}")

# --- Run inference ---
pred_y = model.predict(y0_test)
print(f"Prediction shape: {pred_y.shape}")

# --- Generate ground truth for comparison ---
y_true_transposed = integrate_with_scan(fhn_dynamics, y0_test, dt, steps - 1)
y_true = jnp.transpose(y_true_transposed, (1, 0, 2))

# --- Visualization ---
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Phase Portrait
ax[0].plot(y_true[0, :, 0], y_true[0, :, 1], 'g--', linewidth=2, label="True FHN")
ax[0].plot(pred_y[0, :, 0], pred_y[0, :, 1], 'b-', linewidth=2, label="Neural ODE")
ax[0].scatter([y0_test[0, 0]], [y0_test[0, 1]], color='red', s=100, zorder=5, label="Initial Condition")
ax[0].set_xlabel('Voltage (v)')
ax[0].set_ylabel('Recovery (w)')
ax[0].set_title('Phase Portrait (Test IC)')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# Time Series
ax[1].plot(t_span, y_true[0, :, 0], 'g--', linewidth=2, label="v True")
ax[1].plot(t_span, pred_y[0, :, 0], 'b-', linewidth=2, label="v Pred")
ax[1].plot(t_span, y_true[0, :, 1], 'g:', linewidth=2, label="w True")
ax[1].plot(t_span, pred_y[0, :, 1], 'b:', linewidth=2, label="w Pred")
ax[1].set_xlabel('Time')
ax[1].set_ylabel('State')
ax[1].set_title('Time Series (Test IC)')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("inference_result.png")
print("Saved inference_result.png")
plt.show()
