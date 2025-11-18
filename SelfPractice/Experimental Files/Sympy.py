import numpy as np
import sympy as sp

# --- 1. Define Symbols and Derive Equations of Motion ---
x, theta, x_dot, theta_dot = sp.symbols('x theta x_dot theta_dot')
m, M, l, g, F = sp.symbols('m M l g F')

# Mass Matrix
A = sp.Matrix([[m + M, m * l * sp.cos(theta)],
               [sp.cos(theta), l]
               ])

# Force Vector
force_vector = sp.Matrix([
    [F + m * l * sp.sin(theta) * theta_dot**2],
    [g * sp.sin(theta)]
])

# Solve for accelerations symbolically
solution = A.inv() * force_vector
x_ddot = solution[0]
theta_ddot = solution[1]

print("--- Symbolic Equations Derived ---")
print("x_ddot =")
sp.pprint(x_ddot)
print("\ntheta_ddot =")
sp.pprint(theta_ddot)
print("-" * 40)

# --- 2. Bridge to NumPy with lambdify ---
# This is the key step to make the symbolic equations usable for numerical simulation.

# List all the symbols that will be inputs to our numerical function.
# The order matters! This will be the order of arguments for the function.
input_symbols = [x, theta, x_dot, theta_dot, m, M, l, g, F]

# Create the fast, numerical function. 'numpy' tells it to use numpy functions.
# We combine the two outputs into a single matrix for efficiency.
eom_func = sp.lambdify(input_symbols, sp.Matrix([x_ddot, theta_ddot]), 'numpy')

# --- 3. Test the Numerical Function ---
print("\n--- Testing the Numerical Function ---")
# Define some concrete numerical values for the system's state and parameters
state_and_params = {
    'x': 0.0, 'theta': np.pi / 4, 'x_dot': 0.5, 'theta_dot': -0.2, # Current state
    'm': 0.1, 'M': 1.0, 'l': 0.5, 'g': 9.81, 'F': 10.0            # System parameters
}

# Call the function with these values. Note the order must match `input_symbols`.
numerical_accelerations = eom_func(**state_and_params)

print(f"Input State: theta={state_and_params['theta']:.2f} rad, F={state_and_params['F']:.1f} N")
print(f"Calculated x_ddot: {numerical_accelerations[0][0]:.4f} m/s^2")
print(f"Calculated theta_ddot: {numerical_accelerations[1][0]:.4f} rad/s^2")