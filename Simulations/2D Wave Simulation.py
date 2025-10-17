import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting

def state_creation(N, v_0):
  sample = np.zeros((N, N))
  mu = N/2
  sigma = N/20

  x_coords = np.arange(N)
  y_coords = np.arange(N)
  X, Y = np.meshgrid(x_coords, y_coords)

  R = np.sqrt((X - mu)**2 + (Y - mu)**2)

  amplitude = 0.5

  envelope = amplitude * np.exp(-R**2 / (2*sigma**2))

  # Increased k for more visible ripples
  k = 2 * np.pi * 5 / (4 * sigma)
  carrier_wave = np.sin(k * R)

  wave_packet = envelope * carrier_wave
  current_state = wave_packet
  previous_state = v_0 * current_state.copy()
  next_state = np.zeros_like(sample)
  return current_state, previous_state, next_state

def wave_simulation_vectorized(current_state, previous_state, next_state, num_steps, cfl_factor_sq):
  simulation_history = [current_state.copy()]
  for _ in range(num_steps):
    laplacian = current_state[2:, 1:-1] + current_state[0:-2, 1:-1] + current_state[1:-1, 2:] + current_state[1:-1, 0:-2] - 4 * current_state[1:-1, 1:-1]
    next_state[1:-1, 1:-1] = cfl_factor_sq * laplacian + 2 * current_state[1:-1, 1:-1] - previous_state[1:-1, 1:-1]

    # CRITICAL FIX: Update previous_state BEFORE current_state is updated.
    previous_state = current_state.copy()
    current_state = next_state.copy()
    simulation_history.append(next_state.copy())
  return simulation_history


def update(frame_number):
  # For 3D animation, it's common to clear the axes and redraw the surface
  ax.clear()
  Z = simulation_history[frame_number]
  # Redraw the surface for the current frame
  surf = ax.plot_surface(X, Y, Z, cmap='viridis', vmin=-max_amplitude, vmax=max_amplitude)
  # Reset the z-axis limit and title, as clear() removes them
  ax.set_zlim(-max_amplitude * 1.1, max_amplitude * 1.1)
  ax.set_title('3D Wave Simulation (Vectorized)')
  return surf,

# --- Simulation Parameters (Adjusted for Performance) ---
num_steps = 300   # Reduced for smoother 3D animation
delta_x = 0.1
delta_t = 0.05
N = 150           # Reduced for smaller memory footprint and faster rendering
c = 1.0

# --- Stability (CFL) Condition for 2D ---
cfl_limit = 1 / np.sqrt(2)
if (c * delta_t / delta_x) > cfl_limit:
    print(f"Warning: CFL condition not met! ({(c * delta_t / delta_x):.2f} > {cfl_limit:.2f})")
    delta_t = cfl_limit * delta_x / c
    print(f"Adjusting dt to {delta_t:.4f} for stability.")

cfl_factor_sq = (c * delta_t / delta_x)**2

current_state, previous_state, next_state = state_creation(N, 1.02)

simulation_history = wave_simulation_vectorized(current_state, previous_state, next_state, num_steps, cfl_factor_sq)
simulation_array = np.array(simulation_history)

# --- Animation Setup ---
# Create X and Y coordinate grids for the 3D plot
x_coords = np.arange(N)
y_coords = np.arange(N)
X, Y = np.meshgrid(x_coords, y_coords)

# Create the 3D figure and axes
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Calculate a fixed amplitude for the color map and z-axis
max_amplitude = np.max(np.abs(simulation_array))

# Create the animation. blit=False is required for 3D animations that clear the axes.
ani = FuncAnimation(fig, update, frames=len(simulation_history), interval=10, blit=False)

plt.show()