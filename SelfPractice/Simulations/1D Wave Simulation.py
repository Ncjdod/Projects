import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def state_creation(N):
  sample = np.zeros(N)
  start_index = int(N/2-N/5)
  end_index = int(N/2+N/5)
  sine_bump = 0.5 * np.sin(np.linspace(0, 3*np.pi, end_index - start_index))
  sample[start_index:end_index] = sine_bump
  mu = N/2
  sigma = N/20
  x = np.arange(N)
  amplitude = 0.5
  envelope = amplitude * np.exp(-(x-mu)**2 / (2*sigma**2))   

  k = 2 * np.pi * 5 / (4 * sigma)
  carrier_wave = np.sin(k * (x - mu))

  wave_packet = envelope * carrier_wave
  current_state = wave_packet
  previous_state = 1.02 * current_state.copy()
  next_state = np.zeros(N)
  return current_state, previous_state, next_state


def wave_simulation_vectorized(current_state, previous_state, num_steps, cfl_factor_sq, N):
    simulation_history = [current_state.copy()]
    
    for _ in range(num_steps):
        laplacian = current_state[2:] - 2 * current_state[1:-1] + current_state[:-2]
        
        next_state_interior = cfl_factor_sq * laplacian + 2 * current_state[1:-1] - previous_state[1:-1]
        
        next_state = np.zeros(N)
        next_state[1:-1] = next_state_interior
        
        previous_state = current_state.copy()
        current_state = next_state.copy()
        simulation_history.append(current_state.copy())
        
    return simulation_history

def update(frame_number):
  y_data = history_array[frame_number]
  line.set_ydata(y_data)
  return line,

# --- Simulation Parameters ---
## FIX 1: Reduce the number of steps to a reasonable number for quick visualization.
num_steps = 3000
delta_x = 0.1
N = 500  # Number of points on the rope
c = 1.0  # Wave speed

# Ensure the CFL condition (c * dt / dx <= 1) is met for stability
dt = 0.05
if (c * dt / delta_x) > 1:
    print("Warning: CFL condition not met! Simulation may be unstable.")
    dt = delta_x / c
    print(f"Adjusting dt to {dt:.4f} for stability.")

cfl_factor_sq = (c * dt / delta_x)**2

# --- Setup and Run Simulation ---
current_state, previous_state, next_state = state_creation(N)

# Call the new, faster vectorized function
history_list = wave_simulation_vectorized(current_state, previous_state, num_steps, cfl_factor_sq, N)
history_array = np.array(history_list)

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlabel('Position')
ax.set_ylabel('Amplitude')
ax.set_title('1D Wave Simulation (Vectorized)')
ax.set_facecolor('#222222') # Dark background for better contrast
fig.set_facecolor('#222222')

# Set fixed y-limits for a stable animation view
max_amplitude = np.max(np.abs(history_array))
ax.set_ylim(-max_amplitude * 1.1, max_amplitude * 1.1)

# Style the plot
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white') 
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')
ax.title.set_color('white')

line, = ax.plot(history_array[0], color='cyan')
ani = animation.FuncAnimation(fig, update, frames=len(history_array), interval=10, blit=True)

plt.show()
