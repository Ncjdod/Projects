import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp
from functools import partial

# ===================================================================
# PART 1: Your Generic Second-Order ODE Solver Class
# ===================================================================
class SecondOrderODESolver:
    """
    A generic, reusable solver for systems of second-order ODEs.
    """
    def __init__(self, eom_func, initial_position, initial_velocity):
        self.eom_func = eom_func
        self.y0 = np.asarray(initial_position)
        self.v0 = np.asarray(initial_velocity)
        
        if self.y0.shape != self.v0.shape:
            raise ValueError("Initial position and velocity must have the same shape.")
            
        self.num_dims = len(self.y0)
        self.t = None
        self.y = None
        self.v = None

    def _first_order_system(self, t, Y):
        """
        Internal method to convert the second-order system into a
        first-order system that SciPy's solve_ivp can handle.
        """
        current_position = Y[:self.num_dims]
        current_velocity = Y[self.num_dims:]

        # Call the user-provided physics function
        accelerations = self.eom_func(t, current_position, current_velocity)
        
        return np.concatenate([current_velocity, np.asarray(accelerations)])

    def solve(self, t_span, t_eval=None, **kwargs):
        """
        Solves the differential equation.
        """
        # Prepare the initial state for SciPy's first-order solver
        initial_state_flat = np.concatenate([self.y0, self.v0])
        
        # Call the powerful SciPy solver
        solution = solve_ivp(
            fun=self._first_order_system,
            t_span=t_span,
            y0=initial_state_flat,
            t_eval=t_eval,
            **kwargs, method='LSODA'  # Pass any extra arguments (like tolerances)
        )
        
        # Unpack the results into a user-friendly format
        self.t = solution.t
        self.y = solution.y[:self.num_dims, :]
        self.v = solution.y[self.num_dims:, :]
        
        return self.t, self.y, self.v

    def animate(self, setup_plot_func, update_frame_func, interval=30, **kwargs):
        """
        Creates and shows an animation of the solved system.
        """
        if self.t is None:
            raise RuntimeError("You must run the 'solve()' method before animating.")

        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Call the user's setup function to get the plot elements
        plot_elements = setup_plot_func(ax)

        # The core animation update loop
        def animation_update(i):
            current_positions = self.y[:, i]
            # Call the user's update function for the current frame
            return update_frame_func(i, current_positions, plot_elements)

        print("Creating animation...")
        # Assign the animation to a variable to prevent it from being garbage-collected
        ani = animation.FuncAnimation(
            fig,
            animation_update,
            frames=len(self.t),
            interval=interval,
            blit=True,
            **kwargs
        )
        plt.show()
        print("Animation finished.")

# ===================================================================
# PART 2: The Specific Physics and Visualization for Our Problem
# ===================================================================
def swinging_atwood_eom(t, position, velocity):
    """
    Defines the EOMs for the Swinging Atwood Machine with robust constraints.
    """
    y, theta = position
    y_dot, theta_dot = velocity
    
    m1, m2, L, g = 1.0, 1.0, 20.0, 9.81
    
    # --- Calculate accelerations using the standard formulas first ---
    pendulum_length = L - y
    # Prevent division by zero if y is exactly L
    if pendulum_length < 1e-6: pendulum_length = 1e-6 

    y_ddot = (g * (m1 - m2 * np.cos(theta)) - m2 * pendulum_length * theta_dot**2) / (m1 + m2)
    theta_ddot = (2 * y_dot * theta_dot - g * np.sin(theta)) / pendulum_length

    # --- Now, CHECK FOR CONSTRAINTS and OVERRIDE accelerations if needed ---
    stiffness = 1e6
    damping = 1e3
    
    # Constraint 1: Prevent m1 from falling too far (y >= L)
    y_max_boundary = L - 1e-6 
    if y > y_max_boundary:
        displacement = y - y_max_boundary
        constraint_force = -stiffness * displacement - damping * y_dot
        # Override y_ddot with the strong restoring force
        y_ddot += constraint_force / (m1 + m2)
        # *** THE CRITICAL FIX ***
        # If pendulum has no length, it cannot have angular acceleration.
        theta_ddot = 0.0

    # Constraint 2: Prevent m1 from hitting the pulley (y <= 0)
    y_min_boundary = 1e-6
    if y < y_min_boundary:
        displacement = y - y_min_boundary
        constraint_force = -stiffness * displacement - damping * y_dot
        # Override y_ddot with the strong restoring force
        y_ddot += constraint_force / (m1 + m2)
        # The pendulum dynamics are still valid here, so no theta_ddot override is needed.

    return np.array([y_ddot, theta_ddot])

def setup_atwood_plot(ax):
    """
    Sets up the plot axes and initializes the plot elements for the animation.
    """
    L = 20.0
    ax.set_xlim(-L*1.1, L*1.1)
    ax.set_ylim(-L*1.2, L*0.2)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("Swinging Atwood Machine")
    
    pulley, = ax.plot(0, 0, 'ko', markersize=8, zorder=10)
    string, = ax.plot([], [], 'k-', lw=1.5)
    mass1, = ax.plot([], [], 'bs', markersize=12)
    mass2, = ax.plot([], [], 'ro', markersize=12)
    trace, = ax.plot([], [], 'r:', lw=1, alpha=0.5)
    
    return string, mass1, mass2, trace

def update_atwood_frame(i, current_positions, elements, full_position_history):
    """
    Updates the plot elements for a single frame of the animation.
    """
    y_i, theta_i = current_positions
    string, mass1, mass2, trace = elements
    L = 20.0

    y_history, theta_history = full_position_history
    
    x1, y1 = 0, -y_i
    x2 = (L - y_i) * np.sin(theta_i)
    y2 = -(L - y_i) * np.cos(theta_i)
    
    string.set_data([x1, 0, x2], [y1, 0, y2])
    mass1.set_data([x1], [y1])
    mass2.set_data([x2], [y2])
    
    trace_x_data = (L - y_history[:i+1]) * np.sin(theta_history[:i+1])
    trace_y_data = -(L - y_history[:i+1]) * np.cos(theta_history[:i+1])
    trace.set_data(trace_x_data, trace_y_data)
    
    return string, mass1, mass2, trace

# ===================================================================
# PART 3: The Main Script to Run the Simulation
# ===================================================================
if __name__ == '__main__':
    # --- SETUP AND RUN ---
    # We can now go back to the original unstable starting position
    # because our constraint will handle it correctly.
    initial_pos = [10.0, np.pi/4]
    initial_vel = [0.0, 0.0]
    t_final = 40.0
    dt = 0.03
    t_points = np.arange(0, t_final, dt)

    # 1. Create the solver instance
    solver = SecondOrderODESolver(swinging_atwood_eom, initial_pos, initial_vel)

    # 2. Solve the system with stricter tolerances for better stability
    t, y_t, v_t = solver.solve(
        t_span=(0, t_final), 
        t_eval=t_points,
        rtol=1e-6, 
        atol=1e-9
    )

    # 3. Animate the Results
    # Use 'functools.partial' to create a simpler update function
    # that already has the historical data "baked in".
    update_func_with_history = partial(update_atwood_frame, full_position_history=y_t)
    
    solver.animate(
        setup_plot_func=setup_atwood_plot, 
        update_frame_func=update_func_with_history, 
        interval=dt*10
    )
