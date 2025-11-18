import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp


class SecondOrderODESolver:
    def __init__(self, eom_func, initial_position, initial_velocity):
        self.eom_func = eom_func
        self.y0 = np.asarray(initial_position)
        self.v0 = np.asarray(initial_velocity)
        self.Y0 = np.concatenate((self.y0.flatten(), 
                                  self.v0.flatten()))
        self.num_dims = len(self.y0)
        self.t = None
        self.y = None
        self.v = None
    
    def first_order_ODE(self, t, Y):
        current_position = Y[:self.num_dims]
        current_velocity = Y[self.num_dims:]

        accelerations = self.eom_func(t, current_position, current_velocity)
        dY_dt = np.concatenate((current_velocity.flatten(), 
                                        accelerations.flatten()))
        return dY_dt
    

    def solve(self, t_span, t_eval=None, **kwargs):
        solution = solve_ivp(self.first_order_ODE, t_span, self.Y0, t_eval=t_eval, **kwargs)
        self.t = solution.t
        self.y = solution.y[:self.num_dims, :]
        self.v = solution.y[self.num_dims:, :]
        return self.t, self.y, self.v
    
        
    def animate(self, setup_plot_func, update_frame_func, interval=30, **kwargs):
        
        if self.t is None:
            raise RuntimeError("You must run the 'solve()' method before animating.")

        fig, ax = plt.subplots(figsize=(8, 8))
        
        plot_elements = setup_plot_func(ax)


        def animation_update(i):
            current_positions = self.y[:, i]
            return update_frame_func(i, current_positions, plot_elements)

        print("Creating animation...")

        self.ani = animation.FuncAnimation(
            fig,
            animation_update,
            frames=len(self.t),
            interval=interval,
            blit=True,
            **kwargs
        )
        plt.show()
        print("Animation finished.")
