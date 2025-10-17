import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def func(x, L):
    psi = 5*np.sin(4*np.pi * x/L)
    return psi


def initial_state(func, N, L):
    x = np.linspace(0, L, N)
    initial_state_vector = func(x, L)
    return initial_state_vector


def wave_mechanics(initial_state, num_steps, total_time, L, c):
    N = initial_state.size
    dx = L/N
    dt = total_time/num_steps

    CFL_factor = (c * dt/dx)**2
    y = np.zeros((N , num_steps))
    y[:, 0] = initial_state

    for j in range(1, num_steps - 1):    
        y[1:-1, j+1] = 2 * y[1:-1, j] - y[1:-1, j - 1] - CFL_factor * (2 * y[1:-1, j] - y[2:, j] - y[:-2, j])

    x = np.linspace(0, L, N)
    t = np.linspace(0, total_time, num_steps)

    return t, x, y

def animate_solution(x, t, y, interval=20, repeat=True, figsize=(8,4)):
    """
    Animate 1D wave solution.
    x: 1D array (N,)
    t: 1D array (num_steps,)
    y: 2D array shape (N, num_steps)
    """
    fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot(x, y[:, 0], lw=2)
    ax.set_xlim(x.min(), x.max())
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    pad = 0.1 * max(abs(ymin), abs(ymax), 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")

    def update(frame):
        line.set_ydata(y[:, frame])
        return (line,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(y.shape[1]),
        interval=interval,
        blit=True,
        repeat=repeat,
    )
    return ani

if __name__ == "__main__":
    L = 1.0
    N = 100
    total_time = 2.0
    num_steps = 4000
    c = 20.0

    init = initial_state(func, N, L)
    t, x, y = wave_mechanics(init, num_steps, total_time, L, c)

    ani = animate_solution(x, t, y, interval=20)
    plt.show()







