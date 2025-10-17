import numpy as np
import sympy as sp
import pickle
import sys


def derive_cartpole_eom():
    x, theta, x_dot, theta_dot = sp.symbols('x theta x_dot theta_dot')
    m, M, l, g, F = sp.symbols('m M l g F')

    A = sp.Matrix([
        [M + m,        m * l * sp.cos(theta)],
        [m * l * sp.cos(theta), m * l**2]
    ])

    force_vector = sp.Matrix([
        [F + m * l * sp.sin(theta) * theta_dot**2],
        [m * g * l * sp.sin(theta)]
    ])

    solution = A.inv() * force_vector
    x_ddot = solution[0]
    theta_ddot = solution[1]

    input_symbols = [x, theta, x_dot, theta_dot, m, M, l, g, F]
    eom_func = sp.lambdify(input_symbols, sp.Matrix([x_ddot, theta_ddot]), 'numpy')

    return eom_func


class CartpoleEnv:
    def __init__(self, m=0.1, M=1.0, l=0.5, g=9.81):
        self.m = m
        self.M = M
        self.l = l
        self.g = g
        self.state = np.array([0.0, np.pi/12, 0.0, -0.2]) # [x, theta, x_dot, theta_dot]
        self.eom_func = derive_cartpole_eom()

    def step(self, F, dt):
        x, theta, x_dot, theta_dot = self.state
        accelerations = self.eom_func(x, theta, x_dot, theta_dot, self.m, self.M, self.l, self.g, F)
        acc = np.asarray(accelerations).flatten()
        x_ddot, theta_ddot = float(acc[0]), float(acc[1])
        x_dot += x_ddot * dt
        theta_dot += theta_ddot * dt
        x += x_dot * dt
        theta += theta_dot * dt

        self.state = np.array([x, theta, x_dot, theta_dot])
        return self.state

    def reset(self):
        self.state = np.zeros(4)
        return self.state
    
    def get_state(self):
        return self.state
    
    def set_state(self, new_state):
        self.state = new_state
        return self.state
    
def main():
    env = CartpoleEnv()
    dt = 0.02
    pickle.dump({
        'state': env.get_state().tolist(),
        'm': env.m,
        'M': env.M,
        'l': env.l,
        'g': env.g,
    }, sys.stdout.buffer)
    sys.stdout.flush()

    while True:
        try:
            command_dict = pickle.load(sys.stdin.buffer)
            command = command_dict.get('command')

            if command == 'step':
                force = command_dict.get('force', 0.0)
                env.step(force, dt)
            elif command == 'quit':
                break

            pickle.dump({
                'state': env.get_state().tolist()
            }, sys.stdout.buffer)
            sys.stdout.flush()

        except EOFError:
            break
        except Exception as e:
            print(f"Cartpole environment error: {e}", file=sys.stderr)
            break

if __name__ == '__main__':
    main()