import subprocess
import pickle
import sys
import os
import numpy as np

ENV_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'Cartpole Environment.py')
ENV_SCRIPT_PATH = os.path.abspath(ENV_SCRIPT_PATH)

if not os.path.exists(ENV_SCRIPT_PATH):
    print(f"Error: Environment script not found at '{ENV_SCRIPT_PATH}'", file=sys.stderr)
    sys.exit(1)

env_process = subprocess.Popen(
    [sys.executable, "-u", ENV_SCRIPT_PATH],  # "-u" => unbuffered stdio (important)
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, 
)

def applied_force(theta, theta_dot, K_p=70, K_d=0.1):
    """A simple PD controller to keep the pole balanced."""
    F = K_p * theta + K_d * theta_dot
    return F

def main():
    try:
        while True:
            env_snapshot = pickle.load(env_process.stdout)
            state = np.asarray(env_snapshot['state'])

            x, theta, x_dot, theta_dot = state
            

            if abs(theta) > 2:
                command_dict = {'command': 'quit'}
                print("Agent decided to quit. Shutting down.", file=sys.stderr)
            else:
                F = applied_force(theta, theta_dot)
                print(f"x={x:.2f}, x_dot={x_dot:.2f}", file=sys.stderr)
                print(f"theta={theta:.2f}, theta_dot={theta_dot:.2f}, F={F:.2f}", file=sys.stderr)
                print()
                command_dict = {
                'command': 'step',
                'force': F
                }
            
            pickle.dump(command_dict, env_process.stdin)
            env_process.stdin.flush()

            if abs(theta) < 10e-4 and abs(theta_dot) < 10e-4:
                break
    finally:
        print("Agent cleanup: Terminating environment process.", file=sys.stderr)
        env_process.terminate()
        env_process.wait()
        
        try:
            err = env_process.stderr.read().decode(errors='replace')
            if err:
                print("Env stderr:\n" + err, file=sys.stderr)
        except Exception:
            pass

if __name__ == '__main__':
    main()

