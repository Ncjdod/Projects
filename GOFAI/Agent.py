import subprocess
import pickle
import sys
import random
import numpy as np
import os

# --- Best Practice: Define the path to the environment script ---
# This makes the script more robust and less dependent on the current working directory.
ENV_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'Line_Environment.py')
ENV_SCRIPT_PATH = os.path.abspath(ENV_SCRIPT_PATH) # Get the absolute path

if not os.path.exists(ENV_SCRIPT_PATH):
    print(f"Error: Environment script not found at '{ENV_SCRIPT_PATH}'", file=sys.stderr)
    sys.exit(1)

env_process = subprocess.Popen([sys.executable, ENV_SCRIPT_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

def main():
    try:
        while True:
            env_state = pickle.load(env_process.stdout)
            
            num_particles = len(env_state.positions)
            threshold = env_state.box_half_size / 2            
            if np.any(np.abs(env_state.positions) > threshold):
                command_choice = 'quit'
            else:
                command_choice = 'push'

            if command_choice == 'push':
                force_values = np.random.uniform(low=-1.0, high=1.0, size=(num_particles, 1))

                command_dict = {
                    'command': 'push',
                    'force_vector': force_values.tolist()
                }
            else: 
                command_dict = {'command': 'quit'}
            
            pickle.dump(command_dict, env_process.stdin)
            env_process.stdin.flush()

            if command_dict.get('command') == 'quit':
                print("Agent decided to quit. Shutting down.", file=sys.stderr)
                # Give the environment a moment to receive the 'quit' command and exit gracefully.
                try:
                    env_process.wait(timeout=1)
                except subprocess.TimeoutExpired: # This is expected if the process is still running
                    pass # We will terminate it in the finally block anyway.
                break

    except EOFError:
        print("Environment closed its pipe. Agent shutting down.", file=sys.stderr)
    finally:
        print("Agent cleanup: Terminating environment process.", file=sys.stderr)
        env_process.terminate() # Ensure the process is killed
        env_process.wait()      # Wait for the process to actually terminate

if __name__ == '__main__':
    main()
