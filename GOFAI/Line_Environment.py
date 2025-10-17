import numpy as np
import sys
import pickle

class Environment:
    def __init__(self, box_size, masses):
        """
        A 1D environment for multiple particles, optimized with NumPy.
        """
        self.box_half_size = box_size / 2
        num_particles = len(masses)
        
        self.masses = np.asarray(masses, dtype=float).reshape(num_particles, 1)
        self.positions = np.zeros((num_particles, 1), dtype=float)
        self.velocities = np.zeros((num_particles, 1), dtype=float)
        
        self.forces = np.zeros((num_particles, 1), dtype=float)

    def apply_force(self, force_vector):
        """Applies a vector of forces to the particles."""
        self.forces = np.asarray(force_vector, dtype=float).reshape(-1, 1)

    def run_step(self, dt):
        """Runs one step of the physics simulation."""
        accelerations = self.forces / self.masses
        self.velocities += accelerations * dt
        self.positions += self.velocities * dt

        is_too_low = self.positions < -self.box_half_size
        is_too_high = self.positions > self.box_half_size
        
        self.positions[is_too_low] = -self.box_half_size
        self.positions[is_too_high] = self.box_half_size
        
        collided = is_too_low | is_too_high
        self.velocities[collided] *= -1

        self.forces.fill(0.0)

    def __getstate__(self):
        """
        Returns a dictionary of the essential state needed to save the object.
        This is called by `pickle`.
        """
        return self.__dict__.copy()


    def __setstate__(self, state):
        """Restores the object's state from the dictionary provided by `unpickle`."""
        self.__dict__.update(state)


def main():
    env = Environment(20, [1, 1])
    dt = 0.1
    pickle.dump(env, sys.stdout.buffer)
    sys.stdout.flush()

    while True:
        try:
            # --- THE FIX: PART 2 ---
            # LISTEN FIRST: Wait for a command dictionary from the agent.
            command_dict = pickle.load(sys.stdin.buffer)
            
            command = command_dict.get('command')

            if command == 'push':
                force_vector = command_dict.get('force_vector', [])
                env.apply_force(force_vector)

            elif command == 'quit':
                break
            
            # UPDATE: Run the physics based on the command.
            env.run_step(dt)

            # --- THE FIX: PART 3 ---
            # SPEAK SECOND: Send the NEW state back to the agent as a reply.
            pickle.dump(env, sys.stdout.buffer)
            sys.stdout.flush()

        except EOFError:
            # This happens when the agent closes the pipe.
            break

if __name__ == '__main__':
    main()
