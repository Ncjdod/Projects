import numpy as np
import sys

class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = float(mass)
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)

class Environment:
    def __init__(self, box_size, mass):
        self.box_size = box_size

        initial_position = np.array([box_size/2, box_size/2])
        initial_velocity = np.array([0, 0])
        self.particle = Particle(mass, initial_position, initial_velocity)

        self.force = np.array([0.0, 0.0])


    def apply_force(self, force_vector):
        self.force = np.asarray(force_vector, dtype=float)

    def run_step(self, dt):
        acceleration = self.force/self.particle.mass
        self.particle.velocity += acceleration * dt
        self.particle.position += self.particle.velocity * dt

        is_too_low = self.particle.position < 0.0
        is_too_high = self.particle.position > self.box_size        

        self.particle.position[is_too_low] = 0
        self.particle.position[is_too_high] = self.box_size
      
        out_of_bounds = is_too_low | is_too_high
        self.particle.velocity[out_of_bounds] *= -0.9

        self.force = np.array([0.0, 0.0])


    def get_state_string(self):
        state = f'x_pos={self.particle.position[0]:.2f}, y_pos={self.particle.position[1]:.2f}, x_vel={self.particle.velocity[0]:.2f}, y_vel={self.particle.velocity[1]:.2f}'
        return state
    

def main():
    env = Environment(20, 1)
    dt = 0.1

    while True:
        print(f'State: {env.get_state_string()}')
        sys.stdout.flush()

        command_line = input('dsfg')
        command_line_without_spaces = command_line.replace(' ', '')
        components = command_line_without_spaces.split(',')

        try:    
            if components[0] == 'push':
                fx = float(components[1])
                fy = float(components[2])
                env.apply_force(np.array([fx, fy]))

            elif components[0] == 'quit':
                break
        except (ValueError, IndexError):
            print('Invalid command. Try "push,fx,fy" or "quit"')

        env.run_step(dt)

if __name__ == '__main__':
    main()
