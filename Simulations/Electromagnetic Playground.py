import numpy as np
import scipy
import sympy as sp


class Particle:
    def __init__(self, mass, charge, position, velocity):
        self.mass = float(mass)
        self.charge = float(charge)
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)


class Fixed_charge:
    def __init__(self, charge, position):
        self.charge = charge
        self.position = position


class Universe:
    def __init__(self, box_size, masses, charges):
        self.x_dim = box_size[0]/2
        self.y_dim = box_size[1]/2
        self.z_dim = box_size[2]/2
        self.particles = [Particle(m, c, [0, 0, 0], [0, 0, 0]) for m, c in zip(masses, charges)]

        num_particles = len(self.particles)
        self.forces_ij = np.zeros((num_particles, num_particles, 3))
        

    def electrical_force(self):
        """Calculates all pairwise electrical forces in a fully vectorized manner."""
        num_particles = len(self.particles)
        positions = np.array([p.position for p in self.particles])  # Shape: (N, 3)
        charges = np.array([p.charge for p in self.particles]).reshape(num_particles, 1)  # Shape: (N, 1)

        r_vectors = positions[:, None, :] - positions[None, :, :]
        r_distances = np.linalg.norm(r_vectors, axis=2) + 1e-9

        charge_products = charges @ charges.T

        k_e = 1 / (4 * np.pi * scipy.constants.epsilon_0)
        self.forces_ij = k_e * (charge_products / r_distances**3)[:, :, None] * r_vectors

    def run_step(self, dt):
        self.electrical_force()

        net_forces = np.sum(self.forces_ij, axis=1)

        num_particles = len(self.particles)
        masses = np.array([p.mass for p in self.particles]).reshape(num_particles, 1)
        current_positions = np.array([p.position for p in self.particles])
        current_velocities = np.array([p.velocity for p in self.particles])

        accelerations = net_forces / masses
        new_velocities = current_velocities + accelerations * dt
        new_positions = current_positions + new_velocities * dt

        r_vectors = new_positions[None, :, :] - new_positions[:, None, :]
        r_distances = np.linalg.norm(r_vectors, axis=2)

        np.fill_diagonal(r_distances, np.inf)
        is_too_close_matrix = r_distances < 1e-6 

        colliding_particles_mask = np.any(is_too_close_matrix, axis=1)

        out_of_bounds_mask = [self.x_dim, self.y_dim, self.z_dim] < np.abs(new_positions)
        out_of_bounds_mask = np.any(out_of_bounds_mask, axis=1)
        colliding_particles_mask |= out_of_bounds_mask

        for i, particle in enumerate(self.particles):
            if not colliding_particles_mask[i]:
                particle.position = new_positions[i]
                particle.velocity = new_velocities[i]
            elif colliding_particles_mask[i]:
                particle.position = current_positions[i]
                particle.velocity = -current_velocities[i]
    
    def animate(self, dt, frames=200, interval=50):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        particle_plots = [ax.plot([], [], [], 'o', markersize=5, color='blue')[0] for _ in self.particles]

        def update(frame):
            self.run_step(dt)
            for i, particle in enumerate(self.particles):
                pos = particle.position
                particle_plots[i].set_data([pos[0]], [pos[1]])
                particle_plots[i].set_3d_properties(zs=[pos[2]])
            return particle_plots

        def setup_plot():
            ax.set_xlim([-self.x_dim, self.x_dim])
            ax.set_ylim([-self.y_dim, self.y_dim])
            ax.set_zlim([-self.z_dim, self.z_dim])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Electromagnetic Particle Simulation')
            for i, particle in enumerate(self.particles):
                pos = particle.position
                particle_plots[i].set_data([pos[0]], [pos[1]])
                particle_plots[i].set_3d_properties(zs=[pos[2]])
            return particle_plots

        ani = FuncAnimation(fig, update, frames=frames, init_func=setup_plot, blit=False, interval=1)
        plt.show()


def main():
    uni = Universe([50, 50, 50], [100, 2], [-1*10**-4, 5*10**-6])
    dt = 0.05
    uni.particles[0].position = np.array([0.0, 0.0, 0.0])
    uni.particles[1].position = np.array([-10.0, 0.0, 0.0])
    uni.particles[0].velocity = np.array([0.0, 0.0, 0.0])
    uni.particles[1].velocity = np.array([0.0, 0.6, 0.0])
    uni.animate(dt=0.05)
    

if __name__ == '__main__':
    main()