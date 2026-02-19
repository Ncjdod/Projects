"""
HH Neural ODE Model

Architecture:
  - Fixed Fourier Features (non-trainable) for spectral learning
  - 4-layer MLP (64 neurons each) with tanh activation
  - Diffrax adaptive integration

Built on: Equinox + Diffrax + Optax (pure JAX stack)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from functools import partial

# ============================================================
# Fixed Fourier Features
# ============================================================
class FourierFeatures(eqx.Module):
    """
    Fixed (non-trainable) random Fourier feature encoding.
    
    Maps input x -> [sin(2*pi*B*x), cos(2*pi*B*x)]
    where B is a fixed random matrix sampled from N(0, sigma^2).
    
    This helps the network learn multi-scale dynamics by providing
    a rich spectral basis as input features.
    """
    B: jnp.ndarray  # Fixed frequency matrix (non-trainable)
    
    def __init__(self, input_dim, n_features, sigma=1.0, *, key):
        """
        Args:
            input_dim:  Dimension of input (e.g., 3 for [t, V, I_ext])
            n_features: Number of Fourier basis functions
            sigma:      Std dev of frequency sampling (controls scale)
            key:        JAX PRNG key
        """
        self.B = jax.random.normal(key, (input_dim, n_features)) * sigma
    
    def __call__(self, x):
        """
        Args:
            x: Input array of shape (input_dim,)
        Returns:
            Fourier features of shape (2 * n_features,)
        """
        projection = 2.0 * jnp.pi * x @ self.B  # (n_features,)
        return jnp.concatenate([jnp.sin(projection), jnp.cos(projection)])


# ============================================================
# HH Neural ODE Model
# ============================================================
class HHNeuralODE(eqx.Module):
    """
    Neural ODE for learning Hodgkin-Huxley voltage dynamics.
    
    Architecture:
        Input: [t, V, I_ext] -> FourierFeatures -> 4x(Linear(64) + tanh) -> Linear(1) -> dV/dt
    
    The model learns: dV/dt = f_net(t, V, I_ext)
    """
    fourier: FourierFeatures
    layers: list
    output_layer: eqx.nn.Linear
    
    def __init__(self, n_fourier=32, sigma=1.0, *, key):
        """
        Args:
            n_fourier: Number of Fourier basis functions (output dim = 2*n_fourier)
            sigma:     Fourier frequency scale
            key:       JAX PRNG key
        """
        keys = jax.random.split(key, 6)
        
        input_dim = 3  # [t, V, I_ext]
        fourier_out_dim = 2 * n_fourier  # sin + cos
        
        # Fixed Fourier features
        self.fourier = FourierFeatures(input_dim, n_fourier, sigma=sigma, key=keys[0])
        
        # 4 hidden layers, 64 neurons each
        self.layers = [
            eqx.nn.Linear(fourier_out_dim, 64, key=keys[1]),
            eqx.nn.Linear(64, 64, key=keys[2]),
            eqx.nn.Linear(64, 64, key=keys[3]),
            eqx.nn.Linear(64, 64, key=keys[4]),
        ]
        
        # Output: dV/dt (scalar)
        self.output_layer = eqx.nn.Linear(64, 1, key=keys[5])
    
    def __call__(self, t, y, I_ext):
        """
        Compute dV/dt given current state.
        
        Args:
            t:     Current time (scalar)
            y:     Current state [V] (shape: (1,))
            I_ext: External current at time t (scalar)
        
        Returns:
            dy/dt: Time derivative [dV/dt] (shape: (1,))
        """
        # Concatenate inputs: [t, V, I_ext]
        x = jnp.concatenate([jnp.array([t]), y, jnp.array([I_ext])])
        
        # Fourier encoding
        x = self.fourier(x)
        
        # 4-layer MLP with tanh
        for layer in self.layers:
            x = jnp.tanh(layer(x))
        
        # Output
        return self.output_layer(x)


# ============================================================
# ODE Integration (Diffrax)
# ============================================================
def make_diffrax_term(model, I_ext_fn):
    """
    Create a diffrax ODETerm from the model.
    
    Args:
        model:     HHNeuralODE instance
        I_ext_fn:  Function t -> I_ext (external current at time t)
    
    Returns:
        diffrax.ODETerm
    """
    def vector_field(t, y, args):
        I_ext = I_ext_fn(t)
        return model(t, y, I_ext)
    
    return diffrax.ODETerm(vector_field)


def integrate(model, y0, t_span, I_ext_fn, dt0=0.01, solver=None, 
              rtol=1e-3, atol=1e-5):
    """
    Integrate the Neural ODE forward in time.
    
    Args:
        model:     HHNeuralODE instance
        y0:        Initial state [V0] (shape: (1,))
        t_span:    Array of output times (shape: (n_steps,))
        I_ext_fn:  Function t -> I_ext
        dt0:       Initial step size
        solver:    Diffrax solver (default: Tsit5)
        rtol, atol: Tolerances for adaptive stepping
    
    Returns:
        ys: Trajectory of shape (n_steps, 1)
    """
    if solver is None:
        solver = diffrax.Tsit5()
    
    term = make_diffrax_term(model, I_ext_fn)
    
    saveat = diffrax.SaveAt(ts=t_span)
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
    
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[-1],
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=16384,
        throw=False,
    )
    
    return sol.ys  # (n_steps, 1)


# ============================================================
# Loss Function (PLACEHOLDER)
# ============================================================
def loss_fn(model, y0, t_span, I_ext_fn, y_target):
    """
    PLACEHOLDER: Loss function to be defined by user.
    
    Args:
        model:     HHNeuralODE instance
        y0:        Initial state [V0]
        t_span:    Time points
        I_ext_fn:  External current function
        y_target:  Target voltage trajectory
    
    Returns:
        loss: Scalar loss value
    """
    # TODO: User will define this
    y_pred = integrate(model, y0, t_span, I_ext_fn)
    loss = jnp.mean((y_pred[:, 0] - y_target) ** 2)  # Placeholder MSE
    return loss


# ============================================================
# Training Step (PLACEHOLDER)
# ============================================================
def train_step(model, opt_state, optimizer, y0, t_span, I_ext_fn, y_target):
    """
    PLACEHOLDER: Single training step.
    
    Args:
        model:     HHNeuralODE instance
        opt_state: Optax optimizer state
        optimizer: Optax optimizer
        y0:        Initial state
        t_span:    Time points
        I_ext_fn:  External current function
        y_target:  Target voltage trajectory
    
    Returns:
        model, opt_state, loss
    """
    # TODO: User will define this
    loss, grads = eqx.filter_value_and_grad(loss_fn)(
        model, y0, t_span, I_ext_fn, y_target
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# ============================================================
# Model Factory
# ============================================================
def create_model(key=None, n_fourier=32, sigma=1.0):
    """
    Create a new HHNeuralODE model.
    
    Args:
        key:       JAX PRNG key (default: random seed 42)
        n_fourier: Number of Fourier basis functions
        sigma:     Fourier frequency scale
    
    Returns:
        model: HHNeuralODE instance
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    model = HHNeuralODE(n_fourier=n_fourier, sigma=sigma, key=key)
    return model


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    print("HH Neural ODE - Architecture Test")
    print("=" * 50)
    
    # Create model
    key = jax.random.PRNGKey(0)
    model = create_model(key=key, n_fourier=32, sigma=1.0)
    
    # Count parameters
    params = eqx.filter(model, eqx.is_array)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Total parameters: {n_params}")
    
    # Test forward pass
    t = 0.0
    y = jnp.array([0.0])  # V = 0 mV
    I_ext = 0.0
    
    dy = model(t, y, I_ext)
    print(f"Input:  t={t}, V={y[0]:.2f}, I_ext={I_ext}")
    print(f"Output: dV/dt = {dy[0]:.6f}")
    
    # Test integration
    t_span = jnp.linspace(0.0, 10.0, 100)  # 10ms
    y0 = jnp.array([-65.0])  # Resting potential
    I_ext_fn = lambda t: 0.0  # No stimulus
    
    ys = integrate(model, y0, t_span, I_ext_fn)
    print(f"\nIntegration test:")
    print(f"  t: [0, 10] ms, {len(t_span)} points")
    print(f"  V(0)  = {ys[0, 0]:.2f} mV")
    print(f"  V(10) = {ys[-1, 0]:.2f} mV")
    print(f"  V range: [{ys[:, 0].min():.2f}, {ys[:, 0].max():.2f}]")
    
    print("\nArchitecture OK!")
