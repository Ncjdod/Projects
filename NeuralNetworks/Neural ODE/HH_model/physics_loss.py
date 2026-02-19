"""
Adversarial Physics Loss

Implements trainable loss weights updated via gradient ASCENT
(Self-Adaptive PINN). The weights automatically increase where 
the physics residual is largest, forcing the network to focus 
on the hardest parts of the dynamics.

Minimax formulation:
  L(theta, s) = sum_i exp(s_i) * R_i(theta) - sum_i s_i
  
  Model params theta: gradient DESCENT (minimize L)
  Loss weights s:     gradient ASCENT  (maximize L)
  
Reference: McClenny & Brainerd, "Self-Adaptive PINNs" (2020)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from HodgkinHuxley import HodgkinHuxley


# ============================================================
# Trainable Loss Weights
# ============================================================
class LossWeights(eqx.Module):
    """
    Trainable loss weights for adversarial physics training.
    
    Stores log-weights s_i. Actual weights are exp(s_i), 
    ensuring positivity. Updated via gradient ASCENT.
    """
    log_weights: jnp.ndarray  # (n_terms,)
    
    def __init__(self, n_terms, init_value=0.0):
        """
        Args:
            n_terms:    Number of loss terms to weight
            init_value: Initial log-weight (0.0 means weight=1.0)
        """
        self.log_weights = jnp.ones(n_terms) * init_value
    
    @property
    def weights(self):
        """Actual (positive) weights."""
        return jnp.exp(self.log_weights)
    
    def regularization(self):
        """
        Regularization term: -sum(s_i)
        Prevents weights from collapsing to zero.
        """
        return -jnp.sum(self.log_weights)


# ============================================================
# Physics Residual
# ============================================================
def physics_residual(model, hh, V_samples, t_samples, I_ext_samples):
    """
    Compute physics residual: how well does the neural ODE 
    match the HH equations at sampled state points.
    
    Since the model only predicts dV/dt and we don't have 
    gating variables, we use steady-state approximations:
      m ~ m_inf(V),  h ~ h_inf(V),  n ~ n_inf(V)
    
    Args:
        model:         HHNeuralODE instance
        hh:            HodgkinHuxley instance
        V_samples:     Voltage samples (N,)
        t_samples:     Time samples (N,)
        I_ext_samples: External current at those times (N,)
    
    Returns:
        residuals: Per-sample squared residual (N,)
    """
    def single_residual(V, t, I_ext):
        # Neural ODE prediction
        y = jnp.array([V])
        dVdt_net = model(t, y, I_ext)[0]
        
        # HH prediction (using steady-state gating variables)
        m = hh.m_inf(V)
        h = hh.h_inf(V)
        n = hh.n_inf(V)
        dVdt_hh = hh.dVdt(V, m, h, n, I_ext)
        
        return (dVdt_net - dVdt_hh) ** 2
    
    # Vectorize over samples
    residuals = jax.vmap(single_residual)(V_samples, t_samples, I_ext_samples)
    return residuals


# ============================================================
# Adversarial Loss
# ============================================================
def adversarial_physics_loss(model, loss_weights, hh,
                              V_samples, t_samples, I_ext_samples):
    """
    Self-Adaptive physics loss with trainable weights.
    
    L = sum_i exp(s_i) * R_i - sum_i s_i
    
    The -s_i regularization prevents weights from going to infinity.
    
    Args:
        model:         HHNeuralODE instance
        loss_weights:  LossWeights instance (trainable)
        hh:            HodgkinHuxley instance
        V_samples:     Voltage collocation points (N,)
        t_samples:     Time collocation points (N,)
        I_ext_samples: External current at collocation points (N,)
    
    Returns:
        total_loss: Scalar loss (to be minimized by model, maximized by weights)
        info:       Dict with diagnostic information
    """
    # Compute per-sample residuals
    residuals = physics_residual(model, hh, V_samples, t_samples, I_ext_samples)
    
    # Weighted loss: exp(s_i) * R_i
    weights = loss_weights.weights  # (N,) or broadcastable
    
    # If we have fewer weights than samples, broadcast
    # e.g., 1 weight per voltage region, or 1 global weight
    if weights.shape[0] < residuals.shape[0]:
        # Replicate weights to match samples
        n_per_weight = residuals.shape[0] // weights.shape[0]
        weights = jnp.repeat(weights, n_per_weight)[:residuals.shape[0]]
    
    weighted_residuals = weights * residuals
    
    # Total loss = weighted residuals - regularization
    loss = jnp.mean(weighted_residuals) + loss_weights.regularization()
    
    info = {
        'physics_loss': jnp.mean(residuals),
        'weighted_loss': jnp.mean(weighted_residuals),
        'mean_weight': jnp.mean(weights),
        'max_weight': jnp.max(weights),
        'min_weight': jnp.min(weights),
    }
    
    return loss, info


# ============================================================
# Full Loss (Data + Physics)
# ============================================================
def total_loss_fn(model, loss_weights, hh, 
                  y0, t_span, I_ext_fn, y_target,
                  V_colloc, t_colloc, I_colloc,
                  physics_weight=1.0):
    """
    PLACEHOLDER: Combined data + adversarial physics loss.
    
    L_total = L_data + physics_weight * L_physics_adversarial
    
    Args:
        model:          HHNeuralODE
        loss_weights:   LossWeights (trainable, gradient ascent)
        hh:             HodgkinHuxley (fixed)
        y0:             Initial voltage
        t_span:         Time points for trajectory
        I_ext_fn:       External current function
        y_target:       Target voltage from Allen Brain data
        V_colloc:       Voltage collocation points for physics
        t_colloc:       Time collocation points
        I_colloc:       Current at collocation points
        physics_weight: Overall physics loss scale
    
    Returns:
        loss, info
    """
    # --- Data loss (placeholder) ---
    # TODO: Integrate model and compute MSE against data
    # from HH_NeuralODE import integrate
    # y_pred = integrate(model, y0, t_span, I_ext_fn)
    # data_loss = jnp.mean((y_pred[:, 0] - y_target) ** 2)
    data_loss = 0.0  # PLACEHOLDER
    
    # --- Adversarial physics loss ---
    phys_loss, phys_info = adversarial_physics_loss(
        model, loss_weights, hh, V_colloc, t_colloc, I_colloc
    )
    
    total = data_loss + physics_weight * phys_loss
    
    info = {
        'total_loss': total,
        'data_loss': data_loss,
        **phys_info,
    }
    
    return total, info


# ============================================================
# Minimax Training Step
# ============================================================
def minimax_step(model, loss_weights, 
                 model_opt_state, weights_opt_state,
                 model_optimizer, weights_optimizer,
                 hh, V_colloc, t_colloc, I_colloc):
    """
    Single minimax training step:
      1. Compute gradients of loss w.r.t. model AND weights
      2. Model:   gradient DESCENT (minimize)
      3. Weights: gradient ASCENT  (maximize)
    
    Args:
        model, loss_weights:                   Trainable modules
        model_opt_state, weights_opt_state:    Optax states
        model_optimizer, weights_optimizer:    Optax optimizers
        hh:                                    HodgkinHuxley (fixed)
        V_colloc, t_colloc, I_colloc:          Collocation points
    
    Returns:
        model, loss_weights, model_opt_state, weights_opt_state, info
    """
    # --- Compute loss and gradients ---
    # Gradient w.r.t. model (arg 0) -> for descent
    @eqx.filter_value_and_grad(has_aux=True)
    def model_loss(model):
        return adversarial_physics_loss(
            model, loss_weights, hh, V_colloc, t_colloc, I_colloc
        )
    
    (loss, info), model_grads = model_loss(model)
    
    # Gradient w.r.t. loss_weights (arg 0 of inner fn) -> for ascent
    @eqx.filter_value_and_grad(has_aux=True)
    def weight_loss(loss_weights):
        return adversarial_physics_loss(
            model, loss_weights, hh, V_colloc, t_colloc, I_colloc
        )
    
    (_, _), weight_grads = weight_loss(loss_weights)
    
    # --- Model update: gradient DESCENT ---
    model_updates, model_opt_state = model_optimizer.update(
        model_grads, model_opt_state, model
    )
    model = eqx.apply_updates(model, model_updates)
    
    # --- Weights update: gradient ASCENT ---
    # Negate gradients so Optax (which does descent) effectively does ascent
    neg_weight_grads = jax.tree.map(lambda g: -g, weight_grads)
    weight_updates, weights_opt_state = weights_optimizer.update(
        neg_weight_grads, weights_opt_state, loss_weights
    )
    loss_weights = eqx.apply_updates(loss_weights, weight_updates)
    
    return model, loss_weights, model_opt_state, weights_opt_state, info


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from HH_NeuralODE import create_model
    
    print("Adversarial Physics Loss - Test")
    print("=" * 50)
    
    key = jax.random.PRNGKey(0)
    
    # Create components
    model = create_model(key=key)
    hh = HodgkinHuxley()
    
    # Create trainable loss weights (e.g., 8 weights for 8 voltage regions)
    n_weights = 8
    loss_weights = LossWeights(n_terms=n_weights, init_value=0.0)
    print(f"Initial weights: {loss_weights.weights}")
    
    # Sample collocation points
    N = 64
    key1, key2, key3 = jax.random.split(key, 3)
    V_colloc = jax.random.uniform(key1, (N,), minval=-80.0, maxval=40.0)
    t_colloc = jax.random.uniform(key2, (N,), minval=0.0, maxval=100.0)
    I_colloc = jnp.ones(N) * 10.0  # Constant current
    
    # Compute loss
    loss, info = adversarial_physics_loss(
        model, loss_weights, hh, V_colloc, t_colloc, I_colloc
    )
    print(f"\nInitial loss: {loss:.4f}")
    print(f"Physics residual: {info['physics_loss']:.4f}")
    print(f"Weight range: [{info['min_weight']:.3f}, {info['max_weight']:.3f}]")
    
    # Test minimax step
    model_opt = optax.adam(1e-3)
    weight_opt = optax.adam(1e-2)  # Weights can learn faster
    
    model_opt_state = model_opt.init(eqx.filter(model, eqx.is_array))
    weight_opt_state = weight_opt.init(eqx.filter(loss_weights, eqx.is_array))
    
    model, loss_weights, model_opt_state, weight_opt_state, info = minimax_step(
        model, loss_weights,
        model_opt_state, weight_opt_state,
        model_opt, weight_opt,
        hh, V_colloc, t_colloc, I_colloc
    )
    
    print(f"\nAfter 1 minimax step:")
    print(f"Loss: {info['physics_loss']:.4f}")
    print(f"Weights: {loss_weights.weights}")
    
    print("\nAdversarial Physics Loss OK!")
