"""
HH Neural ODE - Complete Training Script

Combines all components:
  - Allen Brain data loading & preprocessing
  - HH Neural ODE model (Fourier + 4x64 MLP)
  - Adversarial physics loss (trainable weights, gradient ascent)
  - Curriculum learning (progressive time windows)
  - Minimax training loop

Usage:
    python train.py
"""

import os
import sys
import time

# Add HH_model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'HH_model'))

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import diffrax
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Our modules
from HH_NeuralODE import HHNeuralODE, FourierFeatures, create_model, integrate
from HodgkinHuxley import HodgkinHuxley
from physics_loss import LossWeights, physics_residual, adversarial_physics_loss
from curriculum import CurriculumScheduler
from AllenBrainLoader import download_nwb, find_sweeps, get_sweep_data, SPECIMEN_ID


# ============================================================
# Configuration
# ============================================================
class Config:
    """All hyperparameters in one place."""
    
    # --- Model ---
    n_fourier = 32          # Fourier basis functions
    fourier_sigma = 1.0     # Fourier frequency scale
    seed = 42
    
    # --- Data ---
    downsample_factor = 20  # 200kHz -> 10kHz
    window_pre = 5.0        # ms before first spike
    window_post = 50.0      # ms after first spike
    
    # --- Curriculum ---
    T_start = 5.0           # Start with 5ms window
    T_end = 55.0            # Full window
    n_stages = 10
    epochs_per_stage = 300
    schedule = 'linear'
    physics_weight_start = 10.0
    physics_weight_end = 1.0
    
    # --- Training ---
    model_lr = 1e-3
    weights_lr = 1e-2       # Weights learn faster (ascent)
    n_colloc = 64           # Collocation points per step
    n_loss_weights = 8      # Adversarial weight bins
    log_every = 50          # Print every N epochs
    plot_every = 500        # Plot every N epochs
    
    # --- Integration ---
    dt0 = 0.01
    rtol = 1e-3
    atol = 1e-5
    
    @property
    def total_epochs(self):
        return self.n_stages * self.epochs_per_stage


# ============================================================
# Data Loading
# ============================================================
def load_allen_data(config):
    """Load and preprocess Allen Brain electrophysiology data."""
    print("\n--- Loading Allen Brain Data ---")
    
    filepath = download_nwb()
    assert filepath is not None, "Failed to download NWB file"
    
    with h5py.File(filepath, 'r') as f:
        sweeps = find_sweeps(f)
        print(f"Found {len(sweeps)} sweeps")
        
        # Find best spiking sweep
        best_sweep = None
        best_n_spikes = 0
        
        for sweep_name in sweeps:
            t, v, c = get_sweep_data(f, sweep_name)
            if v is None:
                continue
            crossings = np.diff(np.sign(v - 0.0))
            n_spikes = np.sum(crossings > 0)
            if n_spikes > best_n_spikes:
                best_n_spikes = n_spikes
                best_sweep = sweep_name
        
        print(f"Best sweep: {best_sweep} ({best_n_spikes} spikes)")
        t_raw, v_raw, c_raw = get_sweep_data(f, best_sweep)
    
    # Make time relative
    t_raw = t_raw - t_raw[0]
    
    # Downsample
    ds = config.downsample_factor
    t_ds = t_raw[::ds]
    v_ds = v_raw[::ds]
    c_ds = c_raw[::ds]
    
    # Find first spike and extract window
    crossings = np.diff(np.sign(v_ds - 0.0))
    spike_idx = np.where(crossings > 0)[0]
    
    if len(spike_idx) > 0:
        first_spike_t = t_ds[spike_idx[0]]
        t_start = max(first_spike_t - config.window_pre, t_ds[0])
        t_end = min(first_spike_t + config.window_post, t_ds[-1])
    else:
        print("WARNING: No spikes found. Using first 55ms.")
        t_start = t_ds[0]
        t_end = t_ds[0] + 55.0
    
    mask = (t_ds >= t_start) & (t_ds <= t_end)
    t_train = t_ds[mask] - t_ds[mask][0]  # Shift to start at 0
    v_train = v_ds[mask]
    c_train = c_ds[mask]
    
    # Count spikes in window
    cross_win = np.diff(np.sign(v_train - 0.0))
    n_spikes_win = np.sum(cross_win > 0)
    
    print(f"Training window: {t_train[-1]:.1f}ms, {len(t_train)} points, {n_spikes_win} spikes")
    print(f"V: [{v_train.min():.1f}, {v_train.max():.1f}] mV")
    print(f"I: [{c_train.min():.1f}, {c_train.max():.1f}] pA")
    
    # Convert to JAX
    t_jax = jnp.array(t_train, dtype=jnp.float32)
    v_jax = jnp.array(v_train, dtype=jnp.float32)
    c_jax = jnp.array(c_train, dtype=jnp.float32)
    
    return t_jax, v_jax, c_jax


def make_I_ext_fn(t_data, c_data):
    """Create interpolation function for external current."""
    def I_ext_fn(t):
        return jnp.interp(t, t_data, c_data)
    return I_ext_fn


def get_curriculum_data(t_full, v_full, c_full, T_window):
    """
    Extract a sub-window [0, T_window] from the full training data.
    Used by curriculum to progressively expand the window.
    """
    mask = t_full <= T_window
    t_sub = t_full[mask]
    v_sub = v_full[mask]
    c_sub = c_full[mask]
    return t_sub, v_sub, c_sub


# ============================================================
# Loss Functions
# ============================================================
def data_loss_fn(model, y0, t_span, I_ext_fn, v_target):
    """
    Data loss: MSE between predicted and observed voltage.
    Integrates the Neural ODE and compares against Allen data.
    """
    y_pred = integrate(model, y0, t_span, I_ext_fn,
                       dt0=0.01, rtol=1e-3, atol=1e-5)
    return jnp.mean((y_pred[:, 0] - v_target) ** 2)


def combined_loss_fn(model, loss_weights, hh,
                     y0, t_span, I_ext_fn, v_target,
                     V_colloc, t_colloc, I_colloc,
                     physics_weight):
    """
    Combined loss: Data MSE + Adversarial Physics.
    
    L_total = L_data + lambda * L_physics_adversarial
    
    Model params:  gradient DESCENT on L_total
    Loss weights:  gradient ASCENT  on L_total
    """
    # --- Data loss ---
    y_pred = integrate(model, y0, t_span, I_ext_fn,
                       dt0=0.01, rtol=1e-3, atol=1e-5)
    data_loss = jnp.mean((y_pred[:, 0] - v_target) ** 2)
    
    # --- Adversarial physics loss ---
    phys_loss, phys_info = adversarial_physics_loss(
        model, loss_weights, hh, V_colloc, t_colloc, I_colloc
    )
    
    total = data_loss + physics_weight * phys_loss
    
    info = {
        'total_loss': total,
        'data_loss': data_loss,
        'physics_loss': phys_info['physics_loss'],
        'weighted_phys': phys_info['weighted_loss'],
        'mean_weight': phys_info['mean_weight'],
        'max_weight': phys_info['max_weight'],
    }
    
    return total, info


# ============================================================
# Training Step (Minimax)
# ============================================================
@eqx.filter_jit
def train_step(model, loss_weights,
               model_opt_state, weights_opt_state,
               model_optimizer, weights_optimizer,
               hh, y0, t_span, I_ext_fn, v_target,
               V_colloc, t_colloc, I_colloc,
               physics_weight):
    """
    Single minimax training step.
    
    1. Compute combined loss
    2. Model params:  gradient DESCENT (minimize)
    3. Loss weights:  gradient ASCENT  (maximize)
    """
    # --- Model gradients (descent) ---
    @eqx.filter_value_and_grad(has_aux=True)
    def model_loss(model):
        return combined_loss_fn(
            model, loss_weights, hh,
            y0, t_span, I_ext_fn, v_target,
            V_colloc, t_colloc, I_colloc,
            physics_weight
        )
    
    (loss, info), model_grads = model_loss(model)
    
    # --- Weight gradients (ascent) ---
    @eqx.filter_value_and_grad(has_aux=True)
    def weight_loss(loss_weights):
        return combined_loss_fn(
            model, loss_weights, hh,
            y0, t_span, I_ext_fn, v_target,
            V_colloc, t_colloc, I_colloc,
            physics_weight
        )
    
    (_, _), weight_grads = weight_loss(loss_weights)
    
    # --- Update model (descent) ---
    model_updates, model_opt_state = model_optimizer.update(
        model_grads, model_opt_state, model
    )
    model = eqx.apply_updates(model, model_updates)
    
    # --- Update weights (ascent = negate grads for optax) ---
    neg_weight_grads = jax.tree.map(lambda g: -g, weight_grads)
    weight_updates, weights_opt_state = weights_optimizer.update(
        neg_weight_grads, weights_opt_state, loss_weights
    )
    loss_weights = eqx.apply_updates(loss_weights, weight_updates)
    
    return model, loss_weights, model_opt_state, weights_opt_state, info


# ============================================================
# Visualization
# ============================================================
def plot_progress(model, t_data, v_data, c_data, I_ext_fn,
                  epoch, info, loss_history, save_dir="HH_model"):
    """Plot current model predictions vs data."""
    os.makedirs(save_dir, exist_ok=True)
    
    y0 = jnp.array([v_data[0]])
    y_pred = integrate(model, y0, t_data, I_ext_fn)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Voltage trace ---
    ax = axes[0, 0]
    ax.plot(t_data, v_data, 'b-', lw=1.5, label='Allen Data', alpha=0.7)
    ax.plot(t_data, y_pred[:, 0], 'r--', lw=1.5, label='Neural ODE')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title(f'Epoch {epoch}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Stimulus ---
    ax = axes[0, 1]
    ax.plot(t_data, c_data, 'g-', lw=1.5)
    ax.set_ylabel('Current (pA)')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Stimulus')
    ax.grid(True, alpha=0.3)
    
    # --- Loss history ---
    ax = axes[1, 0]
    if loss_history:
        epochs_h = [h['epoch'] for h in loss_history]
        data_h = [h['data_loss'] for h in loss_history]
        phys_h = [h['physics_loss'] for h in loss_history]
        ax.semilogy(epochs_h, data_h, 'b-', label='Data Loss', alpha=0.7)
        ax.semilogy(epochs_h, phys_h, 'r-', label='Physics Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Training History')
        ax.grid(True, alpha=0.3)
    
    # --- Weight evolution ---
    ax = axes[1, 1]
    if loss_history:
        mean_w = [h.get('mean_weight', 1.0) for h in loss_history]
        max_w = [h.get('max_weight', 1.0) for h in loss_history]
        ax.plot(epochs_h, mean_w, 'k-', label='Mean weight')
        ax.plot(epochs_h, max_w, 'r--', label='Max weight')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Adversarial Weight')
        ax.legend()
        ax.set_title('Loss Weight Evolution')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_epoch_{epoch:05d}.png'), dpi=100)
    plt.close()
    

def plot_final(model, t_data, v_data, c_data, I_ext_fn,
               loss_history, save_dir="HH_model"):
    """Final publication-quality plot."""
    os.makedirs(save_dir, exist_ok=True)
    
    y0 = jnp.array([v_data[0]])
    y_pred = integrate(model, y0, t_data, I_ext_fn)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), 
                              gridspec_kw={'height_ratios': [3, 1, 2]})
    
    # --- Voltage ---
    ax = axes[0]
    ax.plot(t_data, v_data, 'b-', lw=2, label='Allen Brain Data', alpha=0.7)
    ax.plot(t_data, y_pred[:, 0], 'r--', lw=2, label='Neural ODE Prediction')
    ax.set_ylabel('Voltage (mV)', fontsize=12)
    ax.set_title(f'HH Neural ODE - Specimen {SPECIMEN_ID}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # --- Stimulus ---
    ax = axes[1]
    ax.plot(t_data, c_data, 'g-', lw=2)
    ax.set_ylabel('I_ext (pA)', fontsize=12)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # --- Loss curves ---
    ax = axes[2]
    if loss_history:
        epochs_h = [h['epoch'] for h in loss_history]
        data_h = [h['data_loss'] for h in loss_history]
        phys_h = [h['physics_loss'] for h in loss_history]
        total_h = [h['total_loss'] for h in loss_history]
        ax.semilogy(epochs_h, total_h, 'k-', lw=2, label='Total', alpha=0.8)
        ax.semilogy(epochs_h, data_h, 'b-', lw=1.5, label='Data', alpha=0.6)
        ax.semilogy(epochs_h, phys_h, 'r-', lw=1.5, label='Physics', alpha=0.6)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_result.png'), dpi=150)
    plt.close()
    print(f"Saved final_result.png")


# ============================================================
# Main Training Loop
# ============================================================
def train(config=None):
    """Run the full training pipeline."""
    if config is None:
        config = Config()
    
    print("=" * 60)
    print("HH Neural ODE - Training")
    print("=" * 60)
    
    key = jax.random.PRNGKey(config.seed)
    
    # ---- 1. Load Data ----
    t_full, v_full, c_full = load_allen_data(config)
    I_ext_fn_full = make_I_ext_fn(t_full, c_full)
    
    # ---- 2. Create Model ----
    print("\n--- Creating Model ---")
    key, model_key = jax.random.split(key)
    model = create_model(
        key=model_key,
        n_fourier=config.n_fourier,
        sigma=config.fourier_sigma
    )
    
    # Count parameters
    params = eqx.filter(model, eqx.is_array)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Model parameters: {n_params}")
    
    # ---- 3. Create Adversarial Loss Weights ----
    loss_weights = LossWeights(
        n_terms=config.n_colloc,
        init_value=0.0
    )
    print(f"Adversarial weights: {config.n_colloc} terms")
    
    # ---- 4. Physics Model ----
    hh = HodgkinHuxley()
    
    # ---- 5. Optimizers ----
    model_optimizer = optax.adam(config.model_lr)
    weights_optimizer = optax.adam(config.weights_lr)
    
    model_opt_state = model_optimizer.init(eqx.filter(model, eqx.is_array))
    weight_opt_state = weights_optimizer.init(
        eqx.filter(loss_weights, eqx.is_array)
    )
    
    # ---- 6. Curriculum Scheduler ----
    scheduler = CurriculumScheduler(
        T_start=config.T_start,
        T_end=min(config.T_end, float(t_full[-1])),
        n_stages=config.n_stages,
        epochs_per_stage=config.epochs_per_stage,
        schedule=config.schedule,
        physics_weight_start=config.physics_weight_start,
        physics_weight_end=config.physics_weight_end,
    )
    
    print("\n--- Curriculum Schedule ---")
    scheduler.summary()
    
    # ---- 7. Training Loop ----
    print(f"\n--- Training ({config.total_epochs} epochs) ---")
    loss_history = []
    start_time = time.time()
    prev_stage = -1
    
    for epoch in range(config.total_epochs):
        # Get curriculum parameters
        stage = scheduler.get_stage(epoch)
        T_curr = stage['T']
        phys_w = stage['physics_weight']
        stage_num = stage['stage']
        
        # Notify on stage change
        if stage_num != prev_stage:
            print(f"\n>> Stage {stage_num}: T={T_curr:.1f}ms, "
                  f"phys_weight={phys_w:.2f}")
            prev_stage = stage_num
        
        # Get curriculum sub-window of data
        t_sub, v_sub, c_sub = get_curriculum_data(t_full, v_full, c_full, T_curr)
        
        # Skip if too few points
        if len(t_sub) < 10:
            continue
            
        I_ext_fn = make_I_ext_fn(t_sub, c_sub)
        y0 = jnp.array([v_sub[0]])
        
        # Sample collocation points for physics loss
        key, ckey = jax.random.split(key)
        V_colloc, t_colloc, I_colloc = scheduler.get_collocation_points(
            epoch, config.n_colloc, ckey
        )
        
        # Minimax step
        model, loss_weights, model_opt_state, weight_opt_state, info = \
            train_step(
                model, loss_weights,
                model_opt_state, weight_opt_state,
                model_optimizer, weights_optimizer,
                hh, y0, t_sub, I_ext_fn, v_sub,
                V_colloc, t_colloc, I_colloc,
                phys_w
            )
        
        # Log
        info_np = {k: float(v) for k, v in info.items()}
        info_np['epoch'] = epoch
        info_np['stage'] = stage_num
        info_np['T'] = T_curr
        loss_history.append(info_np)
        
        if epoch % config.log_every == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:>5} | "
                  f"Total: {info_np['total_loss']:>10.4f} | "
                  f"Data: {info_np['data_loss']:>10.4f} | "
                  f"Phys: {info_np['physics_loss']:>10.2f} | "
                  f"W_max: {info_np['max_weight']:>6.2f} | "
                  f"T={T_curr:.1f}ms | "
                  f"{elapsed:.0f}s")
        
        # Plot progress
        if epoch % config.plot_every == 0 and epoch > 0:
            plot_progress(model, t_full, v_full, c_full, I_ext_fn_full,
                         epoch, info_np, loss_history)
    
    # ---- 8. Final Results ----
    elapsed_total = time.time() - start_time
    print(f"\n--- Training Complete ({elapsed_total:.0f}s) ---")
    print(f"Final data loss:    {loss_history[-1]['data_loss']:.6f}")
    print(f"Final physics loss: {loss_history[-1]['physics_loss']:.4f}")
    
    # Final plot
    plot_final(model, t_full, v_full, c_full, I_ext_fn_full, loss_history)
    
    # Save model
    model_path = os.path.join("HH_model", "trained_model.eqx")
    eqx.tree_serialise_leaves(model_path, model)
    print(f"Model saved to {model_path}")
    
    return model, loss_history


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    model, history = train()
