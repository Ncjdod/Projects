"""
Curriculum Learning for Neural ODE Training

Progressively increases training difficulty by expanding the 
time window and adjusting collocation point sampling.

Strategies:
  - Time window:  [0, T_short] → [0, T_full]
  - Current ramp:  I_ext = 0 → I_ext_max  (optional)
  - Physics weight: high → low  (optional, as data loss kicks in)
"""

import jax.numpy as jnp


class CurriculumScheduler:
    """
    Manages training curriculum — controls what the model sees at each stage.
    
    Usage:
        scheduler = CurriculumScheduler(
            T_start=2.0,     # Start with 2ms window
            T_end=100.0,     # End with 100ms window
            n_stages=10,     # 10 stages
            epochs_per_stage=200,
        )
        
        for epoch in range(2000):
            stage = scheduler.get_stage(epoch)
            T = stage['T']
            t_span = jnp.linspace(0, T, n_points)
            # ... train on this window
    """
    
    def __init__(self, T_start=2.0, T_end=100.0, 
                 n_stages=10, epochs_per_stage=200,
                 schedule='linear',
                 I_ext_start=0.0, I_ext_end=None,
                 physics_weight_start=10.0, physics_weight_end=1.0):
        """
        Args:
            T_start:             Initial time window (ms)
            T_end:               Final time window (ms)
            n_stages:            Number of curriculum stages
            epochs_per_stage:    Epochs before advancing stage
            schedule:            'linear', 'exponential', or 'cosine'
            I_ext_start:         Starting external current (optional ramp)
            I_ext_end:           Final external current (None = no ramp)
            physics_weight_start: Physics loss weight at start
            physics_weight_end:   Physics loss weight at end
        """
        self.T_start = T_start
        self.T_end = T_end
        self.n_stages = n_stages
        self.epochs_per_stage = epochs_per_stage
        self.total_epochs = n_stages * epochs_per_stage
        self.schedule = schedule
        
        # Optional current ramp
        self.I_ext_start = I_ext_start
        self.I_ext_end = I_ext_end if I_ext_end is not None else I_ext_start
        
        # Optional physics weight decay
        self.physics_weight_start = physics_weight_start
        self.physics_weight_end = physics_weight_end
    
    def _progress(self, epoch):
        """Fraction of curriculum completed [0, 1]."""
        return min(epoch / max(self.total_epochs, 1), 1.0)
    
    def _interpolate(self, start, end, epoch):
        """Interpolate between start and end values based on schedule."""
        p = self._progress(epoch)
        
        if self.schedule == 'linear':
            return start + (end - start) * p
        
        elif self.schedule == 'exponential':
            # Exponential: faster early progress
            return start * (end / max(start, 1e-8)) ** p
        
        elif self.schedule == 'cosine':
            # Smooth cosine: slow start, fast middle, slow end
            cos_p = 0.5 * (1.0 - float(jnp.cos(jnp.pi * p)))
            return start + (end - start) * cos_p
        
        else:
            return start + (end - start) * p
    
    def get_stage_number(self, epoch):
        """Current stage index (0 to n_stages-1)."""
        return min(epoch // self.epochs_per_stage, self.n_stages - 1)
    
    def get_stage(self, epoch):
        """
        Get all curriculum parameters for the current epoch.
        
        Returns:
            dict with keys:
                'T':              Current time window (ms)
                'I_ext':          Current external current
                'physics_weight': Current physics loss weight
                'stage':          Stage number
                'progress':       Fraction complete [0, 1]
        """
        stage_num = self.get_stage_number(epoch)
        progress = self._progress(epoch)
        
        T = self._interpolate(self.T_start, self.T_end, epoch)
        I_ext = self._interpolate(self.I_ext_start, self.I_ext_end, epoch)
        phys_w = self._interpolate(
            self.physics_weight_start, self.physics_weight_end, epoch
        )
        
        return {
            'T': T,
            'I_ext': I_ext,
            'physics_weight': phys_w,
            'stage': stage_num,
            'progress': progress,
        }
    
    def get_t_span(self, epoch, n_points=200):
        """
        Get time array for the current stage.
        
        Args:
            epoch:    Current epoch
            n_points: Number of time points
        
        Returns:
            t_span: jnp array of shape (n_points,) in ms
        """
        T = self.get_stage(epoch)['T']
        return jnp.linspace(0.0, T, n_points)
    
    def get_collocation_points(self, epoch, n_points, key):
        """
        Sample collocation points for physics loss, 
        adapted to the current curriculum stage.
        
        Args:
            epoch:    Current epoch
            n_points: Number of collocation points
            key:      JAX PRNG key
        
        Returns:
            V_colloc:     Voltage sample points
            t_colloc:     Time sample points
            I_ext_colloc: Current at each point
        """
        import jax.random
        
        stage = self.get_stage(epoch)
        T = stage['T']
        I_ext = stage['I_ext']
        
        keys = jax.random.split(key, 2)
        
        # Time: uniform in [0, T]
        t_colloc = jax.random.uniform(keys[0], (n_points,), minval=0.0, maxval=T)
        
        # Voltage: focus on physiological range [-80, 40] mV
        V_colloc = jax.random.uniform(keys[1], (n_points,), minval=-80.0, maxval=40.0)
        
        # Current: constant at curriculum level
        I_ext_colloc = jnp.ones(n_points) * I_ext
        
        return V_colloc, t_colloc, I_ext_colloc
    
    def summary(self):
        """Print curriculum schedule."""
        print(f"Curriculum Learning Schedule ({self.schedule})")
        print(f"{'Stage':>6} {'Epochs':>12} {'T (ms)':>10} {'I_ext':>8} {'Phys_w':>8}")
        print("-" * 50)
        for s in range(self.n_stages):
            epoch = s * self.epochs_per_stage
            stage = self.get_stage(epoch)
            ep_range = f"{epoch}-{epoch + self.epochs_per_stage - 1}"
            print(f"{s:>6} {ep_range:>12} {stage['T']:>10.1f} "
                  f"{stage['I_ext']:>8.1f} {stage['physics_weight']:>8.2f}")
        # Final
        epoch = self.total_epochs
        stage = self.get_stage(epoch)
        print(f"{'Final':>6} {'':>12} {stage['T']:>10.1f} "
              f"{stage['I_ext']:>8.1f} {stage['physics_weight']:>8.2f}")


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    print("Curriculum Learning - Test")
    print("=" * 50)
    
    # Example: 10 stages, each 200 epochs
    scheduler = CurriculumScheduler(
        T_start=2.0,          # Start with 2ms
        T_end=50.0,           # End with 50ms
        n_stages=10,
        epochs_per_stage=200,
        schedule='linear',
        I_ext_start=0.0,      # Start with no stimulus
        I_ext_end=10.0,       # Ramp to 10 uA/cm^2
        physics_weight_start=10.0,
        physics_weight_end=1.0,
    )
    
    scheduler.summary()
    
    # Check specific epochs
    print("\nDetailed check:")
    for epoch in [0, 500, 1000, 1500, 1999]:
        stage = scheduler.get_stage(epoch)
        print(f"  Epoch {epoch:>5}: Stage {stage['stage']}, "
              f"T={stage['T']:.1f}ms, I={stage['I_ext']:.1f}, "
              f"phys_w={stage['physics_weight']:.2f}")
    
    print("\nCurriculum OK!")
