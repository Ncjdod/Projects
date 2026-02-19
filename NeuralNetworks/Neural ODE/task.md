# HH Neural ODE - Task List

## Step 1: Load Allen Brain Data
- [x] Install `pynwb` (Used `h5py` for robustness)
- [x] Download intracellular voltage traces (NWB)
- [x] Extract and visualize stimulus + response (Max V: ~42mV, clear spikes)
- [x] Format data for training (Ready in `AllenBrainLoader.py`)

## Step 2: Fix Neural ODE Training (step-by-step)
- [ ] **NEXT:** Integrate real data into training pipeline
- [ ] Train HH-PINN on real trajectory (forcing term)
- [ ] Evaluate fit (MSE + Spike Timing)
- [ ] Refine model (if needed)

## Previous Work (Complete)
- [x] HodgkinHuxley.py module
- [x] Pure JAX stack (Equinox + Diffrax + Optax)
- [x] v1-v3 training attempts (identified issues)
