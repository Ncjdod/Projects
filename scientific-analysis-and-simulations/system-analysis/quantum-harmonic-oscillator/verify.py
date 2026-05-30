import numpy as np
from qho_operators import QuantumHarmonicOscillator

def benchmark_basis_size(N):
    qho = QuantumHarmonicOscillator(N=N, alpha=1.0, m=1.0, omega=1.0, hbar=1.0)
    e0, _ = qho.get_perturbed_state_block(0)
    e1, _ = qho.get_perturbed_state_block(1)
    e2, _ = qho.get_perturbed_state_block(2)
    
    print(f"Basis size N = {N}:")
    print(f"  E0 = {e0:.12f} (Expected: 0.8037706513)")
    print(f"  E1 = {e1:.12f} (Expected: 2.7378922689)")
    print(f"  E2 = {e2:.12f} (Expected: 5.1792916890)\n")

def main():
    print("=" * 60)
    print("QUARTIC ANHARMONIC OSCILLATOR CONVERGENCE BENCHMARK (lambda = 1.0)")
    print("=" * 60)
    benchmark_basis_size(15)
    benchmark_basis_size(30)
    benchmark_basis_size(60)
    benchmark_basis_size(100)
    print("=" * 60)

if __name__ == "__main__":
    main()
