# Project Status

The Hodgkin-Huxley (HH) Neural ODE trajectory fitting and vector field distillation models are temporarily paused.

## Current Challenges

The principal challenge arises from the high temporal frequency differences in the Hodgkin-Huxley dynamics (e.g., sharp action potential spikes vs. quiet refractory periods). These multi-scale frequency components make it highly challenging for standard continuous-depth neural ordinary differential equations to converge and generalize across long trajectories without gradient issues or compilation bottlenecks.

## Next Steps

To address these temporal scaling limitations, a new modeling approach based on neural operators and Koopman theory will soon be added to the repository to model the infinite-dimensional linear representations of the system dynamics.

