# FP-Transformer: Fokker--Planck transport of particle ensembles

JAX implementation of the Fokker--Planck transport (FPT) sampler.  Particles
are moved along `x <- x - opt(beta grad U - s)`, where `s` is the pointwise
score of a leave-one-out kernel density estimate of the current ensemble.

## Installation

```bash
uv sync --extra cuda --extra test      # Python 3.11, JAX with CUDA 12
```

## Library

| Module | Content |
|---|---|
| `kernel.Gaussian` | Leave-one-out kernel density estimate and its pointwise score (kernel centers held fixed); shared or particlewise bandwidths controlled by the kernel-row mass; `score_backend="fused"` evaluates the torus score in one Pallas row pass |
| `kernel.fused` | Fused leave-one-out torus score and log row mass |
| `kernel.row_mass` | Exact particlewise row-mass bandwidths (bisection and safeguarded Newton) and the resulting torus score |
| `core.Sampler` | FPT update with optional bandwidth refresh, checkpointing and an `observe_fn` hook |
| `optim` | Replica-global RMS optimizer |
| `potential` | Test potentials (Tiwary--Berne, periodic, 100D periodic) and the torsion-space AutoDock Vina score |
| `ecepp` | ECEPP/3 force field on internal coordinates |

## Tests

```bash
pytest tests
```
