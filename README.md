# FP-Transformer: Fokker--Planck transport of particle ensembles

JAX implementation of the Fokker--Planck transport (FPT) sampler.  Particles
are moved along `x <- x - opt(beta grad U - s)`, where `s` is the pointwise
score of a leave-one-out kernel density estimate of the current ensemble.

## Installation

```bash
uv sync --extra cuda --extra test      # Python 3.11, JAX with CUDA 12
```

Optional extras: `md` (Amber force field through jax-md and OpenMM topology
input) and `mpi` (mpi4py, for launch scripts that request
`cluster_detection_method="mpi4py"` in `distributed.init_jax_distributed`).

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
| `potential.AmberPotential`, `core.AmberSystem`, `potential.GAFF` | Cartesian molecular systems (extra `md`) |

## Tests

```bash
pytest tests
```
