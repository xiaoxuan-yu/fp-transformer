# Three-dimensional eight-well target (main text Fig. 3; SI Sec. S3.2)

U_3 = exp{3[3 - sum_j sin^4 theta_j]} - 1 at beta = 3 beta_0, 1024 particles
from the Boltzmann law restricted to the initial well (the octant of
(-pi/2)^3), eight seeds 20260737--20260807, 13500 updates at fixed beta.
Shared-bandwidth FPT (mean row mass in [2, 4]) is advanced by RMSprop 0.15
(transport protocol) or by explicit Euler at the Langevin step 0.008
(integrator-matched diagnosis); Langevin dynamics uses step 0.008, and the
no-score control applies the FPT update to beta grad U alone. Wells are
identified by split-half identification (`examples/common/identification.py`).

| Script | Paper item |
|---|---|
| `run_discovery.py` | Fig. 3(b--d), Table S4 and Fig. S2; with `--equilibrium-audit-lrs` the FPT equilibrium audit (Table S3) |
| `run_ld_step_audit.py` | Table S2 (Langevin step) |
| `screen_fpt.py` | Table S3 (displacement screen) |
| `oracle_fp.py` | exact Fokker--Planck reference (Fig. 3, Fig. S2) |
| `run_exact_row_mass.py` | exact row-mass estimator at K = 3 (Fig. 3, Fig. S2) |
| `run_endpoint_continuation.py` | target-preserving endpoint continuation (Fig. S3) |
| `plot_diagnosis.py`, `plot_oracle.py`, `plot_endpoint_continuation.py` | Fig. 3, Fig. S2, Fig. S3 |

```bash
D=data/numerical_lowdim
# RMSprop transport protocol, Langevin, no-score; FPT equilibrium audit
python -m examples.periodic3d.run_discovery --output-dir $D/periodic_n3_rms_main_bb_visits \
    --save-endpoints --equilibrium-audit-lrs 0.15
# shared-bandwidth FPT and no-score by explicit Euler at the Langevin step
python -m examples.periodic3d.run_discovery --output-dir $D/periodic_n3_shared_euler_bb_visits \
    --fpt-optimizer sgd --fpt-lr 0.008 --methods fpt noscore --save-endpoints
# parameter selection
python -m examples.periodic3d.run_ld_step_audit --output-dir $D/n3_ula_step_audit_large
python -m examples.periodic3d.screen_fpt --output $D/n3_rms_screen.json --lrs 0.015 0.03 0.06 0.09 0.12 0.15
for band in "2 4" "5 10" "20 40"; do set -- $band
  python -m examples.periodic3d.run_discovery --methods fpt --seeds 20260737 --fpt-kernel-min $1 \
      --fpt-kernel-max $2 --equilibrium-audit-lrs 0.015 0.03 0.06 --output-dir $D/n3_rms_audit/band$1_$2
done
# continuum reference and exact row-mass estimator
python -m examples.periodic3d.oracle_fp --output-dir $D/oracle_n3_bb
for s in 20260737 20260747 20260757 20260767; do
  python -m examples.periodic3d.run_exact_row_mass --seed $s --row-mass 3 \
      --output $D/oracle_n3_exact_k3_bb_visits/seed$s.json
done
# endpoint continuation (shared-bandwidth FPT endpoints, explicit Euler)
python -m examples.periodic3d.run_endpoint_continuation \
    --output-dir data/numerical_periodic_endpoint_kernel/n3_shared_euler_bb
python -m examples.periodic3d.plot_diagnosis
python -m examples.periodic3d.plot_oracle
python -m examples.periodic3d.plot_endpoint_continuation
```

In the output files the Langevin baseline is keyed `ula` (unadjusted
Langevin algorithm, i.e. Euler--Maruyama), and the suffix `_fixed` marks the
fixed target temperature. GPU reductions are not bitwise deterministic, so
long explicit-Euler runs reproduce the reported numbers statistically rather
than bit for bit.
