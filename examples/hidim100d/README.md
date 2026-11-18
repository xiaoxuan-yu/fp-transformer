# One-hundred-dimensional periodic landscape (main text Fig. 4; SI Sec. S3.2)

U_D on the D-torus with sigma = pi sqrt(2)/6; for D = 100 the stable set is
the all-pi corner and the 100 single-zero corners.  At beta = 5, 5120
particles start from a wrapped Gaussian of scale 0.4 about one single-zero
corner (seeds 0--3) and run for 20000 updates.  Basins are identified by
split-half identification of quenched configurations
(`examples/common/identification.py`), calibrated on equilibrium Langevin
ensembles.

| Script | Paper item |
|---|---|
| `run_discovery.py` | Fig. 4, Table S5: shared-bandwidth FPT (fused row pass, row mass in [20, 40], RMSprop 0.14), Langevin, no-score; equilibrium reference of identification |
| `make_equilibrium.py` | pre-equilibrated Langevin ensembles (equilibrium reference, Fig. S4) |
| `run_exact_row_mass.py` | exact row-mass FPT from the equilibrium ensembles (Fig. S4b) |
| `plot_discovery.py`, `plot_si.py` | Fig. 4, Fig. S4 |

```bash
D=data/numerical_hidim_periodic_discovery
python -m examples.hidim100d.run_discovery --methods fpt --output-dir $D/tuned/beta5_fpt_fused_row_k20_40_p1_visits
python -m examples.hidim100d.run_discovery --methods ld noscore --output-dir $D/tuned/beta5_ld_noscore_visits
# beta = 5 equilibrium start
python -m examples.hidim100d.make_equilibrium --beta 5 --out-dir $D/equilibrium_beta5
I=$D/equilibrium_beta5/ld_beta5_seed0.npy
# equilibrium reference of split-half identification (four noise seeds)
python -m examples.hidim100d.run_discovery --methods ld --ld-steps 0.003 --init-file $I \
    --output-dir $D/equilibrium_beta5/ld_split_half
python -m examples.hidim100d.run_discovery --methods ld --beta 5 --seeds 0 --num-steps 2000 --ld-steps 0.003 0.01 \
    --init-file $I --output-dir $D/equilibrium_beta5/ld_continue
python -m examples.hidim100d.run_discovery --methods fpt --beta 5 --seeds 0 --num-steps 2000 \
    --init-file $I --output-dir $D/equilibrium_beta5/global_row_rmsprop014_k20_40
python -m examples.hidim100d.run_exact_row_mass --beta 5 --seeds 0 --num-steps 2000 --dts 0.03 0.1 \
    --row-masses 27 100 400 --init-file $I --output-dir $D/equilibrium_beta5/perparticle_euler
# dimension scan: (D, beta) = (5, 3.53), (10, 4.3), (20, 5.35); replica scan for D = 5, 10
python -m examples.hidim100d.make_equilibrium --dim 10 --beta 4.3 --out-dir $D/equilibrium_dimscan/D10_beta4.3
#   then ld_continue / perparticle_euler (K = 27 100 400) / perparticle_euler_smallK (K = 4 8, dt 0.03)
#   with --dim 10 --beta 4.3 as above, and for --replicas 10240, 20480 the
#   nscan_N<N> runs (300 updates, dt 0.03, K = 8 and round(0.16 N^0.6)).
python -m examples.hidim100d.plot_discovery
python -m examples.hidim100d.plot_si
```
