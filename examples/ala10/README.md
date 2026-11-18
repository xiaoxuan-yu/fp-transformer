# Neutral alanine decapeptide with ECEPP/3 (main text Fig. 5, SI Sec. S4.1)

NH2-(Ala)10-COOH on the 20 free dihedrals at 462 K.  FPT and Langevin
dynamics start 4096 replicas from a narrowly perturbed coil conformation and
run 20000 updates for eight paired seeds 20260812--20260819.  The helical
families H0--H3 are identified blindly: each half of the output is clustered
by HDBSCAN in the backbone cosine--sine features, and the canonical Monte
Carlo atlas only names the clusters.

| Script | Role |
|---|---|
| `data/` | not versioned; `system.npz` (exported from SMMP by `export_system.py`) and the canonical MC reference, from the Zenodo dataset (see the top-level README) |
| `run_mc_reference.py` | canonical Monte Carlo reference (100000 sweeps) |
| `run_fpt.py` | FPT with Fisher--Rao reweighting and post-realization block moves |
| `run_ld.py`, `run_ld_controls.py` | Langevin controls replaying the FPT block-move event times |
| `audit_ld_stability.py` | Langevin step stability audit |
| `analyze_identification.py` | blind H0--H3 family identification (Fig. 5b,d) |
| `analyze_outputs.py` | strict-helix mass and canonical-PCA projections (Fig. 5c,e,f) |
| `plot_figure.py` | Fig. 5 (panel a needs `pymol`) |
| `lib/` | ECEPP wrapper and metrics, von Mises estimator, Fisher--Rao operators, block moves, family atlas |

```bash
python -m examples.ala10.run_mc_reference --output examples/ala10/data
for s in $(seq 20260812 20260819); do
  python -m examples.ala10.run_fpt --seed $s --output data/ala10/fpt/band30_50_m128_seed$s
done
python -m examples.ala10.run_ld_controls --fpt-runs data/ala10/fpt/band30_50_m128_seed* \
    --output-root data/ala10/ld_controls
python -m examples.ala10.audit_ld_stability --production data/ala10/ld_controls/seed*/ld_m128 \
    --output data/ala10/ld_stability_audit.csv
python -m examples.ala10.analyze_identification --fpt-root data/ala10/fpt \
    --ld-root data/ala10/ld_controls --output-dir data/ala10/identification
python -m examples.ala10.analyze_outputs --fpt-root data/ala10/fpt --ld-root data/ala10/ld_controls \
    --output-dir data/ala10/recovery
python -m examples.ala10.plot_figure
```

`export_system.py` needs an f2py build of SMMP, which is not redistributed
because of its license terms; the exported `data/system.npz` is part of the
Zenodo dataset.
