# Tiwary--Berne nonstationary transport (main text Fig. 2, SI Sec. S3.1)

Exact row-mass FPT on the two-dimensional Tiwary--Berne potential at
beta = 1.25, from N((-0.5, 0.5), 0.3^2 I), for N = 2048, 4096, 8192 with
K_N = round(0.16 N^0.6) and four seeds, compared with a reversible
finite-volume Fokker--Planck solution and with the numerical continuum oracle.

| Script | Role |
|---|---|
| `run_transport.py` | one particle run (explicit Euler, fresh particlewise row-mass bandwidths) |
| `run_batch.py` | the 12 production runs |
| `analyze_transport.py` | finite-volume reference, W2 to reference and target, smooth tests |
| `oracle_flow.py`, `merge_oracle_flow.py` | continuum-oracle decomposition E_tot, E_dyn, E_init |
| `export_figure_data.py` | compact figure data and numerical audits |
| `plot_transport.py` | Fig. 2 and the SI diagnostics figure |
| `common.py`, `numerics.py` | potential, target metrics, finite-volume and OT primitives |

```bash
python -m examples.tb.run_batch --output-dir data/tb_transport
python -m examples.tb.analyze_transport                                   # -> data/tb_transport/analysis
python -m examples.tb.analyze_transport --reference-resolution 128 --output-dir data/tb_transport/analysis_grid128
python -m examples.tb.analyze_transport --reference-resolution 256 --output-dir data/tb_transport/analysis_grid256
python -m examples.tb.oracle_flow --output-dir data/tb_transport/oracle_flow_analysis
# oracle audits on the representative run
python -m examples.tb.oracle_flow --only-n 2048 --seed 20260820 --reference-resolution 128 \
    --output-dir data/tb_transport/oracle_grid128      # copy its oracle_flow_runs.csv to
                                                       # data/tb_transport/oracle_flow_grid128_audit.csv
python -m examples.tb.oracle_flow --only-n 2048 --seed 20260820 --oracle-time-step 5e-4 \
    --output-dir data/tb_transport/oracle_step_audit
# time-step audit of the representative run
python -m examples.tb.run_transport --replicas 2048 --seed 20260820 --row-mass 16 --time-step 1e-4 \
    --output-dir data/tb_transport_dt_audit/n2048/seed20260820
python -m examples.tb.export_figure_data                                  # -> data/tb_transport/figure_data
python -m examples.tb.plot_transport
```

`oracle_flow.py` accepts `--only-n`/`--seed` to split the oracle runs; the
per-split outputs are combined by `merge_oracle_flow.py OUTPUT INPUTS...`.
