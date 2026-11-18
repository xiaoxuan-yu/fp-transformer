# PoseBusters docking with torsional Vina scores (main text Fig. 6, SI Secs. S4.2--S4.3)

190 PoseBusters systems (cleaned 308-case journal subset) with receptor,
ligand translation and ligand rotation fixed; 512 replicas, 1000 updates at
300 K, seeds 6300--6304.  MC: random-walk Metropolis over ligand torsions;
FPT: pointwise leave-one-out torus KDE score, fixed bandwidth 0.4, Adam 0.03.

| Script | Role |
|---|---|
| `download_dataset.py`, `prepare_dataset.py`, `validate_dataset.py` | PoseBusters archive -> `data/raw` -> PDBQT in `data/prepared` (+ `manifest.json/csv`) |
| `run_benchmark.py`, `search.py`, `pdbqt.py`, `dataset.py` | MC/FPT search |
| `analyze_results.py`, `statistical_analysis.py` | run summaries and SI tables |
| `plot_paper_figures.py`, `render_panel_a_pymol.py` | Fig. 6 and SI figures |
| `summarize_subset.py`, `mc_temperature_scan_systems.txt` | SI MC temperature scan |

`data/raw` (the 308 journal-subset complexes from Zenodo record 8278563,
CC-BY 4.0) and `data/prepared` are provided by the Zenodo dataset of this
release (see the top-level README) or regenerated with the preparation
commands below.

```bash
# preparation (optional when the Zenodo inputs are used)
pip install -r examples/vina/requirements-prepare.txt   # separate environment
python -m examples.vina.download_dataset
python -m examples.vina.prepare_dataset
python -m examples.vina.validate_dataset
# benchmark
python -m examples.vina.run_benchmark --num-seeds 5 --output-dir data/vina/full_row_mc300
python -m examples.vina.run_benchmark --num-seeds 5 --mc-temperature 75 --output-dir data/vina/full_row_mc75
python -m examples.vina.analyze_results --output-dir data/vina/full_row_mc300
python -m examples.vina.plot_paper_figures --output-dir data/vina/full_row_mc300
# MC temperature scan
for T in 300 150 75 30; do
  python -m examples.vina.run_benchmark --num-seeds 5 --mc-temperature $T \
      --systems $(cat examples/vina/mc_temperature_scan_systems.txt) --output-dir data/vina/scan_row_mc$T
done
python -m examples.vina.summarize_subset data/vina/scan_row_mc{300,150,75,30}
```
