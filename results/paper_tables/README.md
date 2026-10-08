# Numbers printed in the paper

One CSV per table of *Weisfeiler Leman is Incomplete On Graphs With A Simple Spectrum*, copied from the
manuscript. These are the reported values, not outputs of this repository.

Columns: `method, dataset, metric, mean, std, n_seeds`.

* `mean` and `std` are written as printed (same number of digits). `std` and `n_seeds` are empty where the
  table has no spread or the table does not state a number of seeds.
* `dataset` is `<dataset>/<backbone>` for `tab_molecular_results.csv`, the BREC category for
  `tab_brec_comparison.csv`, `ZINC-12K val, sigma=<value>` for `tab_noise_sensitivity.csv` and
  `Erdos-Renyi n=<size>` for `tab_runtime_scaling.csv`.
* `OOM` in `mean` marks the cell that the table reports as out of memory.
* Counts: `tab_brec_comparison.csv` (`pairs_distinguished_of_<N>`), `tab_invariance.csv` and
  `tab_noise_sensitivity.csv` (`failures_of_1000` per seed). In `tab_runtime_scaling.csv`,
  `slowdown_vs_PRiSM` is the ratio of the MAP or OAP median to the PRiSM median.
* `tab_rbound_stats.csv` and `tab_multiplicity_stats.csv` are the tables in long form, one row per
  dataset and statistic (`pct_*` columns in percent).

| file | paper table |
|---|---|
| `tab_brec_comparison.csv` | `tab:brec-comparison` |
| `tab_molecular_results.csv` | `tab:molecular-results` |
| `tab_transformer_results.csv` | `tab:transformer-results` |
| `tab_rbound_stats.csv` | `tab:rbound-stats` |
| `tab_multiplicity_stats.csv` | `tab:multiplicity-stats` |
| `tab_invariance.csv` | `tab:invariance` |
| `tab_noise_sensitivity.csv` | `tab:noise-sensitivity` |
| `tab_runtime_scaling.csv` | `tab:runtime-scaling` |
