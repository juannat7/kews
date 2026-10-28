# Reproducing the model experiments

Follow the [repository installation instructions](../README.md#installation), then run each notebook in cell order from a fresh kernel. Figures are saved to `outputs/figures/` in the repository root.

| Notebook | Analysis |
| --- | --- |
| [01a_bifur_harvesting.ipynb](01a_bifur_harvesting.ipynb) | May harvesting |
| [01b_bifur_rosenzweig_macarthur_hopf.ipynb](01b_bifur_rosenzweig_macarthur_hopf.ipynb) | Rosenzweig–MacArthur Hopf and ACF comparison |
| [02a_rate_saddle.ipynb](02a_rate_saddle.ipynb) | Rate-induced saddle-node |
| [02b_rate_bautin.ipynb](02b_rate_bautin.ipynb) | Rate-induced Bautin |
| [03_amoc_model_ews.ipynb](03_amoc_model_ews.ipynb) | AMOC box model: bifurcation- and rate-induced tipping |

The six simulation models are bundled in `experiments/systems/`. The notebooks generate 50 tipping and 50 non-tipping trajectories per system with the paper's seeds and settings. No external simulation package is required.
