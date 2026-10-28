# Reproducing the coupled AMOC and empirical experiments

Follow the [repository installation instructions](../README.md#installation), then run each notebook in cell order from a fresh kernel. Figures are saved to `outputs/figures/` in the repository root.

## Coupled AMOC

| Notebook | Analysis |
| --- | --- |
| [00_inspect_data.ipynb](amoc/00_inspect_data.ipynb) | Depth profiles |
| [01a_full_analysis.ipynb](amoc/01a_full_analysis.ipynb) | Early warning signals |
| [01b_ablation_analysis.ipynb](amoc/01b_ablation_analysis.ipynb) | Forcing ablation |
| [02_spectral_analysis.ipynb](amoc/02_spectral_analysis.ipynb) | Spectra |

Run the full and spectral notebooks with both `STATE_VAR = 'amoc48N_ssp245'` and `STATE_VAR = 'amoc26N_ssp245'`. The compact NetCDF input is included.

The depth-profile notebook additionally uses approximately 717 MB of annual NASA GISS files from the simulations described by [Romanou et al. (2023)](https://journals.ametsoc.org/view/journals/clim/36/18/JCLI-D-22-0536.1.xml). Download them from the repository root:

```bash
python scripts/download_giss_nccs.py \
  --out-dir experiments/amoc/data \
  --scenarios a c g i --frequency annual --variable-type oceanic \
  --grid-types ojl --years 2015 2399
```

Use `--dry-run` to inspect the URLs. The depth profiles cover 2015–2399; the bundled scalar series extends through 2400.

## Empirical systems

Run [01_empirical_ews.ipynb](empirical/01_empirical_ews.ipynb) with `SYSTEM_NAME` set to `voice`, `climate`, `blackout`, `microcosm`, or `mitochondria`. It downloads and caches the inputs from [Zenodo record 7379358](https://zenodo.org/records/7379358). Microcosm uses six separate analysis segments. The archive's README and license are retained with the data; consult their use and publication terms.
