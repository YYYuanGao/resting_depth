
# Laminar effective connectivity between visual and caudal frontal cortex at 7T

Analysis code for:

> Gao Y, Saberi A, Manoli A, Royer J, Jia K, Valk SL.
> *[title]*. [journal, year]. [DOI]

The code estimates depth-resolved effective connectivity with regression dynamic causal modelling (rDCM) from 7T resting-state fMRI, relates feedforward (FF) and feedback (FB) coupling to spatial working-memory performance, places the three cortical composites (EVC, PPC, cauPFC) on the sensorimotor–association (S–A) axis, and relates coupling to intracortical microstructure (qT1, microstructural profile covariance).

## Data

Participant-level MRI and behavioural data are not included in this repository. [Data availability statement.]
Public resources used:

| Resource | Used for | Source |
|---|---|---|
| HCP-MMP1.0 parcellation | parcels, composites | Glasser et al. 2016; figshare 3498446, 4249400 |
| S–A axis ranks (Glasser 360) | Results 2.4, Fig. 4, Fig. S1–S4 | Sydnor et al. 2021, https://github.com/PennLINC/S-A_ArchetypalAxis |
| HCP group functional connectivity (Glasser 360) | Fig. S1, S2 | ENIGMA Toolbox |
| Canonical gradients | Fig. S2 | BrainSpace |

## Repository layout

| Folder | Content | Manuscript |
|---|---|---|
| `config/` | paths and software locations | — |
| `environment/` | Python packages, external software | — |
| `01_anatomy_layers/` | equivolumetric cortical depths (SUMA, LayNii) | Methods 4.3 |
| `02_registration_rois/` | T1w → EPI registration (ANTs); HCP-MMP labels, depths and 5% outer-pial band in functional space; non-overlapping parcel × depth masks | Methods 4.3–4.4 |
| `03_timeseries/` | tSNR cleaning; parcel × depth time series after 5% outer-pial exclusion | Methods 4.4 |
| `04_rdcm_six_node/` | six-node EVC–cauPFC rDCM; FF, FB and H; composite-signal QC | Results 2.1, Fig. 1, Tables S1–S2 |
| `05_rdcm_nine_node/` | nine-node EVC–PPC–cauPFC rDCM; pathway statistics; pair omission | Results 2.4–2.5, Fig. 4, Tables S6–S7 |
| `06_voxel_matching/` | voxel-count matching across nodes (100 repetitions) | Results 2.1–2.3, Fig. S3, Tables S1, S4 |
| `07_recovery/` | parameter-recovery simulations | Results 2.2, Fig. 2, Table S3 |
| `08_behaviour/` | coupling–behaviour correlations, permutation tests, leave-one-out | Results 2.3, Fig. 3, Table S4 |
| `09_SA_axis/` | S–A positions of the composites and their separations | Results 2.4, Fig. 4a, Table S5 |
| `10_functional_gradients/` | functional gradients from our data; split-half reliability | Fig. S2, S8 |
| `11_microstructure/` | depth-resolved qT1, MPC gradients, qT1/MPC–coupling associations, MPC–functional alignment | Results 2.6, Fig. S4–S7, Tables S8–S9 |
| `12_figures/` | figure scripts and shared plotting style | Figs 4–5, S1–S8 |

`RUN_ORDER.md` gives the commands in order; `RESULTS_MAP.md` lists, for every reported number, table and figure, the script and output file it comes from. `PROVENANCE.tsv` records where each file originated and the SHA-256 prefix of the original.

## Setup

```bash
cp config/paths.sh.example config/paths.sh   # edit the paths
source config/paths.sh
pip install -r environment/requirements.txt
```

All scripts read the project location from `PROJECT_ROOT`; MATLAB scripts read `PROJECT_ROOT`, `SPM_PATH` and `TAPAS_PATH` with `getenv`. Derivatives are written below `$PROJECT_ROOT/derivatives_0331/`. Folder `06_voxel_matching/` and the recovery and QC subfolders take their paths from a `config.json` next to the scripts; replace the `/path/to/...` placeholders there.

External software is listed in `environment/SOFTWARE.md`.

## Participants

Fifteen participants (`sub-01`–`sub-14`, `sub-16`) enter every analysis; the list is set in each script's defaults.

## License

[none]

## Contact

[corresponding author]
