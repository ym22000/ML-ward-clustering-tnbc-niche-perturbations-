# Pipeline

## 1. Published spatial data

The input is one processed `h5ad` object per Visium section. It contains spot
coordinates, an H&E image, sample metadata, continuous Chrysalis weights,
cell2location estimates and pathway scores. The project uses these deposited
summaries and does not retrain Chrysalis or cell2location.

N3 and N6 are removed as likely technical compartments, following the paper.
The remaining 11 niche weights are renormalised.

## 2. Matched treatment comparison

The cisplatin samples derive from parental tumour 2.1. Only untreated primary
tumours from the same parental tumour are used as reference.

```text
Primary -> MRD day 7 -> MRD day 12 -> Recurrence
```

Treated sections are contrasted with primary sections from the same experimental
batch whenever a matched reference is available.

## 3. Tumour-level profiles

One tumour, not one spatial spot, is treated as an independent biological unit.
For every tumour and niche, continuous Chrysalis weights are used to calculate:

- centred log-ratio niche abundance;
- EMT and proliferation fractions;
- hypoxia pathway activity;
- immune, fibroblast, macrophage and tumour fractions.

This produces 143 tumour × niche profiles.

## 4. Perturbation matrix

Each treated profile is compared with its matched primary reference:

```text
treated tumour × niche profile - matched primary profile
```

Changes from MRD day 7, MRD day 12 and recurrence are concatenated. The resulting
matrix has 11 niches and 24 standardised stage–feature contrasts.

## 5. Unsupervised clustering

Ward hierarchical clustering groups niches with similar multivariate treatment
trajectories. PCA is used only to display the structure in two dimensions.

The algorithm receives numeric profiles but no biological niche names. Candidate
solutions from `k = 2` to `k = 6` are compared using silhouette and minimum
cluster size. The selected solution has two groups.

## 6. Consensus and sensitivity

The clustering is refitted 1,000 times. Each iteration resamples tumours within
stage and batch. In 70% of iterations, one molecular feature family is also
removed. A consensus matrix records how often every pair of niches is assigned
to the same group.

Additional checks include:

- leave-one-tumour-out refitting;
- leave-one-feature-family-out refitting;
- feature-wise permutation testing;
- reproduction in the TAC treatment arm.

## 7. Biological annotation

After the clusters are fixed, `src/annotations.py` joins each Chrysalis ID to its
published biological interpretation:

```text
N2  -> EMT tumour niche
N5  -> proliferating tumour
N10 -> hypoxic tumour
N11 -> activated immune niche
N12 -> plasma-cell/humoral niche
```

The annotation uses published cell-type associations, gene programmes,
histology and temporal response. It does not change the ML assignment.

The two macro-patterns are then described as:

```text
MRD-associated immune–stromal remodelling
Primary/recurrent tissue architecture
```

N2 is reported separately as a persistent EMT exception. This prevents the
macro-pattern label from being mistaken for an individual-niche abundance test.

## 8. Outputs

The workflow produces annotated tissue maps, an all-niche UMAP, a perturbation
heatmap, PCA, temporal trajectories, uncertainty intervals, model diagnostics
and an interactive Streamlit explorer.

| File | Role |
|---|---|
| `run_analysis.py` | complete analysis from processed `h5ad` files |
| `src/visium.py` | data extraction and feature construction |
| `src/core.py` | matched contrasts, clustering and validation |
| `src/annotations.py` | published biological annotation layer |
| `src/figures.py` | analysis figures |
| `src/readme_figures.py` | large repository figures |
| `app.py` | interactive biological and model explorer |
