# Spatial Breast Cancer Niche Clustering

This project is a small re-analysis of the spatial transcriptomics data from
Túrós *et al.* It was developed to learn how an unsupervised machine-learning
result is built, tested and connected to biological evidence.

The analysis asks whether niches with similar abundance and molecular changes
after chemotherapy form reproducible response groups. It confirms the broad
remodelling described by the paper; it does not claim a new mechanism.

- Paper: [Nature Communications (2026)](https://doi.org/10.1038/s41467-026-74125-6)
- Data: [Zenodo 15102983](https://doi.org/10.5281/zenodo.15102983)

## Data

The mouse Visium files contain spatial coordinates, H&E images, continuous
Chrysalis weights, cell2location estimates and pathway scores. The cisplatin
analysis contains 27,931 spots from 13 tumours. N3 and N6 are removed as likely
technical compartments, following the paper, leaving 11 biological niches.

## Workflow

![Spatial niche clustering workflow](assets/readme/spatial_niche_clustering_pipeline.png)

*Figure 1. From deposited Visium sections to biologically interpreted response
patterns.*

Figure 1 follows the complete analysis from tissue measurements to response
groups, stability checks and biological annotation. One tumour × niche profile
is one biological observation. Treated tumours are compared with primary
tumours from the same parental tumour and batch. The final matrix contains 11
niches and 24 standardised stage–feature contrasts.

## From tissue spots to model input

![Visium spots across treatment stages](assets/readme/visium_spot_sections.png)

*Figure 2. Spatial distribution of the two response macro-patterns across the
four treatment stages.*

As shown in Figure 2, red marks the primary/recurrent-associated response group
and blue marks the MRD-associated immune–stromal group. The four large panels
show one complete stage series selected by a fixed rule; they are not
longitudinal sections from one tumour. This spot-level view gives spatial
context to the tumour-level profiles used by the model.

A Visium spot is a small measured tissue area containing several cells. The
coloured circle is therefore not a single cell and not a definitive niche
label. Chrysalis gives each spot continuous weights: one spot can partly belong
to several niches. For each tumour and niche, the workflow computes a weighted
mean of abundance and seven molecular measurements. Abundance is
CLR-transformed because niche proportions are compositional.

MRD7, MRD12 and recurrent tumours are then compared with same-batch primary
tumours. This avoids treating thousands of spots from one section as thousands
of independent biological replicates. The result is one trajectory per niche:
eight measurements at three stages, or 24 contrasts. The 24 columns are
standardised before clustering so one measurement cannot dominate only because
its numerical scale is larger.

## Unsupervised model

![Ward clustering model choice](assets/readme/model_choice_explainer.png)

*Figure 3. Dataset constraints and comparison of candidate unsupervised
models.*

Figure 3A summarises the small unlabelled 11 x 24 input. Figure 3B compares the
main candidate methods and explains the choice of Ward clustering.

There are no known response-group labels, so this is unsupervised learning. Ward
hierarchical clustering is suited to the very small input of 11 objects: it is
deterministic, needs no train/test classifier split and leaves a tree in which
every merge can be checked. At each step it joins the pair producing the smallest increase
in within-cluster variance. PCA is used only to display the 24-dimensional
result; clustering itself uses all 24 standardised contrasts.

K-means is a reasonable sensitivity check but has random starting centres and
no hierarchy. A Gaussian mixture estimates too many covariance parameters for
11 objects. Density clustering is unreliable with so few points, and a neural
network would mainly memorise them.

## Selecting the number of patterns

![Ward solutions from k=2 to k=6](assets/readme/clustering_k_comparison.png)

*Figure 4. Ward solutions obtained by cutting the same hierarchical tree from
`k = 2` to `k = 6`.*

Figure 4 compares the same Ward tree at `k = 2–6`; the PCA panels show the
groups in two dimensions, while clustering uses all 24 contrasts. PCA
(principal component analysis) is only a simpler view of the main variation.
The decision combines separation and minimum cluster size. `k = 2` has the largest silhouette
(`0.439`) and balanced groups of six and five niches. Larger values reduce the
silhouette and create groups of only one or two niches, which are not credible
macro-patterns here.

![Detailed selected k=2 solution](assets/readme/clustering_k2_detail.png)

*Figure 5. Selected two-group dendrogram and complete standardised input
matrix.*

Figure 5A shows the complete Ward dendrogram and its two main branches. Figure
5B shows the full 11 × 24 matrix used by the model; the vivid blue-to-red
gradient represents relative decreases and increases after standardisation.
Biological names are added only after these numeric groups are fixed.

## Biological annotation

Biological names are assigned after clustering. They come from the paper's
cell-type associations, niche gene programmes, histology and treatment response.
The original `N` identifiers remain visible for provenance.

| ID | Biological name | Class | Main programme or context |
|---|---|---|---|
| N0 | Subset-specific tumour | tumour | rare, incompletely assigned tumour compartment |
| N1 | Macrophage-rich niche | immune TME | macrophage/monocyte and TREM2–APOE programmes |
| N2 | EMT tumour niche | tumour | EMT, motility, adhesion and drug-tolerant plasticity |
| N4 | Fibrotic reactive stroma | stromal TME | collagen and extracellular-matrix organisation |
| N5 | Proliferating tumour | tumour | cell cycle, DNA replication and E2F/MYC |
| N7 | Fibro-inflammatory TME | stromal/immune TME | connective tissue and immune infiltration |
| N8 | Chemo-depleted tumour | tumour | selectively lost after chemotherapy |
| N9 | Contractile stroma | stromal TME | smooth-muscle and structural programmes |
| N10 | Hypoxic tumour | tumour | hypoxia and metabolic stress near necrosis |
| N11 | Activated immune niche | immune TME | macrophage, defence and complement programmes |
| N12 | Plasma-cell/humoral niche | immune TME | plasma cells and B-cell/humoral activity |

The complete annotation table is written to
[`results/tables/niche_annotations.csv`](results/tables/niche_annotations.csv).

## Model checks

![Reliability metrics and their limits](assets/readme/reliability_metrics_explained.png)

*Figure 6. Complementary checks of the selected two-group solution.*

Figure 6A-C examines model choice, silhouette separation (how well each niche
fits its group) and pairwise consensus (how often two niches remain together
across repeated runs). Figure 6D tests robustness to tumour and feature
removal, Figure 6E compares the cisplatin and TAC arms, and Figure 6F compares
the observed result with shuffled data. They test different failure modes
rather than predictive accuracy.

| Check | Result |
|---|---:|
| Selected patterns | 2 |
| Silhouette (separation between groups) | 0.439 |
| PCA variance in two dimensions (variation retained in the plot) | 80.2% |
| Mean bootstrap stability (average repeatability across refits) | 1.00 |
| Leave-one-tumour-out ARI (agreement after removing one tumour) | 1.00 |
| Minimum feature-removal ARI (agreement after removing one measurement family) | 1.00 |
| TAC-arm agreement ARI (agreement between treatment arms) | 1.00 |
| Median cross-arm Spearman correlation (rank agreement) | 0.814 |
| Cluster permutation test (comparison with shuffled data) | 0.0002 |

These checks support the stability of a coarse two-pattern solution. They are
not test-set accuracy, and TAC is a related treatment arm from the same study.

- **Silhouette** compares each niche's mean distance to its own group with its
  distance to the other group. Values near 1 indicate clear separation, near 0
  indicate a boundary, and negative values suggest possible misassignment.
- **Consensus** is the fraction of 1,000 refits (new runs after resampling the
  data) in which each pair of niches is placed together. It measures how often
  the same groups return.
- **Adjusted Rand index (ARI)** compares two complete groupings while
  correcting agreement expected by chance. `1` means identical groups;
  values near `0` mean chance-level agreement.
- **Leave-one-tumour-out and feature-out ARI** recompute the full analysis after
  removing one tumour or one feature family. They test dependence on a single
  sample or measurement.
- **Spearman correlation** compares the rank order of cisplatin and TAC response
  trajectories for the same niche. It tests cross-arm agreement, not causality.
- **Permutation p-value** shuffles features within columns 5,000 times and asks
  how often random data reach the observed best silhouette. Here
  `p = (0 + 1) / (5000 + 1) = 0.0002`.

## Result

| Macro-pattern | Niches | Interpretation |
|---|---|---|
| MRD-associated immune–stromal remodelling | N1, N7, N9, N11, N12 | immune, macrophage, humoral and stromal programmes become prominent during residual disease |
| Primary/recurrent tissue architecture | N0, N2, N4, N5, N8, N10 | mainly tumour and peritumour programmes diminish or remodel during MRD and return at recurrence |

N2 is a required biological caveat. The original study describes the EMT niche
as persistent and chemotherapy-tolerant. Its placement in the second ML group
reflects similarity of the complete multivariate trajectory, not proof that its
abundance strongly contracts.

## Project layout

```text
niche_perturbation_patterns/
├── README.md
├── PIPELINE.md
├── FIGURES.md
├── RESULTS.md
├── run_analysis.py
├── app.py
├── config.yaml
├── src/
│   ├── annotations.py
│   ├── core.py
│   ├── explain_figures.py
│   ├── figures.py
│   ├── readme_figures.py
│   └── visium.py
├── tests/
├── data/
│   ├── raw/
│   └── processed/
├── assets/readme/
└── results/
    ├── figures/
    └── tables/
```

## Run

```powershell
cd C:\Users\pcyou\Desktop\bioinfo_llm\niche_perturbation_patterns
python -m pip install -r requirements.txt
python run_analysis.py --h5ad-dir data\raw\extracted\visium_mouse\processed\mouse_main --treatment cisplatin_6mg/kg
python -m streamlit run app.py
```

The app opens at `http://localhost:8507`.

## Limits

- No clinical classifier is trained.
- Biological annotation is interpretive and does not alter the clustering.
- Cell-type summaries used as ML features cannot serve as independent validation.
- Histology and paper-derived gene programmes provide the less circular evidence.
- The second treatment arm is internal rather than external validation.
