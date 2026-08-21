# Spatial Breast Cancer Niche Clustering

This project identifies tissue regions with similar chemotherapy responses
without using known group labels.

The analysis is a small re-analysis of the **spatial
transcriptomics** data from Túrós *et al.* Spatial transcriptomics measures gene
activity while keeping information about where each measurement came from in a
tissue section. The project applies **unsupervised machine learning**
(methods that search for structure in data without a known answer to predict) to
a concrete biological question.

The analysis tests whether **niches** (tissue compartments with related cellular
and molecular properties) that change in similar ways after chemotherapy form
reproducible response groups. The result supports the broad tissue remodelling
reported in the source paper. It does not establish a new biological mechanism.

- Paper: [Nature Communications (2026)](https://doi.org/10.1038/s41467-026-74125-6)
- Data: [Zenodo 15102983](https://doi.org/10.5281/zenodo.15102983)

## Data

The input consists of mouse **Visium** files. Visium is a spatial-transcriptomics
technology that divides a tissue section into small measured areas called
spots. The files contain spatial coordinates, H&E images (haematoxylin-and-eosin
stained tissue images used to inspect histology), continuous Chrysalis weights,
cell2location estimates (computational estimates of cell-type abundance), and
pathway scores (numerical summaries of related biological processes).

The cisplatin analysis contains 27,931 spots from 13 tumours. Following the
source paper, N3 and N6 are removed because they are likely technical
compartments rather than biological niches. This leaves 11 biological niches
for the machine-learning analysis.

## Workflow

![Spatial niche clustering workflow](assets/readme/spatial_niche_clustering_pipeline.png)

*Figure 1. From deposited Visium sections to biologically interpreted response
patterns.*

Figure 1 follows the full reasoning path: tissue measurements become tumour-level
profiles, the profiles become response groups, and the groups are checked before
they receive a biological interpretation. One tumour × niche profile is treated
as one **biological observation** (one independent unit used by the analysis).
Treated tumours are compared with primary tumours from the same parental tumour
and experimental batch. The final matrix contains 11 niches and 24 standardised
stage–feature contrasts.

## From tissue spots to model input

![Visium spots across treatment stages](assets/readme/visium_spot_sections.png)

*Figure 2. Spatial distribution of the two response macro-patterns across the
four treatment stages.*

As shown in Figure 2, red marks the primary/recurrent-associated response group,
while blue marks the **MRD-associated** immune–stromal group. MRD means minimal
residual disease: tumour cells and their surrounding tissue that remain after
treatment. The four large panels show one complete stage series selected by a
fixed rule; they are not repeated sections from the same tumour over time. This
spot-level view gives spatial context to the tumour-level profiles used by the
model.

A Visium spot contains several cells. A coloured circle is therefore neither a
single cell nor a definitive niche label. **Chrysalis** is the upstream method
used by the authors to represent tissue compartments. It gives each spot
continuous weights, so one spot can partly belong to several niches instead of
being forced into only one.

For each tumour and niche, the workflow calculates a **weighted mean** (an
average in which spots contribute according to their niche weights) for
abundance and seven molecular measurements. Abundance is transformed with CLR,
or centred log-ratio, because the niche proportions are **compositional**: they
are parts of one whole and must sum to one.

The two residual-disease stages, MRD7 and MRD12, and recurrent tumours are then
compared with same-batch primary tumours. This prevents **pseudoreplication**
(incorrectly treating many spots from the same tumour as many independent
tumours). The result is one trajectory per niche: eight measurements at three
stages, or 24 contrasts. A contrast is a difference between a treated condition
and its matched primary reference. The 24 columns are **standardised** (put on a
common numerical scale) before clustering, so one measurement cannot dominate
only because its raw values are larger.

The formulas below do not add a separate mathematical layer to the project.
They make each transformation explicit, so the biological meaning of every row
and column entering the model can be checked.

For tumour $t$, niche $n$ and molecular feature $f$, the weighted tumour-level
profile is

$$
x_{tnf}=\frac{\sum_{s=1}^{S_t} w_{sn}x_{sf}}
{\sum_{s=1}^{S_t} w_{sn}},
$$

where $s$ denotes a Visium spot, $w_{sn}$ is its continuous Chrysalis weight for
niche $n$, and $x_{sf}$ is its **feature** value. A feature is one numerical
measurement supplied to the model. The corresponding relative niche abundance
is

$$
p_{tn}=\frac{\sum_s w_{sn}}{\sum_m\sum_s w_{sm}}.
$$

Because all niche abundances from one tumour sum to one, they are transformed
with the centred log-ratio (CLR):

$$
\mathrm{CLR}(p_{tn})=\log\left(
\frac{p_{tn}+\varepsilon}
{\left[\prod_{m=1}^{N}(p_{tm}+\varepsilon)\right]^{1/N}}
\right).
$$

Here, $\varepsilon$ is a small **pseudocount** (a tiny added value that makes the
logarithm defined when an abundance is zero). Each feature is first expressed
relative to its variation among primary tumours:

$$
x^{\ast}_{tnf}=\frac{x_{tnf}}{\sigma_{f,\mathrm{primary}}}.
$$

A treated tumour is then compared with the mean primary value from the same
batch. For treatment stage $r$, this gives

$$
\Delta_{nfr}=\frac{1}{|T_r|}\sum_{t\in T_r}
\left(x^{\ast}_{tnf}-\overline{x}^{\ast}_{nf,\mathrm{primary},b(t)}\right),
$$

where $T_r$ is the set of treated tumours at stage $r$ and $b(t)$ is the batch
of tumour $t$. Finally, each of the 24 contrast columns is standardised across
niches:

$$
z_{nj}=\frac{\Delta_{nj}-\mu_j}{\sigma_j}.
$$

The resulting matrix $Z\in\mathbb{R}^{11\times24}$ is the direct input to the
unsupervised model: 11 niche rows described by 24 response columns.

## Unsupervised model

![Ward clustering model choice](assets/readme/model_choice_explainer.png)

*Figure 3. Dataset constraints and comparison of candidate unsupervised
models.*

Figure 3A summarises the small unlabelled 11 × 24 input. **Unlabelled** means
that no correct response group is supplied in advance. Figure 3B compares the
main candidate methods and explains why Ward clustering is selected.

There are no known response-group labels, so this is unsupervised learning.
**Hierarchical clustering** builds a tree by progressively joining similar
objects. Ward's method is well suited to this very small input of 11 niches: it
is deterministic (the same input gives the same result), needs no train/test
classifier split (a split used when training a model to predict known classes),
and leaves a **dendrogram** (a tree showing every merge) that can be inspected.
At each step, it joins the pair that causes the smallest increase in
within-cluster variance, meaning the smallest increase in variation inside the
new group.

**PCA, or principal component analysis**, is used only to display the
24-dimensional result on two axes. PCA summarizes the main directions of
variation, but the clustering itself still uses all 24 standardised contrasts.

The alternatives are useful, but less appropriate for the main analysis.
K-means is a reasonable sensitivity check, yet it starts from random group
centres and does not produce a hierarchy. A Gaussian mixture would estimate too
many covariance parameters (values describing how features vary together) from
only 11 objects. Density clustering is unreliable with so few points, and a
neural network would have enough flexibility to memorize them rather than learn
a general pattern. Ward clustering is therefore chosen because its assumptions
and output match the size and purpose of this dataset.

The model receives only the matrix $Z$ and produces one **cluster index** (the
numeric group assigned by the algorithm) for each niche:

$$
Z\in\mathbb{R}^{11\times24}
\quad\longrightarrow\quad
c=(c_1,\ldots,c_{11}),\qquad c_i\in\{1,\ldots,k\}.
$$

There is no known target value $y$ to predict. Ward clustering starts with one
niche per group and repeatedly merges the pair $A,B$ that minimises

$$
\Delta(A,B)=\frac{|A||B|}{|A|+|B|}
\left\|\boldsymbol{\mu}_A-\boldsymbol{\mu}_B\right\|_2^2,
$$

where $|A|$ and $|B|$ are group sizes and $\boldsymbol{\mu}_A$ and
$\boldsymbol{\mu}_B$ are their mean 24-dimensional profiles. The Euclidean norm
$\|\cdot\|_2$ measures the straight-line distance between these mean profiles.
The complete quantity is the increase in total within-group squared variation
caused by the merge.

PCA does not change these assignments. It finds new directions, called
principal components, that capture decreasing amounts of variation. Formally,
it finds vectors
$\mathbf{v}_\ell$ satisfying

$$
\frac{Z^\mathsf{T}Z}{n-1}\mathbf{v}_\ell
=\lambda_\ell\mathbf{v}_\ell,
\qquad
\mathbf{t}_\ell=Z\mathbf{v}_\ell,
$$

then uses the first two score vectors $\mathbf{t}_1$ and $\mathbf{t}_2$ as the
two axes of the PCA plots.

## Selecting the number of patterns

![Ward solutions from k=2 to k=6](assets/readme/clustering_k_comparison.png)

*Figure 4. Ward solutions obtained by cutting the same hierarchical tree from
`k = 2` to `k = 6`.*

Figure 4 compares the same Ward tree after cutting it into `k = 2–6` groups,
where `k` is the requested number of clusters. The PCA panels show these groups
in two dimensions, while Ward still uses all 24 contrasts. The selection combines
group separation with minimum cluster size.

`k = 2` has the largest **silhouette score** (`0.439`) and balanced groups of
six and five niches. The silhouette measures whether an object is closer to its
own group than to another group. Larger values of `k` reduce this score and
create groups containing only one or two niches, which are too small to support
credible response macro-patterns here.

For niche $i$, $a(i)$ is its mean distance from the other niches in its own
group. The value $b(i)$ is the smallest mean distance to any other group:

$$
a(i)=\frac{1}{|C_i|-1}
\sum_{\substack{j\in C_i\\j\ne i}}d(i,j),
\qquad
b(i)=\min_{C\ne C_i}\frac{1}{|C|}\sum_{j\in C}d(i,j).
$$

Its silhouette is therefore

$$
s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}},
\qquad
S=\frac{1}{n}\sum_{i=1}^{n}s(i).
$$

The reported value `0.439` is the mean $S$ over all 11 niches. The formula
rewards small distances within a group and large distances between groups. It
supports `k = 2`, but it is evidence about separation rather than proof that two
groups are the only possible biological description.

![Detailed selected k=2 solution](assets/readme/clustering_k2_detail.png)

*Figure 5. Selected two-group dendrogram and complete standardised input
matrix.*

Figure 5A shows the complete Ward dendrogram and its two main branches. Figure
5B shows the full 11 × 24 matrix used by the model. This **heatmap** represents
each numerical value with a colour: vivid blue indicates a relative decrease,
and vivid red indicates a relative increase after standardisation. Biological
names are added only after the numeric groups are fixed, which prevents the
names from influencing the clustering.

## Biological annotation

The algorithm returns numbers, not biological explanations. Biological names
are therefore assigned only after clustering by comparing each numeric group
with the paper's cell-type associations, niche gene programmes, histology, and
treatment response. This step is **biological annotation** (connecting a
data-derived pattern to existing biological evidence). The original `N`
identifiers remain visible for **provenance**, meaning that every interpretation
can be traced back to its source niche.

In the table, **TME** means tumour microenvironment: the non-tumour cells,
extracellular material, and signals surrounding tumour cells.

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

Figure 6A–C examines model choice, silhouette separation (how well each niche
fits its group), and pairwise consensus (how often two niches remain together
across repeated analyses). Figure 6D tests **robustness** (whether the conclusion
survives a controlled change) after removing one tumour or feature family.
Figure 6E compares the cisplatin arm with the related TAC treatment regimen.
Figure 6F compares the observed structure with shuffled data. These checks
target different possible failure modes; they are not measures of predictive
accuracy because this model does not predict known labels.

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

Taken together, these checks support the stability of a coarse two-pattern
solution. A **coarse solution** captures the broadest structure and does not
claim that every niche inside a group behaves identically. The values are not
test-set accuracy, and TAC is a related treatment arm from the same study rather
than a fully independent external dataset.

- **Silhouette** compares each niche's mean distance to its own group with its
  distance to the other group. Values near 1 indicate clear separation, near 0
  indicate a boundary, and negative values suggest possible misassignment.
- **Consensus** is the fraction of 1,000 **refits** (new analyses after
  resampling the data) in which each pair of niches is placed together. It
  measures how often the same groups return.
- **Adjusted Rand index (ARI)** compares two complete groupings while
  correcting agreement expected by chance. `1` means identical groups;
  values near `0` mean chance-level agreement.
- **Leave-one-tumour-out and feature-out ARI** recompute the full analysis after
  removing one tumour or one feature family. They test dependence on a single
  sample or measurement.
- **Spearman correlation** compares the rank order of cisplatin and TAC response
  trajectories for the same niche. Rank order means first, second, third, and so
  on rather than the exact raw values. This tests cross-arm agreement, not
  causality.
- **Permutation p-value** shuffles features within columns 5,000 times to create
  data with no preserved niche structure, then asks how often these random data
  reach the observed best silhouette. Here
  `p = (0 + 1) / (5000 + 1) = 0.0002`.

For $B=1000$ **bootstrap refits** (repeated analyses built from tumour-level
samples drawn with replacement), pairwise consensus is calculated as

$$
C_{ij}=\frac{1}{B}\sum_{b=1}^{B}
\mathbf{1}\left[c_i^{(b)}=c_j^{(b)}\right].
$$

$C_{ij}=1$ means niches $i$ and $j$ remain together in every refit. The
stability of niche $i$ is the mean consensus with the other niches assigned to
its final group $G_i$:

$$
\mathrm{Stability}_i=
\frac{1}{|G_i|-1}
\sum_{\substack{j\in G_i\\j\ne i}}C_{ij}.
$$

The adjusted Rand index uses a **contingency table** (a table counting how two
sets of group assignments overlap). If $n_{uv}$ is the number of niches shared
by group $u$ in the first clustering and group $v$ in the second, with row sums
$a_u$ and column sums $b_v$, then

$$
\begin{aligned}
I &= \sum_{uv}\binom{n_{uv}}{2}, &
A &= \sum_u\binom{a_u}{2},\\
B &= \sum_v\binom{b_v}{2}, &
E &= \frac{AB}{\binom{n}{2}}.
\end{aligned}
$$

With $I$ as the observed pair agreement and $E$ as the agreement expected by
chance,

$$
\mathrm{ARI}=\frac{I-E}{\tfrac{1}{2}(A+B)-E}.
$$

This chance correction is why ARI is more informative than simply counting the
fraction of unchanged niche labels.

For two response rankings without ties (equal ranks), Spearman correlation can
be written as

$$
\rho_s=1-\frac{6\sum_{i=1}^{n}d_i^2}{n(n^2-1)},
$$

where $d_i$ is the difference between the two ranks for niche $i$. The generic
permutation p-value uses the same **finite-sample correction** (adding one to
avoid reporting an impossible exact zero from a limited number of shuffles) as
the value reported above:

$$
p=\frac{1+\sum_{b=1}^{B}
\mathbf{1}\left[S_b^{\mathrm{perm}}\ge S_\mathrm{obs}\right]}{B+1}.
$$

## Result

| Macro-pattern | Niches | Interpretation |
|---|---|---|
| MRD-associated immune–stromal remodelling | N1, N7, N9, N11, N12 | immune, macrophage, humoral and stromal programmes become prominent during residual disease |
| Primary/recurrent tissue architecture | N0, N2, N4, N5, N8, N10 | mainly tumour and peritumour programmes diminish or remodel during MRD and return at recurrence |

The model separates two broad response trajectories. One group is most closely
associated with immune and stromal remodelling during residual disease. The
other mainly follows primary and recurrent tissue architecture.

N2 is an important biological caveat. **EMT, or epithelial-to-mesenchymal
transition**, is a cell-state programme linked here to motility, plasticity, and
treatment tolerance. The original study describes the EMT niche as persistent
and chemotherapy-tolerant. Its placement in the second machine-learning group
means that its complete 24-feature trajectory is numerically more similar to
that group. It is not proof that N2 abundance strongly contracts.

## Project layout

The repository separates source code, tests, input data, generated results, and
README figures. This makes it possible to inspect one part of the workflow
without mixing it with the others.

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

Create the results and open the interactive application with:

```powershell
git clone https://github.com/ym22000/ML-ward-clustering-tnbc-niche-perturbations-.git
cd ML-ward-clustering-tnbc-niche-perturbations-
python -m pip install -r requirements.txt
python run_analysis.py --h5ad-dir data\raw\extracted\visium_mouse\processed\mouse_main --treatment cisplatin_6mg/kg
python -m streamlit run app.py
```

The analysis command rebuilds the perturbation profiles, clustering, checks,
tables, and figures from the processed input. **Streamlit** is a Python library
for small interactive data applications; the final command opens the app at
`http://localhost:8507`.

## Limits

This is a reproducible learning project and a small research re-analysis, not a
clinical decision system. Its main limits are:

- No clinical classifier is trained. The output groups niches; it does not
  predict treatment response for a new patient.
- Biological annotation is interpretive and does not alter the clustering.
- Cell-type summaries used as machine-learning features cannot also serve as
  independent validation, because they already contribute information to the
  model input.
- Histology and paper-derived gene programmes provide less circular evidence,
  although they still come from the same study.
- The second treatment arm is internal rather than external validation. A new
  cohort from another experiment would provide a stronger test of
  generalisation.

## References and documentation

The workflow uses the processed mouse Visium objects released with the study.
It does not rerun Chrysalis or cell2location. Their compartment scores,
cell-type estimates, and biological annotations were produced upstream by the
authors and are treated here as input data. The perturbation profiles, Ward
clustering, and reliability analyses are implemented independently in this
repository. This distinction makes clear which results are reused and which are
recomputed here.

### Study and data

- Túrós *et al.* (2026), [Spatiotemporal organisation of residual disease in
  mouse and human BRCA1-deficient mammary tumours and breast
  cancer](https://doi.org/10.1038/s41467-026-74125-6), *Nature
  Communications*. This is the source paper for the biological question,
  experimental design, and niche annotations.
- [Processed spatial transcriptomics data](https://doi.org/10.5281/zenodo.15102983),
  Zenodo record 15102983. The mouse Visium `.h5ad` files analysed in this
  repository come from this deposit.
- [GEO accession GSE299631](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE299631),
  the raw sequencing record reported by the study.
- [Authors' residual-disease code repository](https://github.com/rottenberglab/residual-disease),
  cited in the paper's Code availability section and released for the original
  published analyses.
- Túrós *et al.* (2024), [Chrysalis: decoding tissue compartments in spatial
  transcriptomics with archetypal
  analysis](https://doi.org/10.1038/s42003-024-07165-7), *Communications
  Biology*. Chrysalis generated the continuous niche scores available in the
  processed objects; a continuous score allows partial niche membership.
- Kleshchevnikov *et al.* (2022), [Cell2location maps fine-grained cell types
  in spatial transcriptomics](https://doi.org/10.1038/s41587-021-01139-4),
  *Nature Biotechnology*. Cell2location generated the cell-type estimates used
  by the original study to help interpret the niches.

### Statistical methods

These references define the main statistical tools used in the workflow. They
are included so that each model choice and reliability check can be connected
to its original method rather than treated as a software option.

- Ward (1963), [Hierarchical Grouping to Optimize an Objective
  Function](https://doi.org/10.1080/01621459.1963.10500845), introduced Ward's
  minimum-variance hierarchical clustering criterion.
- Aitchison (1982), [The Statistical Analysis of Compositional
  Data](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x), provides the
  framework for analysing proportions through log-ratios rather than raw
  Euclidean differences.
- Rousseeuw (1987), [Silhouettes: A graphical aid to the interpretation and
  validation of cluster analysis](https://doi.org/10.1016/0377-0427(87)90125-7),
  defines the silhouette used to compare values of $k$.
- Hubert and Arabie (1985), [Comparing
  partitions](https://doi.org/10.1007/BF01908075), defines the adjusted Rand
  index used to compare cluster assignments after refitting.
- Monti *et al.* (2003), [Consensus Clustering: A Resampling-Based Method for
  Class Discovery and Visualization of Gene Expression Microarray
  Data](https://doi.org/10.1023/A:1023949509487), provides the resampling logic
  behind the consensus matrix.
- Efron (1979), [Bootstrap Methods: Another Look at the
  Jackknife](https://doi.org/10.1214/aos/1176344552), is the basis for the
  tumour-level bootstrap confidence intervals and stability checks.
- Spearman (1904), [The Proof and Measurement of Association between Two
  Things](https://doi.org/10.2307/1412159), introduced the rank correlation
  used to compare niche response orders between treatment arms.
- Phipson and Smyth (2010), [Permutation P-values Should Never Be
  Zero](https://doi.org/10.2202/1544-6115.1585), supports the finite-sample
  correction $(b+1)/(B+1)$ used for the permutation p-values.
- McInnes *et al.* (2018), [UMAP: Uniform Manifold Approximation and
  Projection](https://doi.org/10.21105/joss.00861), describes the nonlinear
  two-dimensional projection used only for visual inspection.

### Software documentation

The following documentation describes the Python tools used to store the data,
perform the analysis, and create the figures and application.

- Data objects and input: [AnnData](https://anndata.readthedocs.io/en/stable/)
  and [Scanpy `read_h5ad`](https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_h5ad.html).
- Tables and numerical operations: [pandas](https://pandas.pydata.org/docs/)
  and [NumPy](https://numpy.org/doc/stable/).
- Hierarchical clustering: [SciPy `linkage`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).
- Scaling and dimension reduction: scikit-learn
  [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
  [`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
  and [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
- Reliability metrics: scikit-learn
  [`silhouette_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
  and [`adjusted_rand_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html),
  with pandas [`Series.corr`](https://pandas.pydata.org/docs/reference/api/pandas.Series.corr.html)
  for Spearman correlation.
- Two-dimensional visualisation: [UMAP documentation](https://umap-learn.readthedocs.io/en/latest/)
  and [UMAP parameter guide](https://umap-learn.readthedocs.io/en/latest/parameters.html).
- Figures and interface: [Matplotlib](https://matplotlib.org/stable/),
  [seaborn](https://seaborn.pydata.org/) and
  [Streamlit](https://docs.streamlit.io/).
- Configuration files: [PyYAML documentation](https://pyyaml.org/wiki/PyYAMLDocumentation).

### Reproducible software environment

The analyses, figures, and application were generated and checked with Python
3.14.0 and the following direct dependencies:

```text
anndata 0.12.9          matplotlib 3.10.7    numpy 2.3.5
pandas 2.3.3            Pillow 12.0.0         PyYAML 6.0.3
Scanpy 1.12             scikit-learn 1.7.2   SciPy 1.16.3
seaborn 0.13.2          Streamlit 1.61.1     umap-learn 0.5.11
```

These exact runtime versions are **pinned** (fixed to the versions used here) in
[`requirements.txt`](requirements.txt). The test environment adds pytest 9.1.1
through [`requirements-dev.txt`](requirements-dev.txt). Recording the complete
environment helps another person reproduce the reported result and prevents a
later package update from silently changing the analysis.
