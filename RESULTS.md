# Results

The cisplatin analysis contains 27,931 tumour spots from 13 tumours: 6 primary,
2 MRD day 7, 3 MRD day 12 and 2 recurrent tumours. The workflow retains the 11
biological Chrysalis compartments described in the paper after excluding N3 and
N6 as likely technical artefacts.

## Unsupervised result

The most stable solution contains two macro-patterns.

| Macro-pattern | Biological compartments | Temporal reading |
|---|---|---|
| MRD-associated immune–stromal remodelling | N1 macrophage-rich; N7 fibro-inflammatory; N9 contractile stroma; N11 activated immune; N12 plasma-cell/humoral | becomes prominent during residual disease and decreases again at recurrence |
| Primary/recurrent tissue architecture | N0 subset-specific tumour; N2 EMT tumour; N4 fibrotic stroma; N5 proliferating tumour; N8 chemo-depleted tumour; N10 hypoxic tumour | dominates primary tumours, is remodelled during MRD and is restored at recurrence |

This supports a reversible shift from a tumour-dominated primary architecture to
an immune–stromal residual-disease environment.

## Biological exception: the EMT niche

N2 should not be interpreted as a simply contracting niche. The paper describes
N2 as an EMT-associated, chemotherapy-tolerant compartment that persists during
MRD. Its ML assignment is based on similarity of the complete 24-feature
trajectory, not on a binary abundance rule.

The biologically appropriate reading is therefore:

> N2 is a persistent EMT exception within the primary/recurrent-associated
> macro-pattern.

## Model checks

| Check | Result |
|---|---:|
| Silhouette for two patterns | 0.439 |
| PCA variance in two dimensions | 80.2% |
| Mean bootstrap stability | 1.000 |
| Stable niches | 11 / 11 |
| Leave-one-tumour-out agreement | 1.000 |
| Leave-one-feature-family-out agreement | 1.000 |
| Cisplatin versus TAC agreement | 1.000 |
| Cluster permutation test | p = 0.0002 |

These checks show that the coarse two-way partition is reproducible in this
dataset. They do not measure prediction accuracy and do not establish causality.

## Biological evidence used for interpretation

The names are supported by four layers from the original study:

1. cell2location cell-type associations;
2. Chrysalis gene programmes and pathway enrichment;
3. histological and spatial localisation;
4. abundance changes across primary, MRD and recurrence.

Cell-type estimates are also part of the ML feature matrix and are therefore
descriptive rather than independent validation. Histology, published niche gene
programmes and the TAC treatment arm provide complementary evidence.

## Limits

- Only two cisplatin tumours are available at MRD day 7 and recurrence.
- TAC comes from the same study and is not an external cohort.
- N0, N7 and N8 do not have completely specific cell-state assignments.
- The original paper includes scRNA-seq, IMC and experimental validation that
  are outside this compact re-analysis.
- Finer three- or four-pattern solutions are less secure.

The appropriate conclusion is that a simple unsupervised workflow recovers and
stress-tests a published, biologically interpretable spatial response pattern.
