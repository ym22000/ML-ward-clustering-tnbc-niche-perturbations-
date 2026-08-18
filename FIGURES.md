# Figures

README figures are stored in `assets/readme/` as high-resolution PNG files.
Analysis copies are stored in `results/figures/`.

| File | Content |
|---|---|
| `spatial_niche_clustering_pipeline.png` | complete workflow, including biological annotation |
| `visium_spot_sections.png` | four large H&E sections with red/blue Visium response-group spots |
| `model_choice_explainer.png` | reason for Ward clustering and comparison with alternatives |
| `clustering_k_comparison.png` | fixed PCA display of the Ward cuts from `k = 2` to `k = 6` |
| `clustering_k2_detail.png` | selected dendrogram and vivid heatmap of the full model input |
| `reliability_metrics_explained.png` | silhouette, consensus, ARI, cross-arm agreement and permutation test |

Additional PNG outputs include `figure_main.png`, `tissue_sections.png`,
`pattern_trajectories.png`, `perturbation_heatmap.png`, `pca_patterns.png`,
`abundance_effects.png`, `model_diagnostics.png`, `tumour_niche_spatial.png`,
`tumour_niche_umap.png` and `ml_model_summary.png`.

## Colour system

- blue: MRD-associated immune–stromal remodelling;
- red: primary/recurrent tissue architecture;
- individual vivid colours: biological niche identities in UMAP and spatial
  niche projections.

All legends retain the Chrysalis ID and add the biological name, for example
`N2 · EMT tumour niche`. N2 is described in captions as a persistent EMT
exception rather than as a simple treatment-sensitive contraction.
