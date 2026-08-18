"""Run the complete re-analysis from the authors' processed Visium files."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.annotations import annotate_patterns, annotation_frame
from src.core import (
    abundance_effects, aggregate_spots, bootstrap_stability,
    cluster_diagnostics, cluster_permutation_test, feature_sensitivity, fit_patterns,
    leave_one_tumour_out, partition_silhouette, perturbation_matrix,
    treatment_reproducibility,
)
from src.figures import make_figures
from src.explain_figures import make_explainer_figures
from src.readme_figures import make_readme_figures
from src.visium import export_histology_folder, read_visium_folder, read_visium_h5ad


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Unsupervised tumour-niche re-analysis")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--h5ad", type=Path)
    source.add_argument("--h5ad-dir", type=Path)
    parser.add_argument("--treatment", default="cisplatin_6mg/kg")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    features = cfg["features"]
    stages = cfg["stages"]

    # Read published summaries once, then select a fair primary-versus-treatment arm.
    validation_spots = None
    if args.h5ad_dir:
        all_spots = read_visium_folder(
            args.h5ad_dir, cfg["gene_sets"], exclude_niches=cfg["exclude_niches"]
        )
        spots = _select_arm(all_spots, args.treatment)
        validation_treatment = cfg.get("validation_treatment")
        if validation_treatment and validation_treatment != args.treatment:
            validation_spots = _select_arm(all_spots, validation_treatment)
    else:
        spots = read_visium_h5ad(
            args.h5ad, cfg["gene_sets"], args.treatment, cfg["exclude_niches"]
        )

    stages = [x for x in stages if x in spots["stage"].unique()]

    processed = ROOT / "data" / "processed"
    tables = ROOT / "results" / "tables"
    figures = ROOT / "results" / "figures"
    for folder in [processed, tables, figures]:
        folder.mkdir(parents=True, exist_ok=True)

    # A tumour, not a Visium spot, is the biological unit used by the model.
    spots.to_csv(processed / "spots.csv", index=False)
    profiles = aggregate_spots(spots, features)
    matrix = perturbation_matrix(profiles, features, stages, cfg["baseline"])
    # Consensus clustering asks whether the same niche groups survive resampling.
    stability, consensus, consensus_labels = bootstrap_stability(
        profiles, features, stages, cfg["baseline"], cfg["n_clusters"],
        cfg["n_bootstrap"], cfg["seed"], cfg["feature_dropout"],
    )
    patterns, tree, variance = fit_patterns(
        matrix, cfg["n_clusters"], cfg["seed"], labels=consensus_labels
    )
    patterns["stability"] = stability
    patterns = annotate_patterns(patterns)
    cluster_test = cluster_permutation_test(
        matrix, consensus_labels, n_permutations=5000, seed=cfg["seed"]
    )
    off_diagonal = consensus.to_numpy()[~np.eye(len(consensus), dtype=bool)]
    diagnostics = cluster_diagnostics(matrix, cfg["n_clusters"])
    loo = leave_one_tumour_out(
        profiles, features, stages, cfg["baseline"], cfg["n_clusters"], cfg["seed"],
        reference_labels=consensus_labels,
    )
    sensitivity = feature_sensitivity(
        matrix, cfg["n_clusters"], cfg["seed"], reference_labels=consensus_labels
    )
    effects = abundance_effects(
        profiles, stages, cfg["baseline"], n_bootstrap=2000, seed=cfg["seed"]
    )

    validation_ari = None
    validation_correlations = pd.DataFrame()
    validation_stats = {}
    # TAC is held out from the cisplatin fit and used only as a related check.
    if validation_spots is not None and len(validation_spots):
        validation_profiles = aggregate_spots(validation_spots, features)
        validation_matrix = perturbation_matrix(
            validation_profiles, features, stages, cfg["baseline"]
        )
        validation_stability, validation_consensus, validation_labels = bootstrap_stability(
            validation_profiles, features, stages, cfg["baseline"], cfg["n_clusters"],
            cfg["n_bootstrap"], cfg["seed"] + 10000, cfg["feature_dropout"],
        )
        validation_ari, validation_correlations, validation_stats = treatment_reproducibility(
            matrix, validation_matrix, cfg["n_clusters"], cfg["seed"], n_permutations=5000,
            reference_labels=consensus_labels, validation_labels=validation_labels,
        )
        validation_profiles.to_csv(tables / "tac_niche_profiles.csv", index=False)
        validation_matrix.to_csv(tables / "tac_perturbation_matrix.csv")
        validation_correlations.to_csv(tables / "tac_reproducibility.csv", index=False)
        validation_consensus.to_csv(tables / "tac_bootstrap_consensus.csv")

    # The diagnostic table compares simple alternative cluster resolutions.
    for i, row in diagnostics.iterrows():
        k = int(row.k)
        k_loo = leave_one_tumour_out(profiles, features, stages, cfg["baseline"], k, cfg["seed"])
        k_sensitivity = feature_sensitivity(matrix, k, cfg["seed"])
        diagnostics.loc[i, "mean_loo_ari"] = k_loo.adjusted_rand_index.mean()
        diagnostics.loc[i, "min_feature_ari"] = k_sensitivity.adjusted_rand_index.min()
        if validation_spots is not None and len(validation_spots):
            diagnostics.loc[i, "tac_ari"] = treatment_reproducibility(
                matrix, validation_matrix, k, cfg["seed"]
            )[0]
    selected_mask = diagnostics.selected
    diagnostics.loc[selected_mask, "silhouette"] = partition_silhouette(matrix, consensus_labels)
    diagnostics.loc[selected_mask, "mean_loo_ari"] = loo.adjusted_rand_index.mean()
    diagnostics.loc[selected_mask, "min_feature_ari"] = sensitivity.adjusted_rand_index.min()
    if validation_ari is not None:
        diagnostics.loc[selected_mask, "tac_ari"] = validation_ari

    profiles.to_csv(tables / "niche_profiles.csv", index=False)
    matrix.to_csv(tables / "perturbation_matrix.csv")
    patterns.to_csv(tables / "patterns.csv")
    annotation_frame().to_csv(tables / "niche_annotations.csv")
    consensus.to_csv(tables / "bootstrap_consensus.csv")
    diagnostics.to_csv(tables / "model_diagnostics.csv", index=False)
    loo.to_csv(tables / "leave_one_tumour_out.csv", index=False)
    sensitivity.to_csv(tables / "feature_sensitivity.csv", index=False)
    effects.to_csv(tables / "abundance_effects.csv", index=False)
    model_summary = {
        "selected_k": cfg["n_clusters"],
        "silhouette": float(partition_silhouette(matrix, consensus_labels)),
        "pca_variance_2d": float(variance[:2].sum()),
        "mean_stability": float(patterns.stability.mean()),
        "stable_niches": int((patterns.stability >= 0.75).sum()),
        "mean_leave_one_tumour_out_ari": float(loo.adjusted_rand_index.mean()),
        "minimum_feature_sensitivity_ari": float(sensitivity.adjusted_rand_index.min()),
        "validation_treatment_ari": None if validation_ari is None else float(validation_ari),
        "median_validation_spearman": None if validation_correlations.empty else float(
            validation_correlations.spearman_r.median()
        ),
        "validation_ari_permutation_p": validation_stats.get("ari_permutation_p"),
        "validation_spearman_permutation_p": validation_stats.get("median_spearman_permutation_p"),
        "reliability": "robust for exploration; not externally validated",
        "model": "batch-matched compositional consensus clustering",
        "consensus_pac": float(((off_diagonal > 0.1) & (off_diagonal < 0.9)).mean()),
        "cluster_permutation_p": float(cluster_test["permutation_p"]),
        "null_silhouette_95th": float(cluster_test["null_silhouette_95th"]),
    }
    (tables / "model_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
    metadata = {
        "source": str(args.h5ad_dir or args.h5ad),
        "n_spots": len(spots),
        "n_samples": int(spots.sample_id.nunique()),
        "n_niches": int(spots.niche.nunique()),
        "stages": stages,
        "treatment": args.treatment,
        "parental_tumours": sorted(spots.parental_tumor.dropna().astype(str).unique()),
        "samples_per_stage": {
            str(k): int(v) for k, v in spots.drop_duplicates("sample_id").stage.value_counts().items()
        },
    }
    (tables / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    histology = None
    if args.h5ad_dir:
        image_dir = processed / "histology"
        histology = export_histology_folder(args.h5ad_dir, image_dir, args.treatment)
        histology = histology[histology.sample_id.isin(spots.sample_id.unique())].copy()
        histology.to_csv(processed / "histology_index.csv", index=False)
        histology["image_path"] = histology.image_file.map(lambda x: image_dir / x)

    make_figures(
        spots, profiles, matrix, patterns, tree, variance, figures, stages,
        histology=histology, diagnostics=diagnostics, effects=effects,
    )
    if histology is not None and len(histology):
        make_readme_figures(
            spots, patterns, histology, diagnostics, model_summary,
            ROOT / "assets" / "readme", tables, cfg["seed"],
        )
        make_explainer_figures(
            matrix, patterns, diagnostics, consensus, loo, sensitivity,
            validation_correlations, model_summary,
            ROOT / "assets" / "readme", cfg["seed"],
        )
    print(f"Analysis complete: {figures / 'figure_main.png'}")


def _select_arm(spots, treatment):
    treated = spots.treatment == treatment
    primary = (spots.stage == "primary") & (spots.treatment == "no_treatment")
    if "parental_tumor" in spots:
        parents = set(spots.loc[treated, "parental_tumor"].dropna().astype(str))
        primary &= spots.parental_tumor.astype(str).isin(parents)
    keep = treated | primary
    selected = spots.loc[keep].copy()
    if selected.empty:
        raise ValueError(f"No sample found for treatment: {treatment}")
    return selected


if __name__ == "__main__":
    main()
