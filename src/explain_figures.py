"""Large explanatory figures for the repository README."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.annotations import niche_label, pattern_group


GROUP_COLORS = {"mrd": "#00A6D6", "primary": "#E64B5D"}
CLUSTER_COLORS = ["#00A6D6", "#E64B5D", "#6F4EBC", "#F28E2B", "#00A878", "#E83E8C"]
NICHE_COLORS = {
    "N0": "#00A6D6", "N1": "#F2C14E", "N2": "#F28E2B", "N4": "#00A878",
    "N5": "#008B8B", "N7": "#7AC943", "N8": "#2E86AB", "N9": "#8DAA00",
    "N10": "#D7263D", "N11": "#8C6D31", "N12": "#FF6F61",
}
METRIC_SHORT = {
    "abundance": "Abun", "emt": "EMT", "proliferation": "Prolif",
    "hypoxia": "Hyp", "immune": "Imm", "fibroblast": "Fibro",
    "macrophage": "Macro", "tumour": "Tum",
}
STAGE_SHORT = {"mrd7": "D7", "mrd12": "D12", "relapsed": "Rec"}


def make_explainer_figures(matrix, patterns, diagnostics, consensus, loo,
                           sensitivity, validation_correlations, summary,
                           out_dir, seed=42):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _style()
    clustering_k_comparison(matrix, patterns, diagnostics, out)
    clustering_k2_detail(matrix, patterns, diagnostics, out)
    model_choice(matrix, out)
    reliability_metrics(
        matrix, patterns, diagnostics, consensus, loo, sensitivity,
        validation_correlations, summary, out, seed,
    )


def clustering_k_comparison(matrix, patterns, diagnostics, out):
    """Show every tested tree cut on one fixed PCA projection."""
    x = StandardScaler().fit_transform(matrix)
    tree = linkage(x, method="ward", optimal_ordering=True)
    coords = patterns[["PC1", "PC2"]].reindex(matrix.index)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)

    for ax, k in zip(axes.flat[:5], range(2, 7)):
        labels = _ordered_labels(fcluster(tree, k, criterion="maxclust"), coords.PC1)
        row = diagnostics.loc[diagnostics.k == k].iloc[0]
        for label in sorted(np.unique(labels)):
            keep = labels == label
            ax.scatter(
                coords.PC1[keep], coords.PC2[keep], s=170,
                color=CLUSTER_COLORS[label - 1], edgecolor="white", linewidth=1.4,
            )
        for niche, point in coords.iterrows():
            ax.text(point.PC1 + 0.10, point.PC2 + 0.08, niche,
                    fontsize=13, fontweight="bold")
        ax.axhline(0, color="0.88", lw=1, zorder=0)
        ax.axvline(0, color="0.88", lw=1, zorder=0)
        ax.set_title(
            f"k = {k}   |   silhouette = {row.silhouette:.3f}\n"
            f"smallest cluster = {int(row.smallest_cluster)} niches",
            fontsize=17, fontweight="bold", pad=10,
            color="#00A6D6" if k == 2 else "0.16",
        )
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel("PC2", fontsize=14)
        sns.despine(ax=ax)

    ax = axes.flat[5]
    ax2 = ax.twinx()
    ax.plot(diagnostics.k, diagnostics.silhouette, "-o", color="#00A6D6",
            lw=3, ms=10, label="Silhouette")
    ax2.plot(diagnostics.k, diagnostics.smallest_cluster, "-s", color="#E64B5D",
             lw=3, ms=9, label="Smallest cluster")
    ax.axvspan(1.85, 2.15, color="#00A6D6", alpha=0.12)
    ax.set(xlabel="Number of clusters (k)", ylabel="Silhouette score")
    ax2.set_ylabel("Smallest cluster size")
    ax.set_xticks(diagnostics.k)
    ax.set_title("k = 2 balances separation and cluster size",
                 fontsize=17, fontweight="bold", pad=10)
    handles = [
        Line2D([0], [0], color="#00A6D6", marker="o", lw=3, label="Silhouette"),
        Line2D([0], [0], color="#E64B5D", marker="s", lw=3, label="Smallest cluster"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=13)
    sns.despine(ax=ax)
    sns.despine(ax=ax2, left=True, right=False)

    fig.suptitle("Ward groupings of the same 11 niche trajectories",
                 x=0.04, ha="left", fontsize=29, fontweight="bold")
    fig.text(
        0.99, 0.012,
        "PCA is used only for display · Ward clustering uses all 24 standardised contrasts · k = 2 gives the clearest balanced grouping",
        ha="right", fontsize=13, color="0.35",
    )
    fig.subplots_adjust(left=0.07, right=0.94, top=0.88, bottom=0.07,
                        wspace=0.28, hspace=0.30)
    _save(fig, out / "clustering_k_comparison")


def clustering_k2_detail(matrix, patterns, diagnostics, out):
    """Show the selected tree and its full input matrix."""
    x = StandardScaler().fit_transform(matrix)
    tree = linkage(x, method="ward", optimal_ordering=True)
    # A left-facing dendrogram draws its first leaf at the bottom. Reverse the
    # order so the heatmap rows align with the tree from top to bottom.
    order = leaves_list(tree)[::-1]
    ordered_niches = matrix.index[order]
    ordered_matrix = pd.DataFrame(x, index=matrix.index, columns=matrix.columns).loc[ordered_niches]
    group = patterns.pattern.map(pattern_group)
    fig = plt.figure(figsize=(21, 9.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 2.7], wspace=0.52)

    ax0 = fig.add_subplot(gs[0, 0])
    dendrogram(
        tree, orientation="left", no_labels=True,
        color_threshold=0, above_threshold_color="0.40", ax=ax0,
    )
    ax0.set_title("A  Ward dendrogram", loc="left", fontsize=18, fontweight="bold")
    ax0.set_xlabel("Increase in within-cluster variance", fontsize=13)
    sns.despine(ax=ax0, left=True)

    ax2 = fig.add_subplot(gs[0, 1])
    labels = [_column_label(c) for c in ordered_matrix.columns]
    vivid_diverging = mpl.colors.LinearSegmentedColormap.from_list(
        "vivid_diverging",
        ["#08306B", "#2171B5", "#6BAED6", "#F7FBFF",
         "#FEE391", "#F46D43", "#B30000"],
    )
    sns.heatmap(
        ordered_matrix, cmap=vivid_diverging, center=0, vmin=-1.6, vmax=1.6,
        xticklabels=labels, yticklabels=[niche_label(n) for n in ordered_niches],
        cbar_kws={"label": "Column-standardised perturbation", "shrink": 0.72}, ax=ax2,
    )
    for tick, niche in zip(ax2.get_yticklabels(), ordered_niches):
        tick.set_color(GROUP_COLORS[group[niche]])
        tick.set_fontweight("bold")
    for boundary in (8, 16):
        ax2.axvline(boundary, color="white", lw=3)
    ax2.tick_params(axis="x", labelrotation=65, labelsize=10)
    ax2.tick_params(axis="y", labelrotation=0, labelsize=11)
    ax2.set_title("B  The complete 11 × 24 model input", loc="left",
                  fontsize=18, fontweight="bold")
    ax2.set_xlabel("Stage × measurement contrast versus matched primary")
    ax2.set_ylabel("")

    selected = diagnostics.loc[diagnostics.selected].iloc[0]
    fig.suptitle("Selected Ward solution: two reproducible biological macro-patterns",
                 x=0.035, ha="left", fontsize=29, fontweight="bold")
    fig.text(
        0.99, 0.015,
        f"Silhouette = {selected.silhouette:.3f} · cluster sizes = 6 and 5 · clustering is performed before biological naming",
        ha="right", fontsize=13, color="0.35",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.19)
    _save(fig, out / "clustering_k2_detail")


def model_choice(matrix, out):
    """Show how Ward clustering matches this small unsupervised problem."""
    n_objects = len(matrix)
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [1.0, 1.55]},
    )

    ax0.axis("off")
    y = [0.88, 0.65, 0.42, 0.19]
    boxes = [
        ("No known response labels", "Supervised learning cannot be trained"),
        (f"{n_objects} niches, {matrix.shape[1]} contrasts", "Too few objects for a flexible probabilistic or deep model"),
        ("Need groups + a readable tree", "Distances and every merge remain visible"),
        ("Ward hierarchical clustering", "Simple, deterministic and interpretable"),
    ]
    for i, ((title, body), ypos) in enumerate(zip(boxes, y)):
        color = "#00A6D6" if i == 3 else "#ECE8F5"
        text_color = "white" if i == 3 else "#282334"
        ax0.text(
            0.50, ypos, f"{title}\n{body}", ha="center", va="center",
            fontsize=15, fontweight="bold" if i in (0, 3) else "normal",
            color=text_color, transform=ax0.transAxes,
            bbox={"boxstyle": "round,pad=0.65", "facecolor": color,
                  "edgecolor": "#564379", "linewidth": 1.4},
        )
        if i < 3:
            ax0.annotate("", xy=(0.5, ypos - 0.13), xytext=(0.5, ypos - 0.05),
                         xycoords=ax0.transAxes,
                         arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#35518A"})
    ax0.set_title("A  The data define the model requirements", loc="left",
                  fontsize=19, fontweight="bold")

    ax1.axis("off")
    comparisons = [
        ("WARD", "CHOSEN", f"Works with n = {n_objects} · deterministic\nShows every merge in a dendrogram", "#00A6D6", "white"),
        ("k-means", "possible check", "Needs k in advance · random starts\nNo hierarchy to audit", "#E8F6FA", "#222222"),
        ("Gaussian mixture", "not retained", f"Estimates means and covariances\nToo many parameters for {n_objects} objects", "#F1ECF6", "#222222"),
        ("DBSCAN / HDBSCAN", "not retained", f"Searches density and outliers\nDensity is unstable at n = {n_objects}", "#F1ECF6", "#222222"),
        ("Deep network", "not appropriate", f"Needs many independent examples\nWould mostly memorise {n_objects} niches", "#F1ECF6", "#222222"),
    ]
    y_positions = np.linspace(0.82, 0.18, len(comparisons))
    for (method, status, reason, face, text_color), ypos in zip(comparisons, y_positions):
        ax1.add_patch(Rectangle(
            (0.02, ypos - 0.065), 0.96, 0.125, transform=ax1.transAxes,
            facecolor=face, edgecolor="#564379" if method == "WARD" else "white",
            linewidth=1.5 if method == "WARD" else 0.8,
        ))
        ax1.text(0.055, ypos + 0.018, method, transform=ax1.transAxes,
                 fontsize=15, fontweight="bold", color=text_color, va="center")
        ax1.text(0.055, ypos - 0.025, status.upper(), transform=ax1.transAxes,
                 fontsize=10.5, fontweight="bold", color=text_color, va="center")
        ax1.text(0.40, ypos, reason, transform=ax1.transAxes,
                 fontsize=12.5, color=text_color, va="center", linespacing=1.25)
    ax1.set_title("B  Other methods fit these data less well", loc="left",
                  fontsize=19, fontweight="bold", pad=14)

    fig.suptitle("Hierarchical Ward clustering matches this dataset",
                 x=0.035, ha="left", fontsize=29, fontweight="bold")
    fig.text(
        0.99, 0.015,
        "This analysis does not predict a clinical outcome.",
        ha="right", fontsize=13, color="0.35",
    )
    fig.subplots_adjust(left=0.045, right=0.975, top=0.87, bottom=0.13, wspace=0.24)
    _save(fig, out / "model_choice_explainer")


def reliability_metrics(matrix, patterns, diagnostics, consensus, loo, sensitivity,
                        validation_correlations, summary, out, seed):
    """Explain each robustness metric with the observed project values."""
    x = StandardScaler().fit_transform(matrix)
    labels = patterns.cluster.reindex(matrix.index).to_numpy()
    silhouette = pd.Series(silhouette_samples(x, labels), index=matrix.index).sort_values()
    null = _permutation_silhouettes(x, len(np.unique(labels)), 5000, seed)
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.flat

    ax0.plot(diagnostics.k, diagnostics.silhouette, "-o", color="#00A6D6", lw=3, ms=10)
    ax0.plot(diagnostics.k, diagnostics.smallest_cluster / 10, "--s",
             color="#E64B5D", lw=2.5, ms=8)
    ax0.axvspan(1.85, 2.15, color="#00A6D6", alpha=0.13)
    ax0.set_xticks(diagnostics.k)
    ax0.set(xlabel="Number of clusters (k)", ylabel="Score")
    ax0.set_title("A  k selection", loc="left", fontsize=18, fontweight="bold")
    ax0.text(2.15, 0.43, "highest silhouette\n5 niches in smallest group",
             fontsize=12, color="#006E90", va="top")
    sns.despine(ax=ax0)

    colors = [GROUP_COLORS[pattern_group(patterns.loc[n, "pattern"])] for n in silhouette.index]
    ax1.barh(silhouette.index, silhouette.values, color=colors)
    ax1.axvline(summary["silhouette"], color="0.20", ls="--", lw=2,
                label=f"mean = {summary['silhouette']:.3f}")
    ax1.axvline(0, color="0.50", lw=1)
    ax1.set(xlabel="Silhouette per niche", ylabel="")
    ax1.set_title("B  Separation of each niche", loc="left",
                  fontsize=18, fontweight="bold")
    ax1.legend(frameon=False, fontsize=12, loc="lower right")
    sns.despine(ax=ax1)

    ordered = patterns.sort_values(["cluster", "PC1"]).index
    sns.heatmap(
        consensus.loc[ordered, ordered], cmap="mako", vmin=0, vmax=1, square=True,
        xticklabels=ordered, yticklabels=ordered,
        cbar_kws={"label": "Co-clustering frequency", "shrink": 0.75}, ax=ax2,
    )
    ax2.set_title("C  1,000-refit consensus", loc="left",
                  fontsize=18, fontweight="bold")
    ax2.tick_params(axis="x", rotation=0)
    ax2.tick_params(axis="y", rotation=0)

    metrics = pd.Series({
        "Bootstrap\nmean": summary["mean_stability"],
        "Tumour-out\nmean ARI": loo.adjusted_rand_index.mean(),
        "Feature-out\nminimum ARI": sensitivity.adjusted_rand_index.min(),
        "TAC arm\nARI": summary["validation_treatment_ari"],
    })
    bars = ax3.barh(metrics.index, metrics.values,
                    color=["#6F4EBC", "#00A878", "#F28E2B", "#00A6D6"])
    ax3.axvline(0.75, color="0.35", ls="--", lw=2, label="predefined stability threshold")
    ax3.set_xlim(0, 1.06)
    ax3.set_xlabel("ARI or consensus stability")
    ax3.set_title("D  Robustness to reasonable perturbations", loc="left",
                  fontsize=18, fontweight="bold")
    for bar, value in zip(bars, metrics.values):
        ax3.text(value - 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                 ha="right", va="center", color="white", fontsize=13, fontweight="bold")
    ax3.legend(frameon=False, fontsize=11, loc="lower left")
    sns.despine(ax=ax3)

    validation = validation_correlations.sort_values("spearman_r")
    ax4.barh(validation.niche, validation.spearman_r,
             color=[NICHE_COLORS[n] for n in validation.niche])
    median = validation.spearman_r.median()
    ax4.axvline(median, color="0.20", ls="--", lw=2,
                label=f"median = {median:.3f}")
    ax4.set_xlim(0, 1)
    ax4.set_xlabel("Same-niche cisplatin–TAC Spearman correlation")
    ax4.set_title("E  Related treatment-arm reproducibility", loc="left",
                  fontsize=18, fontweight="bold")
    ax4.legend(frameon=False, fontsize=12, loc="lower right")
    sns.despine(ax=ax4)

    ax5.hist(null, bins=34, color="#B7B2C8", edgecolor="white")
    ax5.axvline(summary["null_silhouette_95th"], color="#F28E2B", ls="--", lw=2.5,
                label=f"null 95% = {summary['null_silhouette_95th']:.3f}")
    ax5.axvline(summary["silhouette"], color="#00A6D6", lw=3,
                label=f"observed = {summary['silhouette']:.3f}")
    ax5.set(xlabel="Best k = 2 silhouette after feature shuffling", ylabel="Permutations")
    ax5.set_title("F  Permutation test", loc="left", fontsize=18, fontweight="bold")
    ax5.legend(frameon=False, fontsize=12, loc="upper left")
    ax5.text(0.98, 0.90, "p = 0.0002", transform=ax5.transAxes,
             ha="right", fontsize=15, fontweight="bold", color="#006E90")
    sns.despine(ax=ax5)

    fig.suptitle("Reliability metrics quantify distinct failure modes",
                 x=0.035, ha="left", fontsize=29, fontweight="bold")
    fig.text(
        0.99, 0.014,
        "Stability and ARI show whether groups change · Spearman compares response order · permutation compares the result with chance · none is predictive accuracy",
        ha="right", fontsize=13, color="0.35",
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.89, bottom=0.08,
                        wspace=0.34, hspace=0.33)
    _save(fig, out / "reliability_metrics_explained")


def _ordered_labels(labels, x_coordinate):
    means = pd.Series(x_coordinate.to_numpy()).groupby(labels).mean().sort_values()
    mapping = {old: new for new, old in enumerate(means.index, start=1)}
    return np.array([mapping[label] for label in labels])


def _column_label(column):
    stage, metric = column.split(":", 1)
    return f"{STAGE_SHORT[stage]}\n{METRIC_SHORT[metric]}"


def _permutation_silhouettes(x, k, n_permutations, seed):
    rng = np.random.default_rng(seed)
    values = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])])
        trial = linkage(shuffled, method="ward")
        labels = fcluster(trial, k, criterion="maxclust")
        values[i] = silhouette_score(shuffled, labels)
    return values


def _style():
    sns.set_theme(style="ticks", context="paper", font_scale=1.10)
    mpl.rcParams.update({
        "font.family": "Arial", "axes.linewidth": 0.9,
        "axes.titlesize": 18, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "savefig.bbox": None,
    })


def _save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
