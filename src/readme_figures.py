"""Large figures used on the repository front page."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib import patheffects
from sklearn.preprocessing import RobustScaler
from umap import UMAP

from src.annotations import niche_label, pattern_group


STAGES = ["primary", "mrd7", "mrd12", "relapsed"]
STAGE_LABELS = {
    "primary": "Primary",
    "mrd7": "MRD · day 7",
    "mrd12": "MRD · day 12",
    "relapsed": "Recurrence",
}
FEATURES = [
    "emt", "proliferation", "hypoxia", "immune",
    "fibroblast", "macrophage", "tumour",
]
TUMOUR_NICHES = ["N0", "N2", "N5", "N8", "N10"]
PATTERN_COLORS = {
    "mrd": "#0072B2",
    "primary": "#D73027",
}
NICHE_COLORS = {
    "N0": "#00A6D6",
    "N1": "#F2C14E",
    "N2": "#F28E2B",
    "N4": "#00A878",
    "N5": "#008B8B",
    "N7": "#7AC943",
    "N8": "#2E86AB",
    "N9": "#8DAA00",
    "N10": "#D7263D",
    "N11": "#8C6D31",
    "N12": "#FF6F61",
}
STAGE_COLORS = {
    "primary": "#333333",
    "mrd7": "#D55E00",
    "mrd12": "#E69F00",
    "relapsed": "#0072B2",
}


def make_readme_figures(spots, patterns, histology, diagnostics, model_summary,
                        out_dir, table_dir, seed=42):
    """Build the small set of figures shown in the README."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _style()

    samples = _complete_series(spots)
    visium_spot_sections(spots, patterns, histology, samples, out)
    tumour_niche_sections(spots, histology, samples, out)
    embedding = tumour_niche_umap(spots, out, seed)
    embedding.to_csv(Path(table_dir) / "tumour_niche_umap.csv", index=False)
    model_specification(diagnostics, model_summary, patterns, out)


def visium_spot_sections(spots, patterns, histology, samples, out):
    """Show the two response groups on four large real Visium sections."""
    by_niche = patterns.pattern.to_dict()
    hist = histology.set_index("sample_id")

    sample_by_stage = dict(samples)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax, stage in zip(axes.flat, STAGES):
        sample = sample_by_stage[stage]
        row = hist.loc[sample]
        image = plt.imread(row.image_path)
        frame = spots[spots.sample_id == sample].copy()
        frame["pattern_group"] = frame.niche.map(lambda n: pattern_group(by_niche[n]))

        ax.imshow(image, alpha=0.76)
        for group in ["primary", "mrd"]:
            part = frame[frame.pattern_group == group]
            ax.scatter(
                part.x * row.scale_factor,
                part.y * row.scale_factor,
                s=23,
                color=PATTERN_COLORS[group],
                alpha=0.88,
                edgecolor="white",
                linewidth=0.38,
                rasterized=True,
            )
        ax.set_title(STAGE_LABELS[stage], fontsize=22, fontweight="bold", pad=12)
        _set_spatial_limits(ax, frame, row, image)
        ax.axis("off")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor="none", markersize=12, label=label)
        for label, color in [
            ("MRD immune–stromal remodelling", PATTERN_COLORS["mrd"]),
            ("Primary/recurrent tissue architecture", PATTERN_COLORS["primary"]),
        ]
    ]
    fig.suptitle("Visium spots across treatment stages", x=0.04,
                 ha="left", fontsize=29, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
               ncol=2, frameon=False, fontsize=16, markerscale=1.5,
               handletextpad=0.6, columnspacing=2.0)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.105,
                        wspace=0.025, hspace=0.10)
    _save(fig, out / "visium_spot_sections")


def tumour_niche_sections(spots, histology, samples, out):
    """Project the five paper-defined tumour niches back onto tissue."""
    hist = histology.set_index("sample_id")
    sample_by_stage = dict(samples)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for panel, (ax, stage) in enumerate(zip(axes.flat, STAGES)):
        sample = sample_by_stage[stage]
        row = hist.loc[sample]
        image = plt.imread(row.image_path)
        frame = spots[spots.sample_id == sample]
        selected = frame[frame.niche.isin(TUMOUR_NICHES)]

        ax.imshow(image, alpha=0.82)
        ax.scatter(
            frame.x * row.scale_factor,
            frame.y * row.scale_factor,
            s=8,
            color="0.82",
            alpha=0.22,
            linewidth=0,
            rasterized=True,
        )
        for niche in TUMOUR_NICHES:
            part = selected[selected.niche == niche]
            ax.scatter(
                part.x * row.scale_factor,
                part.y * row.scale_factor,
                s=14,
                color=NICHE_COLORS[niche],
                alpha=0.78,
                edgecolor="white",
                linewidth=0.3,
                rasterized=True,
            )
        ax.set_title(STAGE_LABELS[stage], fontsize=19, fontweight="bold", pad=9)
        ax.text(0.01, 0.985, "ABCD"[panel], transform=ax.transAxes,
                ha="left", va="top", fontsize=21, fontweight="bold")
        _set_spatial_limits(ax, frame, row, image)
        ax.axis("off")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NICHE_COLORS[n],
               markeredgecolor="none", markersize=11, label=niche_label(n))
        for n in TUMOUR_NICHES
    ]
    fig.suptitle("Projection of the five tumour-associated niches", x=0.04,
                 ha="left", fontsize=26, fontweight="bold")
    fig.legend(handles=handles, title="Published tumour niches", loc="lower center",
               bbox_to_anchor=(0.5, 0.052), ncol=2, frameon=False,
               fontsize=17, title_fontsize=18,
               markerscale=1.5, handletextpad=0.55, columnspacing=2.2)
    fig.text(
        0.99, 0.012,
        "Same complete matched slide series as above; grey points are other published niches.",
        ha="right", fontsize=11, color="0.35",
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.89, bottom=0.265,
                        wspace=0.06, hspace=0.12)
    _save(fig, out / "tumour_niche_spatial")


def tumour_niche_umap(spots, out, seed):
    """Single descriptive UMAP of all retained niches using molecular features."""
    niches = [n for n in NICHE_COLORS if n in set(spots.niche)]
    frame = spots[spots.niche.isin(niches)].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

    sampled = []
    for _, group in frame.groupby(["niche", "stage"], observed=True):
        sampled.append(group.sample(min(len(group), 450), random_state=seed))
    frame = pd.concat(sampled, ignore_index=True)

    values = RobustScaler(quantile_range=(10, 90)).fit_transform(frame[FEATURES])
    values = np.clip(values, -8, 8)
    coords = UMAP(
        n_neighbors=35,
        min_dist=0.22,
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
    ).fit_transform(values)
    frame["UMAP1"], frame["UMAP2"] = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(16, 10))
    for niche in niches:
        part = frame[frame.niche == niche]
        ax.scatter(part.UMAP1, part.UMAP2, s=11, color=NICHE_COLORS[niche],
                   alpha=0.62, linewidth=0, rasterized=True, label=niche_label(niche))
        centre = part[["UMAP1", "UMAP2"]].median()
        label = ax.text(centre.UMAP1, centre.UMAP2, niche, fontsize=15,
                        fontweight="bold", ha="center", va="center", color="white")
        label.set_path_effects([patheffects.withStroke(linewidth=4.0,
                                                       foreground=NICHE_COLORS[niche])])

    ax.set_xlabel("UMAP 1", fontsize=17)
    ax.set_ylabel("UMAP 2", fontsize=17)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(title="Biological niche", frameon=False, fontsize=15,
              title_fontsize=17, markerscale=2.7, ncol=1,
              loc="center left", bbox_to_anchor=(1.01, 0.50))
    sns.despine(ax=ax, left=True, bottom=True)

    fig.suptitle("All 11 biological niches in molecular-feature space", x=0.055,
                 ha="left", fontsize=26, fontweight="bold")
    fig.text(
        0.99, 0.012,
        f"n = {len(frame):,} balanced spots · 7 molecular features · niche weights excluded · descriptive projection",
        ha="right", fontsize=11, color="0.35",
    )
    fig.subplots_adjust(left=0.07, right=0.60, top=0.89, bottom=0.09)
    _save(fig, out / "tumour_niche_umap")
    return frame[["spot_id", "sample_id", "stage", "niche", "UMAP1", "UMAP2"]]


def model_specification(diagnostics, summary, patterns, out):
    """Show model definition, selection, embedding and robustness."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax0, ax1, ax2, ax3 = axes.flat

    ax0.axis("off")
    steps = [
        ("27,931", "spatial spots"),
        ("143", "tumour × niche profiles"),
        ("11 × 24", "niche trajectories"),
        ("2", "response patterns"),
    ]
    ys = np.linspace(0.86, 0.14, len(steps))
    for i, ((value, label), y) in enumerate(zip(steps, ys)):
        color = ["#6F4EBC", "#00A878", "#F28E2B", PATTERN_COLORS["mrd"]][i]
        ax0.text(0.17, y, value, ha="center", va="center", fontsize=15,
                 fontweight="bold", color="white",
                 bbox={"boxstyle": "round,pad=0.55", "fc": color,
                       "ec": "white", "lw": 1.2})
        ax0.text(0.37, y, label, ha="left", va="center", fontsize=15,
                 fontweight="bold" if i == 3 else "normal")
        if i < len(steps) - 1:
            ax0.annotate("", xy=(0.17, ys[i + 1] + 0.070),
                         xytext=(0.17, y - 0.070),
                         arrowprops={"arrowstyle": "-|>", "color": "0.45", "lw": 2})
    ax0.set(xlim=(0, 1), ylim=(0, 1))
    ax0.set_title("Analysis unit and reduction", loc="left", fontsize=17,
                  fontweight="bold", pad=8)

    ax1.plot(diagnostics.k, diagnostics.silhouette, color="0.45", marker="o", lw=2.1,
             markersize=7)
    selected = diagnostics[diagnostics.selected].iloc[0]
    ax1.scatter([selected.k], [selected.silhouette], s=150, color=PATTERN_COLORS["mrd"],
                edgecolor="white", linewidth=1.2, zorder=3)
    ax1.axhline(summary["null_silhouette_95th"], color="0.55", ls="--", lw=1)
    ax1.annotate(f"selected k = {int(selected.k)}\nSilhouette = {selected.silhouette:.3f}",
                 (selected.k, selected.silhouette), xytext=(12, -34), textcoords="offset points",
                 fontsize=12, color=PATTERN_COLORS["mrd"], fontweight="bold")
    ax1.set(xticks=diagnostics.k, xlabel="Number of patterns (k)", ylabel="Silhouette score")
    ax1.tick_params(labelsize=12)
    ax1.xaxis.label.set_size(14)
    ax1.yaxis.label.set_size(14)
    ax1.set_title("Pattern-number selection", loc="left", fontsize=17,
                  fontweight="bold", pad=8)
    ax1.text(5.95, summary["null_silhouette_95th"] + 0.004,
             "95th percentile under permutation", ha="right", va="bottom",
             fontsize=11, color="0.35")
    sns.despine(ax=ax1)

    for niche, row in patterns.iterrows():
        group = pattern_group(row.pattern)
        marker = "o" if group == "mrd" else "s"
        ax2.scatter(row.PC1, row.PC2, s=145, marker=marker,
                    color=PATTERN_COLORS[group], edgecolor="white",
                    linewidth=0.8, zorder=3)
        ax2.annotate(niche, (row.PC1, row.PC2), xytext=(4, 4),
                     textcoords="offset points", fontsize=12,
                     fontweight="bold", color=PATTERN_COLORS[group],
                     zorder=4)
    ax2.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax2.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax2.set(xlabel="PC1", ylabel="PC2")
    ax2.tick_params(labelsize=12)
    ax2.xaxis.label.set_size(14)
    ax2.yaxis.label.set_size(14)
    ax2.set_title("Niche response trajectories", loc="left", fontsize=17,
                  fontweight="bold", pad=8)
    ax2.legend(handles=[
               Line2D([0], [0], marker="o", color="none",
               markerfacecolor=PATTERN_COLORS["mrd"], markeredgecolor="none",
               label="MRD immune–stromal"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=PATTERN_COLORS["primary"], markeredgecolor="none",
               label="Primary/recurrent architecture"),
    ], frameon=False, fontsize=13, loc="best", markerscale=1.4)
    sns.despine(ax=ax2)

    metrics = pd.Series({
        "Bootstrap": summary["mean_stability"],
        "Tumour-out": summary["mean_leave_one_tumour_out_ari"],
        "Feature-out": summary["minimum_feature_sensitivity_ari"],
        "TAC arm": summary["validation_treatment_ari"],
    })
    colors = ["#6F4EBC", "#00A878", "#F28E2B", PATTERN_COLORS["mrd"]]
    ax3.barh(metrics.index, metrics.values, color=colors, height=0.55)
    for y_pos, value in enumerate(metrics.values):
        ax3.text(value - 0.02, y_pos, f"{value:.2f}", ha="right", va="center",
                 color="white", fontsize=12, fontweight="bold")
    ax3.axvline(0.75, color="0.4", ls="--", lw=1)
    ax3.set(xlim=(0, 1.04), xlabel="ARI or consensus stability")
    ax3.tick_params(labelsize=13)
    ax3.xaxis.label.set_size(14)
    ax3.set_title("Robustness checks", loc="left", fontsize=17,
                  fontweight="bold", pad=8)
    sns.despine(ax=ax3)

    for label, ax in zip("ABCD", axes.flat):
        ax.text(-0.10, 1.07, label, transform=ax.transAxes, fontsize=21,
                fontweight="bold")
    fig.suptitle("Unsupervised model and robustness checks", x=0.04, ha="left",
                 fontsize=26, fontweight="bold")
    fig.text(0.99, 0.01,
             f"Permutation p = {summary['cluster_permutation_p']:.4f} · PCA variance = {summary['pca_variance_2d']:.1%} · "
             "robustness metrics are not predictive accuracy; TAC is from the same study.",
             ha="right", fontsize=11, color="0.35")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.10,
                        wspace=0.32, hspace=0.36)
    _save(fig, out / "ml_model_summary")


def _complete_series(spots):
    """Select the first slide containing all four stages; this avoids cherry-picking."""
    meta = spots.drop_duplicates("sample_id")[["sample_id", "stage", "slide"]]
    complete = []
    for slide, frame in meta.groupby("slide", observed=True):
        if set(STAGES).issubset(frame.stage):
            complete.append((str(slide), frame))
    if complete:
        frame = sorted(complete, key=lambda x: x[0])[0][1]
    else:
        frame = meta
    return [
        (stage, sorted(frame.loc[frame.stage == stage, "sample_id"])[0])
        for stage in STAGES if (frame.stage == stage).any()
    ]


def _set_spatial_limits(ax, frame, histology_row, image, padding=0.04):
    """Zoom each panel to the measured tissue area while retaining a small margin."""
    x = frame.x.to_numpy() * histology_row.scale_factor
    y = frame.y.to_numpy() * histology_row.scale_factor
    x_pad = max(np.ptp(x), 1) * padding
    y_pad = max(np.ptp(y), 1) * padding
    ax.set_xlim(max(0, x.min() - x_pad), min(image.shape[1], x.max() + x_pad))
    ax.set_ylim(min(image.shape[0], y.max() + y_pad), max(0, y.min() - y_pad))


def _style():
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    mpl.rcParams.update({
        "font.family": "Arial",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": None,
    })


def _save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
