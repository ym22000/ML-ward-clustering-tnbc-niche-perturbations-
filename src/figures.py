from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list

from src.annotations import niche_label, pattern_group, pattern_label
from src.core import pattern_abundance


STAGE_LABELS = {
    "primary": "Primary",
    "mrd7": "MRD - day 7",
    "mrd12": "MRD - day 12",
    "relapsed": "Recurrence",
}


def make_figures(spots, profiles, matrix, patterns, tree, variance, out_dir, stages,
                 histology=None, diagnostics=None, effects=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _style()
    colors = _pattern_colors(patterns)
    heatmap(matrix, patterns, tree, colors, out)
    pca_plot(patterns, variance, colors, out)
    trajectories(profiles, patterns, stages, colors, out)
    if histology is not None and len(histology):
        tissue_sections(spots, patterns, histology, stages, colors, out)
    else:
        spatial_maps(spots, patterns, stages, colors, out)
    stability_plot(patterns, colors, out)
    simple_results(profiles, matrix, patterns, stages, colors, out)
    if diagnostics is not None:
        model_diagnostics_plot(diagnostics, patterns, variance, colors, out)
    if effects is not None:
        effect_estimates(effects, patterns, colors, out)
    main_figure(profiles, matrix, patterns, tree, variance, stages, colors, out)


def heatmap(matrix, patterns, tree, colors, out):
    row_colors = patterns["pattern"].map(colors)
    lim = np.nanpercentile(np.abs(matrix.to_numpy()), 97)
    g = sns.clustermap(
        matrix, row_linkage=tree, col_cluster=False, row_colors=row_colors,
        cmap="vlag", center=0, vmin=-lim, vmax=lim, linewidths=0.25,
        figsize=(12, 6.8), cbar_kws={"label": "Standardised perturbation"},
    )
    g.ax_heatmap.set_xlabel("Stage and feature")
    g.ax_heatmap.set_ylabel("Biological niche")
    g.ax_heatmap.set_yticklabels([
        niche_label(tick.get_text()) for tick in g.ax_heatmap.get_yticklabels()
    ], fontsize=8)
    g.ax_heatmap.set_xticklabels(
        [x.get_text().replace(":", " · ") for x in g.ax_heatmap.get_xticklabels()],
        rotation=55, ha="right", fontsize=7,
    )
    _save(g.fig, out / "perturbation_heatmap")


def pca_plot(patterns, variance, colors, out):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for pattern, frame in patterns.groupby("pattern"):
        ax.scatter(frame.PC1, frame.PC2, s=75, color=colors[pattern], label=pattern_label(pattern),
                   edgecolor="white", linewidth=0.7)
        for niche, row in frame.iterrows():
            ax.annotate(niche_label(niche), (row.PC1, row.PC2), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    ax.set(xlabel=f"PC1 ({variance[0] * 100:.1f} %)",
           ylabel=f"PC2 ({variance[1] * 100:.1f} %)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    sns.despine(ax=ax)
    _save(fig, out / "pca_patterns")


def trajectories(profiles, patterns, stages, colors, out):
    data = pattern_abundance(profiles, patterns)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for pattern, frame in data.groupby("pattern"):
        summary = frame.groupby("stage", observed=True)["abundance"].agg(["mean", "sem"]).reindex(stages)
        x = np.arange(len(stages))
        ax.plot(x, summary["mean"], marker="o", lw=2.2, color=colors[pattern],
                label=pattern_label(pattern))
        ax.fill_between(x, summary["mean"] - summary["sem"], summary["mean"] + summary["sem"],
                        color=colors[pattern], alpha=0.16, linewidth=0)
    ax.set(xticks=np.arange(len(stages)), xticklabels=[STAGE_LABELS.get(x, x) for x in stages],
           ylabel="Mean abundance", xlabel="")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine(ax=ax)
    _save(fig, out / "pattern_trajectories")


def spatial_maps(spots, patterns, stages, colors, out):
    samples = []
    for stage in stages:
        ids = spots.loc[spots.stage == stage, "sample_id"].unique()
        if len(ids):
            samples.append((stage, ids[0]))
    fig, axes = plt.subplots(1, len(samples), figsize=(3.3 * len(samples), 3.5), squeeze=False)
    niche_pattern = patterns["pattern"]
    for ax, (stage, sample) in zip(axes.flat, samples):
        frame = spots[spots.sample_id == sample].copy()
        frame["pattern"] = frame.niche.map(niche_pattern)
        for pattern, part in frame.groupby("pattern"):
            ax.scatter(part.x, part.y, s=8, color=colors[pattern], alpha=0.85, linewidth=0)
        ax.set_title(f"{STAGE_LABELS.get(stage, stage)}\n{sample}", fontsize=10)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor="none", markersize=6, label=pattern_label(pattern))
        for pattern, color in colors.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    _save(fig, out / "spatial_maps")


def tissue_sections(spots, patterns, histology, stages, colors, out):
    selected = []
    for stage in stages:
        rows = histology[histology.stage == stage]
        if len(rows):
            selected.append(rows.iloc[0])
    fig, axes = plt.subplots(1, len(selected), figsize=(4.2 * len(selected), 4.5), squeeze=False)
    niche_pattern = patterns["pattern"]
    for ax, row in zip(axes.flat, selected):
        image = plt.imread(row.image_path)
        frame = spots[spots.sample_id == row.sample_id].copy()
        frame["pattern"] = frame.niche.map(niche_pattern)
        ax.imshow(image)
        for pattern, part in frame.groupby("pattern"):
            ax.scatter(part.x * row.scale_factor, part.y * row.scale_factor,
                       s=5, color=colors[pattern], alpha=0.48, linewidth=0)
        ax.set_title(f"{STAGE_LABELS.get(row.stage, row.stage)}\n{row.sample_id}", fontsize=10)
        ax.set(xlim=(0, image.shape[1]), ylim=(image.shape[0], 0))
        ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor="none", markersize=6, label=pattern_label(pattern))
        for pattern, color in colors.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    _save(fig, out / "tissue_sections")


def stability_plot(patterns, colors, out):
    frame = patterns.sort_values("stability")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh([niche_label(n) for n in frame.index], frame.stability,
            color=frame.pattern.map(colors))
    ax.axvline(0.75, color="0.35", ls="--", lw=1)
    ax.set(xlim=(0, 1.02), xlabel="Bootstrap stability", ylabel="Niche")
    sns.despine(ax=ax)
    _save(fig, out / "pattern_stability")


def main_figure(profiles, matrix, patterns, tree, variance, stages, colors, out):
    fig = plt.figure(figsize=(13.5, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1), height_ratios=(1.2, 1))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    order = leaves_list(tree)
    ordered = matrix.iloc[order]
    lim = np.nanpercentile(np.abs(matrix.to_numpy()), 97)
    im = ax_a.imshow(ordered, aspect="auto", cmap="vlag", vmin=-lim, vmax=lim)
    ax_a.set(yticks=np.arange(len(ordered)),
             yticklabels=[niche_label(n) for n in ordered.index], ylabel="Biological niche")
    ax_a.set(xticks=np.arange(matrix.shape[1]),
             xticklabels=[x.replace(":", "\n") for x in matrix.columns])
    ax_a.tick_params(axis="x", labelsize=6)
    ax_a.tick_params(axis="x", rotation=55)
    fig.colorbar(im, ax=ax_a, shrink=0.62, label="Standardised perturbation")

    for pattern, frame in patterns.groupby("pattern"):
        ax_b.scatter(frame.PC1, frame.PC2, s=68, color=colors[pattern],
                     label=pattern_label(pattern),
                     edgecolor="white", linewidth=0.6)
        for niche, row in frame.iterrows():
            ax_b.annotate(niche_label(niche), (row.PC1, row.PC2), xytext=(3, 3),
                          textcoords="offset points", fontsize=7)
    ax_b.set(xlabel=f"PC1 ({variance[0] * 100:.1f} %)",
             ylabel=f"PC2 ({variance[1] * 100:.1f} %)")
    ax_b.legend(frameon=False, fontsize=7, loc="best")

    data = pattern_abundance(profiles, patterns)
    for pattern, frame in data.groupby("pattern"):
        summary = frame.groupby("stage", observed=True)["abundance"].agg(["mean", "sem"]).reindex(stages)
        x = np.arange(len(stages))
        ax_c.plot(x, summary["mean"], marker="o", lw=2.1, color=colors[pattern],
                  label=pattern_label(pattern))
        ax_c.fill_between(x, summary["mean"] - summary["sem"], summary["mean"] + summary["sem"],
                          color=colors[pattern], alpha=0.15, linewidth=0)
    ax_c.set(xticks=np.arange(len(stages)), xticklabels=[STAGE_LABELS.get(x, x) for x in stages],
             ylabel="Mean abundance")

    frame = patterns.sort_values("stability")
    ax_d.barh([niche_label(n) for n in frame.index], frame.stability,
              color=frame.pattern.map(colors))
    ax_d.axvline(0.75, color="0.35", ls="--", lw=1)
    ax_d.set(xlim=(0, 1.02), xlabel="Bootstrap stability", ylabel="Niche")

    for label, ax in zip("ABCD", [ax_a, ax_b, ax_c, ax_d]):
        ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontsize=14, fontweight="bold")
    sns.despine(ax=ax_b)
    sns.despine(ax=ax_c)
    sns.despine(ax=ax_d)
    _save(fig, out / "figure_main")


def model_diagnostics_plot(diagnostics, patterns, variance, colors, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    axes = axes.flat

    ax = axes[0]
    ax.plot(diagnostics.k, diagnostics.silhouette, marker="o", color="#4C78A8")
    selected = diagnostics[diagnostics.selected].iloc[0]
    ax.scatter(selected.k, selected.silhouette, s=90, color="#E45756", zorder=3)
    ax.set(xticks=diagnostics.k, xlabel="Number of patterns (k)", ylabel="Silhouette score")
    ax.set_title("Pattern separation")

    ax = axes[1]
    ax.plot(diagnostics.k, diagnostics.mean_loo_ari, marker="o", label="Leave-one-tumour-out")
    if "tac_ari" in diagnostics:
        ax.plot(diagnostics.k, diagnostics.tac_ari, marker="s", label="TAC agreement")
    ax.axhline(0.75, color="0.5", ls="--", lw=1)
    ax.set(xticks=diagnostics.k, ylim=(-0.05, 1.05), xlabel="Number of patterns (k)",
           ylabel="Adjusted Rand index", title="Reproducibility")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    values = np.array(variance[:2]) * 100
    ax.bar(["PC1", "PC2"], values, color=["#4C78A8", "#72B7B2"])
    for i, value in enumerate(values):
        ax.text(i, value + 0.8, f"{value:.1f}%", ha="center", fontsize=9)
    ax.set(ylabel="Explained variance (%)", title="PCA summary", ylim=(0, max(values) * 1.25))

    ax = axes[3]
    frame = patterns.sort_values("stability")
    ax.barh([niche_label(n) for n in frame.index], frame.stability,
            color=frame.pattern.map(colors))
    ax.axvline(0.75, color="0.35", ls="--", lw=1)
    ax.set(xlim=(0, 1.02), xlabel="Bootstrap stability", ylabel="Niche",
           title="Assignment robustness")
    for label, ax in zip("ABCD", axes):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
        sns.despine(ax=ax)
    fig.subplots_adjust(hspace=0.48, wspace=0.32)
    _save(fig, out / "model_diagnostics")


def simple_results(profiles, matrix, patterns, stages, colors, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    data = pattern_abundance(profiles, patterns)
    for pattern, frame in data.groupby("pattern"):
        summary = frame.groupby("stage", observed=True)["abundance"].agg(["mean", "sem"]).reindex(stages)
        x = np.arange(len(stages))
        axes[0].plot(x, summary["mean"], marker="o", lw=2, color=colors[pattern],
                     label=pattern_label(pattern))
        axes[0].fill_between(x, summary["mean"] - summary["sem"], summary["mean"] + summary["sem"],
                             color=colors[pattern], alpha=0.15, linewidth=0)
    axes[0].set(xticks=np.arange(len(stages)), xticklabels=[STAGE_LABELS[x] for x in stages],
                ylabel="Mean niche abundance", title="Pattern trajectories")
    axes[0].legend(frameon=False, fontsize=7)

    values = matrix["mrd12:abundance"].sort_values()
    bar_colors = [colors[patterns.loc[niche, "pattern"]] for niche in values.index]
    axes[1].barh([niche_label(n) for n in values.index], values, color=bar_colors)
    axes[1].axvline(0, color="0.3", lw=0.8)
    axes[1].set(xlabel="Standardised change vs primary", ylabel="Niche",
                title="Abundance change in MRD at day 12")
    for label, ax in zip("AB", axes):
        ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
        sns.despine(ax=ax)
    _save(fig, out / "simple_results")


def effect_estimates(effects, patterns, colors, out):
    stages = [stage for stage in ["mrd7", "mrd12", "relapsed"] if stage in effects.stage.unique()]
    fig, axes = plt.subplots(1, len(stages), figsize=(4.1 * len(stages), 5), squeeze=False)
    for ax, stage in zip(axes.flat, stages):
        frame = effects[effects.stage == stage].sort_values("mean_difference")
        y = np.arange(len(frame))
        xerr = np.vstack([
            frame.mean_difference - frame.ci_low,
            frame.ci_high - frame.mean_difference,
        ])
        point_colors = [colors[patterns.loc[niche, "pattern"]] for niche in frame.niche]
        ax.errorbar(frame.mean_difference, y, xerr=xerr, fmt="none", color="0.45", lw=1)
        ax.scatter(frame.mean_difference, y, color=point_colors, s=35, zorder=3)
        ax.axvline(0, color="0.25", lw=0.8)
        ax.set(yticks=y, yticklabels=[niche_label(n) for n in frame.niche],
               xlabel="Abundance difference vs primary",
               title=STAGE_LABELS.get(stage, stage))
        sns.despine(ax=ax)
    _save(fig, out / "abundance_effects")


def _pattern_colors(patterns):
    colors = {}
    for name in patterns["pattern"].unique():
        colors[name] = "#00A6D6" if pattern_group(name) == "mrd" else "#E64B5D"
    return colors


def _style():
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    mpl.rcParams.update({
        "font.family": "Arial",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
    })


def _save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
