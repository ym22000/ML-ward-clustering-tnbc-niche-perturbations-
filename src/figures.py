from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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

STAGE_SHORT = {"mrd7": "D7", "mrd12": "D12", "relapsed": "Rec"}

FEATURE_SHORT = {
    "abundance": "Abun",
    "emt": "EMT",
    "proliferation": "Prolif",
    "hypoxia": "Hyp",
    "immune": "Imm",
    "fibroblast": "Fibro",
    "macrophage": "Macro",
    "tumour": "Tum",
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
    ordered = matrix.iloc[leaves_list(tree)]
    lim = np.nanpercentile(np.abs(matrix.to_numpy()), 97)
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(
        ordered, ax=ax, cmap="vlag", center=0, vmin=-lim, vmax=lim,
        linewidths=0.35, linecolor="white",
        cbar_kws={"label": "Standardised perturbation", "shrink": 0.78},
    )
    ax.set(
        xlabel="Stage and feature",
        ylabel="Biological niche",
        xticklabels=_matrix_labels(matrix.columns),
        yticklabels=[niche_label(niche) for niche in ordered.index],
    )
    ax.tick_params(axis="x", rotation=0, labelsize=8, pad=5)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    _add_stage_separators(ax, matrix.columns)
    for tick, niche in zip(ax.get_yticklabels(), ordered.index):
        tick.set_color(colors[patterns.loc[niche, "pattern"]])
    fig.subplots_adjust(left=0.27, right=0.94, bottom=0.13, top=0.88)
    _save(fig, out / "perturbation_heatmap")


def pca_plot(patterns, variance, colors, out):
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    for pattern, frame in patterns.groupby("pattern"):
        ax.scatter(
            frame.PC1, frame.PC2, s=90, color=colors[pattern],
            label=_short_pattern_label(pattern), edgecolor="white", linewidth=0.8,
        )
        for niche, row in frame.iterrows():
            ax.annotate(niche, (row.PC1, row.PC2), xytext=(5, 4),
                        textcoords="offset points", fontsize=9, fontweight="bold")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    ax.set(
        xlabel=f"PC1 ({variance[0] * 100:.1f} %)",
        ylabel=f"PC2 ({variance[1] * 100:.1f} %)",
        title="Niche perturbation profiles",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, columnspacing=2.2)
    sns.despine(ax=ax)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.25, top=0.90)
    _save(fig, out / "pca_patterns")


def trajectories(profiles, patterns, stages, colors, out):
    data = pattern_abundance(profiles, patterns)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    _plot_trajectories(ax, data, stages, colors)
    ax.set(title="Mean abundance of the two niche patterns",
           ylabel="Mean niche abundance", xlabel="Disease stage")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, columnspacing=2.2)
    sns.despine(ax=ax)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.29, top=0.89)
    _save(fig, out / "pattern_trajectories")


def spatial_maps(spots, patterns, stages, colors, out):
    samples = []
    for stage in stages:
        ids = spots.loc[spots.stage == stage, "sample_id"].unique()
        if len(ids):
            samples.append((stage, ids[0]))
    rows, cols = _panel_shape(len(samples))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.1 * rows), squeeze=False)
    niche_pattern = patterns["pattern"]
    for panel, (ax, (stage, sample)) in enumerate(zip(axes.flat, samples)):
        frame = spots[spots.sample_id == sample].copy()
        frame["pattern"] = frame.niche.map(niche_pattern)
        for pattern, part in frame.groupby("pattern"):
            ax.scatter(part.x, part.y, s=10, color=colors[pattern], alpha=0.85, linewidth=0)
        ax.set_title(f"{chr(65 + panel)}  {STAGE_LABELS.get(stage, stage)}\n{sample}",
                     fontsize=11, loc="left")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
    _hide_unused_axes(axes, len(samples))
    fig.legend(handles=_pattern_handles(colors), loc="lower center", ncol=2,
               frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.10, top=0.94,
                        hspace=0.20, wspace=0.08)
    _save(fig, out / "spatial_maps")


def tissue_sections(spots, patterns, histology, stages, colors, out):
    selected = []
    for stage in stages:
        rows = histology[histology.stage == stage]
        if len(rows):
            selected.append(rows.iloc[0])
    rows, cols = _panel_shape(len(selected))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 4.5 * rows), squeeze=False)
    niche_pattern = patterns["pattern"]
    for panel, (ax, row) in enumerate(zip(axes.flat, selected)):
        image = plt.imread(row.image_path)
        frame = spots[spots.sample_id == row.sample_id].copy()
        frame["pattern"] = frame.niche.map(niche_pattern)
        ax.imshow(image)
        for pattern, part in frame.groupby("pattern"):
            ax.scatter(
                part.x * row.scale_factor, part.y * row.scale_factor,
                s=8, color=colors[pattern], alpha=0.55, linewidth=0,
            )
        ax.set_title(f"{chr(65 + panel)}  {STAGE_LABELS.get(row.stage, row.stage)}\n{row.sample_id}",
                     fontsize=11, loc="left")
        x = frame.x * row.scale_factor
        y = frame.y * row.scale_factor
        pad = 0.04 * max(x.max() - x.min(), y.max() - y.min())
        ax.set(xlim=(x.min() - pad, x.max() + pad),
               ylim=(y.max() + pad, y.min() - pad))
        ax.set_anchor("N")
        ax.axis("off")
    _hide_unused_axes(axes, len(selected))
    fig.legend(handles=_pattern_handles(colors), loc="lower center", ncol=2,
               frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.10, top=0.94,
                        hspace=0.20, wspace=0.05)
    _save(fig, out / "tissue_sections")


def stability_plot(patterns, colors, out):
    frame = patterns.sort_values("stability")
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    bars = ax.barh([niche_label(n) for n in frame.index], frame.stability,
                   color=frame.pattern.map(colors))
    ax.axvline(0.75, color="0.35", ls="--", lw=1, label="0.75 reference")
    ax.set(xlim=(0, 1.08), xlabel="Bootstrap stability", ylabel="Biological niche",
           title="Stability of niche assignments")
    ax.bar_label(bars, labels=[f"{x:.2f}" for x in frame.stability],
                 padding=3, fontsize=8)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    sns.despine(ax=ax)
    fig.subplots_adjust(left=0.31, right=0.96, bottom=0.12, top=0.90)
    _save(fig, out / "pattern_stability")


def main_figure(profiles, matrix, patterns, tree, variance, stages, colors, out):
    fig = plt.figure(figsize=(16, 11.5))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.28, 1), height_ratios=(1.05, 1),
                            hspace=0.42, wspace=0.44)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    ordered = matrix.iloc[leaves_list(tree)]
    lim = np.nanpercentile(np.abs(matrix.to_numpy()), 97)
    im = ax_a.imshow(ordered, aspect="auto", cmap="vlag", vmin=-lim, vmax=lim)
    ax_a.set(
        yticks=np.arange(len(ordered)),
        yticklabels=[niche_label(n) for n in ordered.index],
        xticks=np.arange(matrix.shape[1]),
        xticklabels=_matrix_labels_one_line(matrix.columns),
        ylabel="Biological niche",
        title="A  Perturbation profiles",
    )
    ax_a.tick_params(axis="x", labelsize=7, rotation=90, pad=3)
    ax_a.tick_params(axis="y", labelsize=8)
    _add_stage_separators(ax_a, matrix.columns, headings=False)
    fig.colorbar(im, ax=ax_a, shrink=0.72, pad=0.03, label="Standardised perturbation")

    for pattern, frame in patterns.groupby("pattern"):
        ax_b.scatter(
            frame.PC1, frame.PC2, s=80, color=colors[pattern],
            label=_short_pattern_label(pattern), edgecolor="white", linewidth=0.7,
        )
        for niche, row in frame.iterrows():
            ax_b.annotate(niche, (row.PC1, row.PC2), xytext=(4, 3),
                          textcoords="offset points", fontsize=8, fontweight="bold")
    ax_b.axhline(0, color="0.87", lw=0.8)
    ax_b.axvline(0, color="0.87", lw=0.8)
    ax_b.set(
        xlabel=f"PC1 ({variance[0] * 100:.1f} %)",
        ylabel=f"PC2 ({variance[1] * 100:.1f} %)",
        title="B  PCA overview",
    )
    ax_b.legend(frameon=False, fontsize=8, loc="lower center", ncol=1)

    data = pattern_abundance(profiles, patterns)
    _plot_trajectories(ax_c, data, stages, colors)
    ax_c.set(ylabel="Mean niche abundance", xlabel="Disease stage",
             title="C  Pattern trajectories")
    ax_c.legend(frameon=False, fontsize=8, loc="best")

    frame = patterns.sort_values("stability")
    ax_d.barh([niche_label(n) for n in frame.index], frame.stability,
              color=frame.pattern.map(colors))
    ax_d.axvline(0.75, color="0.35", ls="--", lw=1)
    ax_d.set(xlim=(0, 1.03), xlabel="Bootstrap stability", ylabel="Biological niche",
             title="D  Assignment robustness")

    for ax in (ax_b, ax_c, ax_d):
        sns.despine(ax=ax)
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.08, top=0.95)
    _save(fig, out / "figure_main")


def model_diagnostics_plot(diagnostics, patterns, variance, colors, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4))

    ax = axes[0, 0]
    ax.plot(diagnostics.k, diagnostics.silhouette, marker="o", color="#2878B5")
    selected = diagnostics[diagnostics.selected].iloc[0]
    ax.scatter(selected.k, selected.silhouette, s=95, color="#E64B5D", zorder=3)
    ax.set(xticks=diagnostics.k, xlabel="Number of patterns (k)",
           ylabel="Silhouette score", title="A  Pattern separation")

    ax = axes[0, 1]
    ax.plot(diagnostics.k, diagnostics.mean_loo_ari, marker="o",
            color="#2878B5", label="Leave-one-tumour-out")
    if "tac_ari" in diagnostics:
        ax.plot(diagnostics.k, diagnostics.tac_ari, marker="s",
                color="#F59E0B", label="TAC agreement")
    ax.axhline(0.75, color="0.5", ls="--", lw=1)
    ax.set(xticks=diagnostics.k, ylim=(-0.05, 1.05), xlabel="Number of patterns (k)",
           ylabel="Adjusted Rand index", title="B  Reproducibility")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    values = np.array(variance[:2]) * 100
    bars = ax.bar(["PC1", "PC2"], values, color=["#2878B5", "#22A884"])
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values],
                 padding=4, fontsize=10)
    ax.set(ylabel="Explained variance (%)", title="C  PCA summary",
           ylim=(0, max(values) * 1.25))

    ax = axes[1, 1]
    frame = patterns.sort_values("stability")
    ax.barh(frame.index, frame.stability, color=frame.pattern.map(colors))
    ax.axvline(0.75, color="0.35", ls="--", lw=1)
    ax.set(xlim=(0, 1.03), xlabel="Bootstrap stability", ylabel="Niche ID",
           title="D  Assignment robustness")

    for ax in axes.flat:
        sns.despine(ax=ax)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.09, top=0.94,
                        hspace=0.44, wspace=0.30)
    _save(fig, out / "model_diagnostics")


def simple_results(profiles, matrix, patterns, stages, colors, out):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 10.5),
                             gridspec_kw={"height_ratios": [1, 1.25]})

    data = pattern_abundance(profiles, patterns)
    _plot_trajectories(axes[0], data, stages, colors)
    axes[0].set(ylabel="Mean niche abundance", xlabel="Disease stage",
                title="A  Pattern trajectories")
    axes[0].legend(frameon=False, fontsize=9, loc="upper center",
                   bbox_to_anchor=(0.5, -0.20), ncol=2)

    values = matrix["mrd12:abundance"].sort_values()
    bar_colors = [colors[patterns.loc[niche, "pattern"]] for niche in values.index]
    axes[1].barh([niche_label(n) for n in values.index], values, color=bar_colors)
    axes[1].axvline(0, color="0.3", lw=0.8)
    axes[1].set(xlabel="Standardised change vs primary", ylabel="Biological niche",
                title="B  Abundance change in MRD at day 12")
    for ax in axes:
        sns.despine(ax=ax)
    fig.subplots_adjust(left=0.28, right=0.96, bottom=0.08, top=0.95, hspace=0.56)
    _save(fig, out / "simple_results")


def effect_estimates(effects, patterns, colors, out):
    stages = [stage for stage in ["mrd7", "mrd12", "relapsed"]
              if stage in effects.stage.unique()]
    fig, axes = plt.subplots(len(stages), 1, figsize=(10.5, 4.8 * len(stages)), squeeze=False)
    for panel, (ax, stage) in enumerate(zip(axes.flat, stages)):
        frame = effects[effects.stage == stage].sort_values("mean_difference")
        y = np.arange(len(frame))
        xerr = np.vstack([
            frame.mean_difference - frame.ci_low,
            frame.ci_high - frame.mean_difference,
        ])
        point_colors = [colors[patterns.loc[niche, "pattern"]] for niche in frame.niche]
        ax.errorbar(frame.mean_difference, y, xerr=xerr, fmt="none", color="0.45", lw=1)
        ax.scatter(frame.mean_difference, y, color=point_colors, s=42, zorder=3)
        ax.axvline(0, color="0.25", lw=0.8)
        ax.set(
            yticks=y,
            yticklabels=[niche_label(n) for n in frame.niche],
            xlabel="Abundance difference vs primary",
            title=f"{chr(65 + panel)}  {STAGE_LABELS.get(stage, stage)}",
        )
        sns.despine(ax=ax)
    fig.subplots_adjust(left=0.30, right=0.96, bottom=0.06, top=0.96, hspace=0.48)
    _save(fig, out / "abundance_effects")


def _plot_trajectories(ax, data, stages, colors):
    for pattern, frame in data.groupby("pattern"):
        summary = frame.groupby("stage", observed=True)["abundance"].agg(["mean", "sem"]).reindex(stages)
        x = np.arange(len(stages))
        ax.plot(x, summary["mean"], marker="o", ms=6, lw=2.3, color=colors[pattern],
                label=_short_pattern_label(pattern))
        ax.fill_between(
            x, summary["mean"] - summary["sem"], summary["mean"] + summary["sem"],
            color=colors[pattern], alpha=0.16, linewidth=0,
        )
    ax.set(xticks=np.arange(len(stages)),
           xticklabels=[STAGE_LABELS.get(stage, stage) for stage in stages])


def _matrix_labels(columns):
    labels = []
    for column in columns:
        stage, feature = column.split(":", maxsplit=1)
        labels.append(f"{STAGE_SHORT.get(stage, stage)}\n{FEATURE_SHORT.get(feature, feature)}")
    return labels


def _matrix_labels_one_line(columns):
    labels = []
    for column in columns:
        stage, feature = column.split(":", maxsplit=1)
        labels.append(f"{STAGE_SHORT.get(stage, stage)} {FEATURE_SHORT.get(feature, feature)}")
    return labels


def _add_stage_separators(ax, columns, headings=True):
    stages = [column.split(":", maxsplit=1)[0] for column in columns]
    starts = [0]
    for i in range(1, len(stages)):
        if stages[i] != stages[i - 1]:
            ax.axvline(i, color="white", lw=2.2)
            starts.append(i)
    starts.append(len(stages))
    if headings:
        for start, end in zip(starts[:-1], starts[1:]):
            stage = stages[start]
            x = (start + end) / 2 / len(stages)
            ax.text(x, 1.025, STAGE_LABELS.get(stage, stage), transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")


def _short_pattern_label(pattern):
    label = pattern_label(pattern)
    return label.replace(" remodelling", "\nremodelling").replace(
        " architecture", "\narchitecture"
    )


def _pattern_handles(colors):
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor="none", markersize=7,
               label=_short_pattern_label(pattern).replace("\n", " "))
        for pattern, color in colors.items()
    ]


def _panel_shape(count):
    cols = 2 if count > 1 else 1
    rows = int(np.ceil(count / cols))
    return rows, cols


def _hide_unused_axes(axes, used):
    for ax in axes.flat[used:]:
        ax.axis("off")


def _pattern_colors(patterns):
    colors = {}
    for name in patterns["pattern"].unique():
        colors[name] = "#00A6D6" if pattern_group(name) == "mrd" else "#E64B5D"
    return colors


def _style():
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def _save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
