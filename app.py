import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.annotations import niche_label, pattern_group, pattern_label


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(page_title="Tumour niche perturbation", layout="wide")
st.title("Biological programmes of breast-cancer niche remodelling")
st.caption("An unsupervised re-analysis of published spatial tumour data with biological annotation")

required = [
    TABLES / "patterns.csv", TABLES / "niche_profiles.csv",
    TABLES / "model_summary.json", PROCESSED / "spots.csv",
    PROCESSED / "histology_index.csv",
]
if not all(path.exists() for path in required):
    st.error("Run the real-data analysis with run_analysis.py first.")
    st.stop()

patterns = pd.read_csv(required[0], index_col=0)
profiles = pd.read_csv(required[1])
spots = pd.read_csv(required[3])
histology = pd.read_csv(required[4])
meta = json.loads((TABLES / "run_metadata.json").read_text(encoding="utf-8"))
model = json.loads((TABLES / "model_summary.json").read_text(encoding="utf-8"))
matrix = pd.read_csv(TABLES / "perturbation_matrix.csv", index_col=0)
effects = pd.read_csv(TABLES / "abundance_effects.csv")

stage_order = [x for x in ["primary", "mrd7", "mrd12", "relapsed"] if x in spots.stage.unique()]
stage_labels = {
    "primary": "Primary", "mrd7": "MRD - day 7",
    "mrd12": "MRD - day 12", "relapsed": "Recurrence",
}
pattern_names = sorted(patterns.pattern.unique())
pattern_colors = {
    name: "#00A6D6" if pattern_group(name) == "mrd" else "#E64B5D"
    for name in pattern_names
}

st.sidebar.header("Selection")
niche = st.sidebar.selectbox("Biological niche", patterns.index, format_func=niche_label)
stage = st.sidebar.selectbox("Disease stage", stage_order, format_func=lambda x: stage_labels[x])
samples = histology.loc[histology.stage == stage, "sample_id"].tolist()
sample = st.sidebar.selectbox("Tissue section", samples)

overview, sections, biology, diagnostics = st.tabs([
    "Overview", "Tissue sections", "Biological results", "Model results",
])

with overview:
    st.image(str(ROOT / "results" / "figures" / "figure_main.png"), width="stretch")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tumours", meta["n_samples"])
    c2.metric("Visium spots", f"{meta['n_spots']:,}")
    c3.metric("Detected patterns", model["selected_k"])
    c4.metric("Stable niches", f"{model['stable_niches']} / {meta['n_niches']}")
    overview_table = patterns[[
        "biological_name", "compartment_class", "pattern", "published_response", "stability"
    ]].copy()
    overview_table.index = [niche_label(n) for n in overview_table.index]
    overview_table["pattern"] = overview_table.pattern.map(pattern_label)
    st.dataframe(overview_table.sort_values("pattern"), width="stretch")

with sections:
    row = histology.set_index("sample_id").loc[sample]
    image = plt.imread(PROCESSED / "histology" / row.image_file)
    frame = spots[spots.sample_id == sample].copy()
    frame["pattern"] = frame.niche.map(patterns.pattern)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(image)
    for pattern, part in frame.groupby("pattern"):
        ax.scatter(part.x * row.scale_factor, part.y * row.scale_factor,
                   s=7, color=pattern_colors[pattern], alpha=0.50,
                   linewidth=0, label=pattern)
    selected = frame[frame.niche == niche]
    ax.scatter(selected.x * row.scale_factor, selected.y * row.scale_factor,
               s=30, facecolor="none", edgecolor="black", linewidth=0.7,
               label=f"Selected: {niche_label(niche)}")
    ax.set(xlim=(0, image.shape[1]), ylim=(image.shape[0], 0))
    ax.set_title(f"{stage_labels[stage]} — {sample}")
    ax.axis("off")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    st.pyplot(fig, width="stretch")
    st.caption("H&E section with Visium spots coloured by perturbation pattern. Black circles mark the selected niche.")

with biology:
    annotation = patterns.loc[niche]
    pattern = annotation.pattern
    st.subheader(niche_label(niche))
    st.markdown(f"**ML macro-pattern:** {pattern_label(pattern)}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Compartment:** {annotation.compartment_class}")
        st.markdown(f"**Cellular identity:** {annotation.key_cell_types}")
        st.markdown(f"**Molecular programme:** {annotation.functional_program}")
    with c2:
        st.markdown(f"**Spatial context:** {annotation.spatial_context}")
        st.markdown(f"**Published response:** {annotation.published_response}")
        st.markdown(f"**Evidence:** {annotation.evidence_level}")
    if niche == "N2":
        st.info(
            "N2 is a biologically important exception: the paper describes a persistent, "
            "chemotherapy-tolerant EMT niche. Its ML assignment reflects the complete "
            "24-feature trajectory and must not be read as proof of strong contraction."
        )
    metric = st.selectbox(
        "Feature",
        ["abundance", "emt", "proliferation", "hypoxia", "immune", "fibroblast", "macrophage", "tumour"],
    )
    frame = profiles[profiles.niche == niche].copy()
    frame["stage"] = pd.Categorical(frame.stage, stage_order, ordered=True)
    summary = frame.groupby("stage", observed=True)[metric].agg(["mean", "sem"]).reindex(stage_order)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, current_stage in enumerate(stage_order):
            values = frame.loc[frame.stage == current_stage, metric]
            ax.scatter([i] * len(values), values, color="#8C8C8C", alpha=0.65, s=25)
        ax.errorbar(range(len(stage_order)), summary["mean"], yerr=summary["sem"],
                    marker="o", color="#1F77B4", lw=2, capsize=4)
        ax.set(xticks=range(len(stage_order)), xticklabels=[stage_labels[x] for x in stage_order],
               ylabel=metric.replace("_", " ").title(), title=f"{niche_label(niche)} trajectory")
        sns.despine(ax=ax)
        st.pyplot(fig, width="stretch")
    with right:
        values = matrix["mrd12:abundance"].sort_values()
        colours = [pattern_colors[patterns.loc[x, "pattern"]] for x in values.index]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([niche_label(n) for n in values.index], values, color=colours)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set(xlabel="Standardised change vs primary", ylabel="Niche",
               title="Niche abundance in MRD at day 12")
        sns.despine(ax=ax)
        st.pyplot(fig, width="stretch")
    st.caption("Tumour-level abundance effects with 95% bootstrap intervals")
    st.dataframe(
        effects[effects.niche == niche][
            ["stage", "mean_difference", "ci_low", "ci_high", "direction_confidence", "n_stage"]
        ],
        width="stretch", hide_index=True,
    )

with diagnostics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected k", model["selected_k"])
    c2.metric("Silhouette", f"{model['silhouette']:.2f}")
    c3.metric("Bootstrap stability", f"{model['mean_stability']:.2f}")
    c4.metric("Leave-one-tumour-out", f"{model['mean_leave_one_tumour_out_ari']:.2f}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PCA variance (2D)", f"{100 * model['pca_variance_2d']:.1f}%")
    c2.metric("Feature sensitivity", f"{model['minimum_feature_sensitivity_ari']:.2f}")
    c3.metric("TAC agreement", f"{model['validation_treatment_ari']:.2f}")
    c4.metric("Median TAC correlation", f"{model['median_validation_spearman']:.2f}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Consensus PAC", f"{model['consensus_pac']:.3f}")
    c2.metric("Cluster permutation p", f"{model['cluster_permutation_p']:.4f}")
    c3.metric("Null silhouette (95th)", f"{model['null_silhouette_95th']:.2f}")
    st.caption(
        f"TAC permutation tests: partition p={model['validation_ari_permutation_p']:.3f}; "
        f"median correlation p={model['validation_spearman_permutation_p']:.4f}. "
        "TAC shares the untreated primary reference and is not an external cohort."
    )
    st.image(str(ROOT / "results" / "figures" / "model_diagnostics.png"), width="stretch")
    st.info(
        "This is an unsupervised model, so there is no accuracy score. "
        "Reliability is assessed by tumour-level bootstrap, leave-one-tumour-out refitting, "
        "feature-removal sensitivity and comparison with the separate TAC treatment arm."
    )
    st.markdown(
        "**Interpretation:** the two-macro-pattern solution is robust within this dataset, "
        "but recurrence estimates remain uncertain because the cisplatin arm contains only two recurrent tumours. "
        "Perfect ARI values describe the reproducibility of this coarse two-way split; they are not prediction accuracy. "
        "This is not a clinical classifier."
    )
    with st.expander("Detailed robustness tables"):
        st.write("Leave-one-tumour-out")
        st.dataframe(pd.read_csv(TABLES / "leave_one_tumour_out.csv"), width="stretch", hide_index=True)
        st.write("Leave-one-feature-family-out")
        st.dataframe(pd.read_csv(TABLES / "feature_sensitivity.csv"), width="stretch", hide_index=True)
        st.write("Cisplatin versus TAC")
        st.dataframe(pd.read_csv(TABLES / "tac_reproducibility.csv"), width="stretch", hide_index=True)
