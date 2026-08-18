"""Small, explicit functions used by the unsupervised niche analysis."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.annotations import MIXED_PATTERN, MRD_PATTERN, PRIMARY_PATTERN


REQUIRED_META = ["sample_id", "stage", "treatment", "niche"]
OPTIONAL_META = ["batch", "slide", "parental_tumor"]


def aggregate_spots(spots, features):
    """Make one weighted profile for each tumour and published niche."""
    missing = set(REQUIRED_META + features) - set(spots.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    weight_cols = sorted(
        [c for c in spots if c.startswith("weight_N")],
        key=lambda x: int(x.split("N")[-1]),
    )
    if weight_cols:
        return _aggregate_soft(spots, features, weight_cols)

    keys = [*REQUIRED_META[:-1], *[x for x in OPTIONAL_META if x in spots], "niche"]
    profiles = spots.groupby(keys, observed=True)[features].mean().reset_index()
    counts = spots.groupby(keys, observed=True).size().rename("n_spots").reset_index()
    totals = spots.groupby("sample_id").size().rename("n_total")
    profiles = profiles.merge(counts, on=keys).merge(totals, on="sample_id")
    profiles["abundance"] = profiles["n_spots"] / profiles["n_total"]
    return profiles.drop(columns="n_total")


def _aggregate_soft(spots, features, weight_cols):
    rows = []
    for sample_id, frame in spots.groupby("sample_id", observed=True):
        total_weight = frame[weight_cols].to_numpy().sum()
        metadata = {
            key: frame[key].iloc[0]
            for key in ["stage", "treatment", *OPTIONAL_META]
            if key in frame
        }
        for col in weight_cols:
            weight = frame[col].to_numpy()
            if weight.sum() == 0:
                continue
            rows.append({
                "sample_id": sample_id, **metadata, "niche": col.removeprefix("weight_"),
                **{x: np.average(frame[x], weights=weight) for x in features},
                "n_spots": int((weight > 0).sum()),
                "abundance": weight.sum() / total_weight,
            })
    return pd.DataFrame(rows)


def perturbation_matrix(profiles, features, stages, baseline="primary"):
    """Batch-matched effects; niche abundance is analysed as a composition."""
    # Keep abundance compositional; the remaining features are ordinary scores.
    metrics = ["abundance", *features]
    data = profiles.copy()
    data["abundance"] = _clr_abundance(data)
    primary = data[data.stage == baseline]
    if primary.empty:
        raise ValueError(f"Missing baseline stage: {baseline}")

    for metric in metrics:
        scale = primary[metric].std(ddof=0)
        data[metric] = data[metric] / (scale if scale > 0 else 1)

    use_batch = "batch" in data and data.groupby("batch")["stage"].apply(
        lambda x: baseline in set(x)
    ).all()
    if use_batch:
        # A treated section is contrasted with primary sections from its own batch.
        reference = primary.groupby(["batch", "niche"], observed=True)[metrics].mean()
        treated = data[data.stage != baseline].merge(
            reference, left_on=["batch", "niche"], right_index=True,
            suffixes=("", "_primary"), validate="many_to_one",
        )
        for metric in metrics:
            treated[metric] -= treated[f"{metric}_primary"]
        means = treated.groupby(["stage", "niche"], observed=True)[metrics].mean()
    else:
        means = data.groupby(["stage", "niche"], observed=True)[metrics].mean()
        reference = means.loc[baseline]
        means = pd.concat({
            stage: means.loc[stage] - reference
            for stage in stages if stage != baseline and stage in means.index.get_level_values("stage")
        }, names=["stage"])

    blocks = []
    for stage in stages:
        if stage == baseline or stage not in means.index.get_level_values("stage"):
            continue
        block = means.loc[stage].copy()
        block.columns = [f"{stage}:{metric}" for metric in metrics]
        blocks.append(block)
    if not blocks:
        raise ValueError("No stage available for comparison with baseline")
    return pd.concat(blocks, axis=1).dropna()


def _clr_abundance(profiles):
    wide = profiles.pivot(index="sample_id", columns="niche", values="abundance")
    positive = wide.to_numpy()[wide.to_numpy() > 0]
    pseudocount = positive.min() / 2 if len(positive) else 1e-6
    logged = np.log(wide + pseudocount)
    clr = logged.sub(logged.mean(axis=1), axis=0).stack()
    index = pd.MultiIndex.from_frame(profiles[["sample_id", "niche"]])
    return clr.reindex(index).to_numpy()


def fit_patterns(matrix, n_clusters=3, seed=42, labels=None):
    """Project profiles for display and assign Ward-clustering labels."""
    x = StandardScaler().fit_transform(matrix)
    tree = linkage(x, method="ward", optimal_ordering=True)
    if labels is None:
        labels = fcluster(tree, n_clusters, criterion="maxclust")
    labels = np.asarray(labels)
    pca = PCA(n_components=2, random_state=seed)
    xy = pca.fit_transform(x)
    out = pd.DataFrame({
        "niche": matrix.index, "cluster": labels,
        "PC1": xy[:, 0], "PC2": xy[:, 1],
    }).set_index("niche")
    out["pattern"] = _pattern_names(matrix, labels)
    return out, tree, pca.explained_variance_ratio_


def bootstrap_stability(profiles, features, stages, baseline, n_clusters,
                        n_bootstrap=500, seed=42, feature_dropout=0.7):
    """Consensus over tumour resampling and random feature-family removal."""
    base = perturbation_matrix(profiles, features, stages, baseline)
    niches = base.index
    consensus = np.zeros((len(niches), len(niches)))
    rng = np.random.default_rng(seed)
    strata = ["stage", "batch"] if "batch" in profiles else ["stage"]
    sample_meta = profiles.drop_duplicates("sample_id")

    for iteration in range(n_bootstrap):
        # Sample tumours, never individual spots: spots are not independent replicates.
        sampled = []
        for stratum, frame in sample_meta.groupby(strata, observed=True):
            ids = frame.sample_id.to_numpy()
            for j, sample_id in enumerate(rng.choice(ids, len(ids), replace=True)):
                part = profiles[profiles.sample_id == sample_id].copy()
                part["sample_id"] = f"b{iteration}_{j}_{sample_id}"
                sampled.append(part)
        matrix = perturbation_matrix(pd.concat(sampled), features, stages, baseline).reindex(niches)
        if rng.random() < feature_dropout:
            metrics = sorted({column.split(":", 1)[1] for column in matrix})
            omitted = rng.choice(metrics)
            matrix = matrix[[c for c in matrix if not c.endswith(f":{omitted}")]]
        labels = fit_patterns(matrix, n_clusters, seed + iteration + 1)[0].cluster.to_numpy()
        consensus += labels[:, None] == labels[None, :]

    consensus /= n_bootstrap
    labels = consensus_partition(consensus, n_clusters)
    stability = []
    for i, label in enumerate(labels):
        same = np.where(labels == label)[0]
        same = same[same != i]
        if len(same):
            stability.append(consensus[i, same].mean())
        else:
            stability.append(1 - np.max(np.delete(consensus[i], i)))
    return (
        pd.Series(stability, index=niches, name="stability"),
        pd.DataFrame(consensus, index=niches, columns=niches),
        pd.Series(labels, index=niches, name="cluster"),
    )


def consensus_partition(consensus, n_clusters):
    values = np.asarray(consensus, dtype=float)
    distance = np.clip(1 - values, 0, 1)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average", optimal_ordering=True)
    return fcluster(tree, n_clusters, criterion="maxclust")


def cluster_diagnostics(matrix, selected_k=3, max_k=6):
    x = StandardScaler().fit_transform(matrix)
    tree = linkage(x, method="ward")
    rows = []
    for k in range(2, min(max_k, len(matrix) - 1) + 1):
        labels = fcluster(tree, k, criterion="maxclust")
        counts = pd.Series(labels).value_counts()
        rows.append({
            "k": k, "silhouette": silhouette_score(x, labels),
            "smallest_cluster": int(counts.min()), "selected": k == selected_k,
        })
    return pd.DataFrame(rows)


def partition_silhouette(matrix, labels):
    return silhouette_score(StandardScaler().fit_transform(matrix), np.asarray(labels))


def cluster_permutation_test(matrix, labels, n_permutations=5000, seed=42):
    """Test whether multivariate structure is stronger than shuffled features."""
    x = StandardScaler().fit_transform(matrix)
    observed = silhouette_score(x, np.asarray(labels))
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        # Shuffling each feature breaks cross-feature structure while keeping its values.
        shuffled = np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])])
        tree = linkage(shuffled, method="ward")
        trial_labels = fcluster(tree, len(np.unique(labels)), criterion="maxclust")
        null[i] = silhouette_score(shuffled, trial_labels)
    return {
        "observed_silhouette": observed,
        "null_silhouette_95th": np.quantile(null, 0.95),
        "permutation_p": (1 + (null >= observed).sum()) / (n_permutations + 1),
    }


def leave_one_tumour_out(profiles, features, stages, baseline, n_clusters,
                         seed=42, reference_labels=None):
    reference = perturbation_matrix(profiles, features, stages, baseline)
    if reference_labels is None:
        reference_labels = fit_patterns(reference, n_clusters, seed)[0].cluster
    rows = []
    sample_stage = profiles.drop_duplicates("sample_id").set_index("sample_id")["stage"]
    for i, sample_id in enumerate(sample_stage.index):
        matrix = perturbation_matrix(
            profiles[profiles.sample_id != sample_id], features, stages, baseline
        ).reindex(reference.index)
        labels = fit_patterns(matrix, n_clusters, seed + i + 1)[0].cluster
        rows.append({
            "sample_id": sample_id, "stage": sample_stage[sample_id],
            "adjusted_rand_index": adjusted_rand_score(reference_labels, labels),
        })
    return pd.DataFrame(rows)


def feature_sensitivity(matrix, n_clusters, seed=42, reference_labels=None):
    if reference_labels is None:
        reference_labels = fit_patterns(matrix, n_clusters, seed)[0].cluster
    metrics = sorted({column.split(":", 1)[1] for column in matrix})
    rows = []
    for i, metric in enumerate(metrics):
        keep = [column for column in matrix if not column.endswith(f":{metric}")]
        labels = fit_patterns(matrix[keep], n_clusters, seed + i + 1)[0].cluster
        rows.append({
            "omitted_feature": metric,
            "adjusted_rand_index": adjusted_rand_score(reference_labels, labels),
        })
    return pd.DataFrame(rows)


def treatment_reproducibility(reference, validation, n_clusters, seed=42,
                              n_permutations=0, reference_labels=None,
                              validation_labels=None):
    niches = reference.index.intersection(validation.index)
    columns = reference.columns.intersection(validation.columns)
    left = reference.loc[niches, columns]
    right = validation.loc[niches, columns]
    if reference_labels is None:
        reference_labels = fit_patterns(left, n_clusters, seed)[0].cluster
    if validation_labels is None:
        validation_labels = fit_patterns(right, n_clusters, seed)[0].cluster
    reference_labels = pd.Series(reference_labels, index=niches).loc[niches].to_numpy()
    validation_labels = pd.Series(validation_labels, index=niches).loc[niches].to_numpy()
    correlations = np.array([
        [left.loc[a].corr(right.loc[b], method="spearman") for b in niches]
        for a in niches
    ])
    observed_ari = adjusted_rand_score(reference_labels, validation_labels)
    rows = [{"niche": niche, "spearman_r": correlations[i, i]} for i, niche in enumerate(niches)]
    stats = {"ari_permutation_p": np.nan, "median_spearman_permutation_p": np.nan}
    if n_permutations:
        rng = np.random.default_rng(seed)
        null_ari = np.empty(n_permutations)
        null_correlation = np.empty(n_permutations)
        for i in range(n_permutations):
            order = rng.permutation(len(niches))
            null_ari[i] = adjusted_rand_score(reference_labels, validation_labels[order])
            null_correlation[i] = np.median(correlations[np.arange(len(niches)), order])
        observed_correlation = np.median(np.diag(correlations))
        stats = {
            "ari_permutation_p": (1 + (null_ari >= observed_ari).sum()) / (n_permutations + 1),
            "median_spearman_permutation_p": (
                1 + (null_correlation >= observed_correlation).sum()
            ) / (n_permutations + 1),
        }
    return observed_ari, pd.DataFrame(rows), stats


def abundance_effects(profiles, stages, baseline="primary", n_bootstrap=2000, seed=42):
    """Batch-matched raw abundance differences with tumour-level intervals."""
    rng = np.random.default_rng(seed)
    rows = []
    batches = profiles.batch.unique() if "batch" in profiles else [None]
    for niche, niche_frame in profiles.groupby("niche", observed=True):
        for stage in stages:
            if stage == baseline:
                continue
            observed = []
            groups = []
            for batch in batches:
                frame = niche_frame if batch is None else niche_frame[niche_frame.batch == batch]
                primary = frame.loc[frame.stage == baseline, "abundance"].to_numpy()
                treated = frame.loc[frame.stage == stage, "abundance"].to_numpy()
                if len(primary) and len(treated):
                    observed.extend(treated - primary.mean())
                    groups.append((primary, treated))
            if not observed:
                continue
            draws = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                values = []
                for primary, treated in groups:
                    reference = rng.choice(primary, len(primary), replace=True).mean()
                    values.extend(rng.choice(treated, len(treated), replace=True) - reference)
                draws[i] = np.mean(values)
            rows.append({
                "niche": niche, "stage": stage, "mean_difference": np.mean(observed),
                "ci_low": np.quantile(draws, 0.025), "ci_high": np.quantile(draws, 0.975),
                "direction_confidence": max((draws > 0).mean(), (draws < 0).mean()),
                "n_stage": len(observed),
                "n_primary": sum(len(primary) for primary, _ in groups),
            })
    return pd.DataFrame(rows)


def pattern_abundance(profiles, patterns):
    data = profiles.merge(patterns[["pattern"]], left_on="niche", right_index=True)
    keys = ["sample_id", "stage", "treatment", "pattern"]
    return data.groupby(keys, observed=True, as_index=False)["abundance"].sum()


def _pattern_names(matrix, labels):
    names = {}
    for label in np.unique(labels):
        mean = matrix.loc[labels == label].mean()
        mrd = pd.concat([mean.filter(like="mrd7:"), mean.filter(like="mrd12:")])
        abundance = mrd[mrd.index.str.endswith(":abundance")].mean()
        if abundance > 0.20:
            title = MRD_PATTERN
        elif abundance < -0.20:
            title = PRIMARY_PATTERN
        else:
            title = MIXED_PATTERN
        names[label] = title
    return [names[x] for x in labels]
