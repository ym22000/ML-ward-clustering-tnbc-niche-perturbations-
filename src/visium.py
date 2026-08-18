import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from scipy import sparse


def read_visium_folder(folder, gene_sets, treatment=None, exclude_niches=()):
    files = sorted(Path(folder).glob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No h5ad file found in {folder}")
    frames = []
    for i, path in enumerate(files, 1):
        print(f"[{i:02d}/{len(files)}] {path.name}")
        frame = read_visium_h5ad(path, gene_sets, treatment, exclude_niches)
        if len(frame):
            frames.append(frame)
    if not frames:
        raise ValueError("No spot left after filtering")
    return pd.concat(frames, ignore_index=True)


def export_histology_folder(folder, out_dir, treatment=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path(folder).glob("*.h5ad")):
        adata = sc.read_h5ad(path, backed="r")
        obs = adata.obs
        if obs.empty:
            adata.file.close()
            continue
        row = obs.iloc[0]
        stage = _stage(row)
        keep = row["condition"] != "control"
        if treatment:
            keep &= row["treatment"] == treatment or (
                stage == "primary" and row["treatment"] == "no_treatment"
            )
        if not keep or "spatial" not in adata.uns:
            adata.file.close()
            continue

        sample = str(row["sample_id"])
        spatial = adata.uns["spatial"][sample]
        image = np.asarray(spatial["images"]["hires"])
        image = (np.clip(image, 0, 1) * 255).astype("uint8")
        filename = f"{sample}.png"
        Image.fromarray(image).save(out / filename, optimize=True)
        scale = float(spatial["scalefactors"]["tissue_hires_scalef"])
        diameter = float(spatial["scalefactors"]["spot_diameter_fullres"]) * scale
        rows.append({
            "sample_id": sample, "stage": stage, "image_file": filename,
            "scale_factor": scale, "spot_diameter": diameter,
        })
        adata.file.close()
    return pd.DataFrame(rows)


def read_visium_h5ad(path, gene_sets, treatment=None, exclude_niches=()):
    adata = sc.read_h5ad(path)
    required = {"sample_id", "condition", "treatment", "elapsed_time"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Missing AnnData columns: {sorted(missing)}")
    obs = adata.obs.copy()
    obs["stage"] = obs.apply(_stage, axis=1)
    keep = obs["condition"] != "control"
    if treatment:
        keep &= (obs["treatment"] == treatment) | (
            (obs["stage"] == "primary") & (obs["treatment"] == "no_treatment")
        )
    if not keep.any():
        return pd.DataFrame()
    if "chr_aa" not in adata.obsm:
        raise ValueError(f"Missing Chrysalis weights in {path}")
    adata = adata[keep].copy()
    obs = obs.loc[keep].copy()

    weights = adata.obsm["chr_aa"]
    if hasattr(weights, "to_numpy"):
        weights = weights.to_numpy()
    weights = np.asarray(weights)
    niche_ids = [f"N{i}" for i in range(weights.shape[1])]
    excluded = set(exclude_niches)
    kept = [i for i, niche in enumerate(niche_ids) if niche not in excluded]
    if not kept:
        raise ValueError("All Chrysalis niches were excluded")
    niche_ids = [niche_ids[i] for i in kept]
    weights = weights[:, kept]
    retained_mass = weights.sum(axis=1)
    valid = retained_mass > 0
    if not valid.any():
        return pd.DataFrame()
    # Spots explained only by discarded technical compartments are not biological niches.
    # Keep an AnnData view here; copying `.raw` can be needlessly large.
    adata = adata[valid]
    obs = obs.iloc[np.flatnonzero(valid)].copy()
    weights = weights[valid] / retained_mass[valid, None]
    obs["niche"] = np.asarray(niche_ids)[weights.argmax(axis=1)]
    for i, niche in enumerate(niche_ids):
        obs[f"weight_{niche}"] = weights[:, i]

    coords = np.asarray(adata.obsm["spatial"])
    obs["x"], obs["y"] = coords[:, 0], coords[:, 1]
    published = _published_features(adata)
    for name, genes in gene_sets.items():
        obs[name] = published[name] if name in published else _mean_expression(adata, genes, name)

    obs["spot_id"] = obs.index.astype(str)
    optional = [x for x in ["batch", "slide", "parental_tumor", "replicates"] if x in obs]
    columns = [
        "spot_id", "sample_id", "stage", "condition", "treatment",
        "elapsed_time", *optional, "niche", "x", "y", *gene_sets,
        *[f"weight_{niche}" for niche in niche_ids],
    ]
    return obs[columns].reset_index(drop=True)


def _published_features(adata):
    if "cell2loc" not in adata.obsm:
        return {}
    cells = adata.obsm["cell2loc"]
    if not hasattr(cells, "columns"):
        return {}

    tumour_cols = [
        "TB-EMT", "TBasal1", "TBasal2", "TFlike", "TL-Alv",
        "TLA-EMT", "TMlike", "TProliferating",
    ]
    macrophage_cols = ["CHMacrophage", "InflMacrophage", "Spp1Macrophage"]
    immune_cols = [
        "APC", "Bcell", "CD4T", "CD8T", *macrophage_cols, "NKcell",
        "Plasmacell", "Tcell", "Treg", "cDC", "pDC",
    ]
    total = cells.sum(axis=1).to_numpy()
    tumour_total = cells[tumour_cols].sum(axis=1).to_numpy()

    def ratio(values, denominator):
        return np.divide(values, denominator, out=np.zeros_like(values, dtype=float), where=denominator > 0)

    scores = {
        "emt": ratio(cells[["TB-EMT", "TLA-EMT"]].sum(axis=1).to_numpy(), tumour_total),
        "proliferation": ratio(cells["TProliferating"].to_numpy(), tumour_total),
        "immune": ratio(cells[immune_cols].sum(axis=1).to_numpy(), total),
        "fibroblast": ratio(cells["Fibroblast"].to_numpy(), total),
        "macrophage": ratio(cells[macrophage_cols].sum(axis=1).to_numpy(), total),
        "tumour": ratio(tumour_total, total),
    }
    if "progeny_pathway_activity" in adata.obsm:
        pathway = adata.obsm["progeny_pathway_activity"]
        if hasattr(pathway, "columns") and "Hypoxia" in pathway:
            scores["hypoxia"] = pathway["Hypoxia"].to_numpy()
    return scores


def _stage(row):
    elapsed = str(row.get("elapsed_time", "")).lower()
    if row["condition"] == "residual_tumor":
        return "mrd7" if elapsed.startswith("7") else "mrd12"
    if row["condition"] == "relapsed_tumor":
        return "relapsed"
    if row["condition"] == "primary_tumor" and row["treatment"] != "no_treatment":
        return "mrd7"
    return "primary"


def _mean_expression(adata, genes, label):
    lookup = {str(g).lower(): i for i, g in enumerate(adata.var_names)}
    idx = [lookup[g.lower()] for g in genes if g.lower() in lookup]
    if not idx:
        warnings.warn(f"No gene found for score {label}")
        return np.zeros(adata.n_obs)
    x = adata[:, idx].X
    values = x.mean(axis=1)
    return np.asarray(values).ravel() if sparse.issparse(x) else np.asarray(values).ravel()
