"""Published biological names used to interpret the Chrysalis compartments."""

import pandas as pd


PAPER_URL = "https://doi.org/10.1038/s41467-026-74125-6"

PRIMARY_PATTERN = "Primary/recurrent tissue architecture"
MRD_PATTERN = "MRD-associated immune-stromal remodelling"
MIXED_PATTERN = "Persistent mixed remodelling"


NICHE_ANNOTATIONS = {
    "N0": {
        "biological_name": "Subset-specific tumour",
        "compartment_class": "Tumour",
        "key_cell_types": "Tumour cells; no unique assignment",
        "functional_program": "Tumour-associated programme",
        "spatial_context": "Restricted to a subset of tumours",
        "published_response": "Low prevalence; variable between samples",
        "evidence_level": "Published, limited assignment",
    },
    "N1": {
        "biological_name": "Macrophage-rich niche",
        "compartment_class": "Immune TME",
        "key_cell_types": "Macrophages and monocytes",
        "functional_program": "Myeloid defence; TREM2/APOE signalling",
        "spatial_context": "Immune-rich tumour microenvironment",
        "published_response": "Enriched during residual disease",
        "evidence_level": "Published",
    },
    "N2": {
        "biological_name": "EMT tumour niche",
        "compartment_class": "Tumour",
        "key_cell_types": "Luminal-alveolar EMT tumour cells",
        "functional_program": "EMT; motility; adhesion; plasticity",
        "spatial_context": "Dispersed tumour foci without a fixed location",
        "published_response": "Persists during treatment; drug-tolerant state",
        "evidence_level": "Published and functionally investigated",
    },
    "N4": {
        "biological_name": "Fibrotic reactive stroma",
        "compartment_class": "Stromal TME",
        "key_cell_types": "Fibroblasts and connective-tissue cells",
        "functional_program": "Collagen and extracellular-matrix organisation",
        "spatial_context": "Fibrotic stroma proximal to tumour cells",
        "published_response": "Remodelled during residual disease",
        "evidence_level": "Published",
    },
    "N5": {
        "biological_name": "Proliferating tumour",
        "compartment_class": "Tumour",
        "key_cell_types": "Proliferating and basal tumour cells",
        "functional_program": "Cell cycle; DNA replication; E2F/MYC",
        "spatial_context": "Large regions of the primary tumour mass",
        "published_response": "Strongly depleted during residual disease",
        "evidence_level": "Published and functionally investigated",
    },
    "N7": {
        "biological_name": "Fibro-inflammatory TME",
        "compartment_class": "Stromal/immune TME",
        "key_cell_types": "Fibroblasts with immune infiltration",
        "functional_program": "Connective-tissue and complement-associated programmes",
        "spatial_context": "Intermediate stromal-to-immune compartment",
        "published_response": "Expanded with TME remodelling during MRD",
        "evidence_level": "Published, mixed assignment",
    },
    "N8": {
        "biological_name": "Chemo-depleted tumour",
        "compartment_class": "Tumour",
        "key_cell_types": "Tumour cells; no unique assignment",
        "functional_program": "Tumour-associated programme",
        "spatial_context": "Tumour compartment present before treatment",
        "published_response": "Selectively lost after chemotherapy",
        "evidence_level": "Published, limited assignment",
    },
    "N9": {
        "biological_name": "Contractile stroma",
        "compartment_class": "Stromal TME",
        "key_cell_types": "Smooth-muscle and stromal cells",
        "functional_program": "Contractile and structural programmes",
        "spatial_context": "Smooth-muscle-enriched stromal regions",
        "published_response": "Expanded with stromal remodelling during MRD",
        "evidence_level": "Published",
    },
    "N10": {
        "biological_name": "Hypoxic tumour",
        "compartment_class": "Tumour",
        "key_cell_types": "Hypoxic tumour cells",
        "functional_program": "Hypoxia; glycolysis; metabolic stress; VEGFA/CD44",
        "spatial_context": "Adjacent to necrotic cores",
        "published_response": "Strongly depleted during residual disease",
        "evidence_level": "Published and functionally investigated",
    },
    "N11": {
        "biological_name": "Activated immune niche",
        "compartment_class": "Immune TME",
        "key_cell_types": "Activated immune cells and macrophages",
        "functional_program": "Defence response and complement signalling",
        "spatial_context": "Immune-rich residual-tumour regions",
        "published_response": "Strongly enriched during residual disease",
        "evidence_level": "Published",
    },
    "N12": {
        "biological_name": "Plasma-cell/humoral niche",
        "compartment_class": "Immune TME",
        "key_cell_types": "Plasma cells and B-lineage cells",
        "functional_program": "B-cell activation and humoral immune response",
        "spatial_context": "Plasmacytic cell aggregates",
        "published_response": "Enriched during residual disease, especially after TAC",
        "evidence_level": "Published",
    },
}


def annotation_frame():
    """Return the curated annotation table in paper compartment order."""
    frame = pd.DataFrame.from_dict(NICHE_ANNOTATIONS, orient="index")
    frame.index.name = "niche"
    frame["source"] = PAPER_URL
    return frame


def annotate_patterns(patterns):
    """Attach biological interpretation without changing the ML assignments."""
    return patterns.join(annotation_frame().reindex(patterns.index))


def niche_label(niche, multiline=False):
    name = NICHE_ANNOTATIONS.get(niche, {}).get("biological_name", "Unassigned niche")
    separator = "\n" if multiline else " · "
    return f"{niche}{separator}{name}"


def pattern_group(pattern):
    """Stable colour/legend key derived from the biological macro-pattern name."""
    return "mrd" if MRD_PATTERN.lower() in pattern.lower() else "primary"


def pattern_label(pattern):
    """Return the biological name, while accepting legacy P-numbered outputs."""
    return pattern.split(" - ", 1)[-1]
