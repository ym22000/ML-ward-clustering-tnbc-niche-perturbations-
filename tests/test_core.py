import numpy as np
import pandas as pd

from src.core import (
    aggregate_spots, feature_sensitivity, fit_patterns,
    leave_one_tumour_out, perturbation_matrix,
)
from src.visium import _stage


FEATURES = ["emt", "proliferation", "hypoxia", "immune", "fibroblast", "macrophage", "tumour"]


def toy_spots():
    rows = []
    rng = np.random.default_rng(4)
    stages = ["primary", "mrd7", "mrd12", "relapsed"]
    for stage_i, stage in enumerate(stages):
        for replicate in range(2):
            sample = f"{stage}_{replicate}"
            for niche_i in range(13):
                for spot_i in range(2 + (niche_i + stage_i) % 3):
                    values = rng.normal(size=len(FEATURES)) + stage_i * (niche_i % 4) / 3
                    rows.append({
                        "sample_id": sample, "stage": stage, "treatment": "test",
                        "niche": f"N{niche_i}", **dict(zip(FEATURES, values)),
                    })
    return pd.DataFrame(rows)


def test_profiles_sum_to_one():
    profiles = aggregate_spots(toy_spots(), FEATURES)
    sums = profiles.groupby("sample_id").abundance.sum()
    assert np.allclose(sums, 1)


def test_perturbation_shape():
    profiles = aggregate_spots(toy_spots(), FEATURES)
    matrix = perturbation_matrix(
        profiles, FEATURES, ["primary", "mrd7", "mrd12", "relapsed"], "primary"
    )
    assert matrix.shape == (13, 24)
    assert np.isfinite(matrix.to_numpy()).all()


def test_four_patterns_are_returned():
    profiles = aggregate_spots(toy_spots(), FEATURES)
    matrix = perturbation_matrix(
        profiles, FEATURES, ["primary", "mrd7", "mrd12", "relapsed"], "primary"
    )
    patterns, _, variance = fit_patterns(matrix, 4)
    assert patterns.cluster.nunique() == 4
    assert variance.sum() <= 1


def test_soft_weights_are_used():
    spots = toy_spots()
    spots["weight_N0"] = np.where(spots.niche == "N0", 0.8, 0.1)
    spots["weight_N1"] = np.where(spots.niche == "N1", 0.8, 0.1)
    profiles = aggregate_spots(spots, FEATURES)
    sums = profiles.groupby("sample_id").abundance.sum()
    assert np.allclose(sums, 1)
    assert set(profiles.niche) == {"N0", "N1"}


def test_day_7_and_day_12_are_separate_stages():
    base = {"condition": "residual_tumor", "treatment": "cisplatin_6mg/kg"}
    assert _stage({**base, "elapsed_time": "7_days"}) == "mrd7"
    assert _stage({**base, "elapsed_time": "12_days"}) == "mrd12"


def test_robustness_checks_return_finite_scores():
    profiles = aggregate_spots(toy_spots(), FEATURES)
    stages = ["primary", "mrd7", "mrd12", "relapsed"]
    matrix = perturbation_matrix(profiles, FEATURES, stages, "primary")
    loo = leave_one_tumour_out(profiles, FEATURES, stages, "primary", 3)
    sensitivity = feature_sensitivity(matrix, 3)
    assert np.isfinite(loo.adjusted_rand_index).all()
    assert np.isfinite(sensitivity.adjusted_rand_index).all()
