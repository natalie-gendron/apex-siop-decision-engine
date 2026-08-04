"""Synthetic data generation: reproducibility, validity, required scale."""
from __future__ import annotations

import pandas as pd

from src.data_generator import generate_all
from src.utils import EMS_SITES, INTEGRATION_SITES, PRODUCT_FAMILIES, month_labels
from src.validation import validate_inputs


def test_generation_reproducible(tmp_path):
    a = generate_all(seed=123, out_dir=tmp_path / "a")
    b = generate_all(seed=123, out_dir=tmp_path / "b")
    pd.testing.assert_frame_equal(a.demand, b.demand)
    pd.testing.assert_frame_equal(a.components, b.components)
    pd.testing.assert_frame_equal(a.ems_capacity, b.ems_capacity)
    pd.testing.assert_frame_equal(a.financial_plan, b.financial_plan)


def test_different_seed_differs(tmp_path):
    a = generate_all(seed=1, out_dir=tmp_path / "a")
    b = generate_all(seed=2, out_dir=tmp_path / "b")
    assert not a.demand["base_forecast_units"].equals(b.demand["base_forecast_units"])


def test_generated_data_passes_validation(data):
    issues = validate_inputs(data)
    assert [i for i in issues if i.severity == "error"] == []


def test_required_scale(data):
    assert set(data.demand["product_family"].unique()) == set(PRODUCT_FAMILIES)
    assert len(PRODUCT_FAMILIES) == 5
    assert data.demand["customer"].nunique() >= 8
    assert set(data.ems_capacity["ems_site"].unique()) == set(EMS_SITES)
    assert len(EMS_SITES) == 4
    assert data.integration_capacity["integration_site"].nunique() == len(INTEGRATION_SITES) == 2
    assert 25 <= len(data.components) <= 40
    assert data.demand["month"].nunique() == 18
    assert list(data.financial_plan["month"]) == month_labels()


def test_files_written(tmp_path):
    generate_all(seed=5, out_dir=tmp_path)
    for name in ["demand_plan", "products", "components", "ems_sites",
                 "ems_capacity", "integration_capacity", "financial_plan"]:
        assert (tmp_path / f"{name}.csv").exists()


def test_demand_probabilities_bounded(data):
    for col in ["pull_in_prob", "push_out_prob", "cancel_prob",
                "forecast_confidence", "site_readiness_prob"]:
        assert data.demand[col].between(0, 1).all(), col


def test_demand_is_lumpy(data):
    """Coefficient of variation of monthly family demand should exceed a smooth
    series' — the spec requires lumpy, not smoothed, demand."""
    dem = data.demand.copy()
    dem["units"] = dem["base_forecast_units"] + dem["backlog_units"]
    monthly = dem.groupby(["product_family", "month"])["units"].sum()
    cv = monthly.groupby("product_family").std() / monthly.groupby("product_family").mean()
    assert (cv > 0.15).all()
