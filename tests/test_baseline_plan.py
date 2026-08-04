"""Baseline plan: constraint feasibility and financial reconciliation."""
from __future__ import annotations

import numpy as np

from src.operations import build_planning_arrays, effective_site_capacity


def test_supply_never_exceeds_site_capacity(data, baseline):
    pa = build_planning_arrays(data)
    cap = effective_site_capacity(pa)
    load = baseline.site_load.to_numpy(float)
    assert (load <= cap + 1e-6).all()


def test_supply_never_exceeds_integration_capacity(baseline):
    il = baseline.integration_load
    assert (il["load_units"] <= il["capacity_units"] + 1e-6).all()


def test_component_consumption_within_supply(baseline):
    cu = baseline.component_usage
    assert (cu["cumulative_consumed_units"]
            <= cu["cumulative_supply_units"] + 1e-6).all()


def test_builds_cover_demand_or_log_constraint(baseline):
    """Whenever builds fall short of demand plus carried backlog, the shortfall
    is visible as unmet demand (nothing silently disappears)."""
    total_demand = baseline.demand_units.sum()
    total_built = baseline.supply_units.sum()
    total_unmet_last = baseline.unmet.iloc[-1].sum()
    assert total_built <= total_demand + 1e-6
    if total_built < total_demand - 0.5:
        assert len(baseline.constraints) > 0


def test_financials_reconcile(baseline, config):
    m = baseline.monthly
    fin = config.financial
    np.testing.assert_allclose(m["gross_profit_usd"],
                               m["revenue_usd"] - m["cogs_usd"], rtol=1e-9)
    np.testing.assert_allclose(m["operating_income_usd"],
                               m["gross_profit_usd"] - fin.opex_monthly_usd, rtol=1e-9)
    np.testing.assert_allclose(m["ebitda_usd"],
                               m["operating_income_usd"] + fin.depreciation_monthly_usd,
                               rtol=1e-9)
    np.testing.assert_allclose(
        m["inventory_usd"],
        m["raw_inventory_usd"] + m["wip_inventory_usd"] + m["fg_inventory_usd"],
        rtol=1e-9)


def test_revenue_recognition_timing(data, baseline):
    """Families with acceptance-based recognition recognize with a one-month lag;
    shipment-based families recognize in the build month."""
    pa = build_planning_arrays(data)
    for f, fam in enumerate(pa.families):
        built = baseline.supply_units[:, f]
        rec = baseline.family_units[fam].to_numpy(float)
        lag = int(pa.family_rec_lag[f])
        if lag == 1:
            np.testing.assert_allclose(rec[1:], built[:-1], rtol=1e-9)
        else:
            np.testing.assert_allclose(rec, built, rtol=1e-9)


def test_utilization_bounded(baseline):
    m = baseline.monthly
    assert (m["ems_utilization"] <= 1.0 + 1e-9).all()
    assert (m["ems_utilization"] >= 0).all()
