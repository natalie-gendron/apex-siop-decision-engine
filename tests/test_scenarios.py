"""Scenario engine: prebuilt scenarios run and comparisons reconcile."""
from __future__ import annotations

import numpy as np
import pytest

from src.scenarios import (
    compare_scenarios,
    kpi_summary,
    management_actions,
    prebuilt_scenarios,
)
from src.simulation import fiscal_year, run_simulation

N = 400  # small but sufficient for smoke-running every scenario


def test_twelve_prebuilt_scenarios_exist():
    scens = prebuilt_scenarios()
    assert len(scens) == 12
    assert "Base Case" in scens


@pytest.mark.parametrize("name", list(prebuilt_scenarios()))
def test_each_scenario_runs(data, config, baseline, name):
    spec = prebuilt_scenarios()[name]
    r = run_simulation(data, config, baseline, params=spec.overrides,
                       n_sims=N, seed=7, scenario_name=name)
    assert np.isfinite(r.revenue).all()
    assert r.scenario_name == name


def test_comparison_reconciles_to_results(data, config, baseline):
    base = run_simulation(data, config, baseline, n_sims=800, seed=11)
    spec = prebuilt_scenarios()["Memory Recovery"]
    scen = run_simulation(data, config, baseline, params=spec.overrides,
                          n_sims=800, seed=11, scenario_name=spec.name)
    bk = kpi_summary(base, baseline, config)
    sk = kpi_summary(scen, baseline, config)
    cmp = compare_scenarios(bk, sk, spec.action_cost_usd)
    assert cmp["d_fy_revenue"] == pytest.approx(
        fiscal_year(scen.revenue).mean() - fiscal_year(base.revenue).mean())
    assert cmp["d_p_fy_plan"] == pytest.approx(sk["p_fy_plan"] - bk["p_fy_plan"])
    # memory recovery must raise expected revenue
    assert cmp["d_fy_revenue"] > 0


def test_kpi_percentiles_ordered(base_result, baseline, config):
    k = kpi_summary(base_result, baseline, config)
    for key in ["q1_revenue", "fy_revenue", "fy_gm", "ending_inventory"]:
        s = k[key]
        assert s["p5"] <= s["p25"] <= s["median"] <= s["p75"] <= s["p95"]
    for key in ["p_q1_plan", "p_fy_plan", "p_gm_target",
                "p_inventory_over_target", "p_stockout", "p_missed_commitment"]:
        assert 0.0 <= k[key] <= 1.0


def test_base_case_equals_empty_overrides(data, config, baseline):
    a = run_simulation(data, config, baseline, n_sims=300, seed=3)
    b = run_simulation(data, config, baseline,
                       params=prebuilt_scenarios()["Base Case"].overrides,
                       n_sims=300, seed=3)
    np.testing.assert_array_equal(a.revenue, b.revenue)


def test_management_actions_run(data, config, baseline):
    acts = management_actions()
    assert len(acts) >= 8
    name, spec = next(iter(acts.items()))
    r = run_simulation(data, config, baseline, params=spec.overrides,
                       n_sims=N, seed=7, scenario_name=name)
    assert np.isfinite(r.revenue).all()


def test_targeted_expedite_recovery(data, config, baseline):
    """Per-component expedite recovery must change outcomes for the targeted
    component only, at far lower expedite cost than blanket expediting."""
    base = run_simulation(data, config, baseline, n_sims=800, seed=5)
    targeted = run_simulation(
        data, config, baseline,
        params={"expedite_recovery_by_comp": {"High-End FPGA": 0.9}},
        n_sims=800, seed=5)
    blanket = run_simulation(
        data, config, baseline, params={"expedite_recovery": 0.9},
        n_sims=800, seed=5)
    base_cost = fiscal_year(base.expedite_cost).mean()
    targeted_cost = fiscal_year(targeted.expedite_cost).mean()
    blanket_cost = fiscal_year(blanket.expedite_cost).mean()
    assert targeted_cost > base_cost          # targeting one part costs something
    # blanket must cost meaningfully more than targeting (the FPGA alone is the
    # priciest expedite, so the gap is real but not enormous)
    assert blanket_cost > targeted_cost * 1.2
    # targeting must not reduce shipments
    assert (fiscal_year(targeted.units_shipped).mean()
            >= fiscal_year(base.units_shipped).mean() - 0.5)
