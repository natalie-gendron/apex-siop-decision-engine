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


def test_prebuilt_scenarios_are_exogenous_worlds():
    """Scenarios are world-states: no decision costs (responses live in the
    management-action catalog)."""
    scens = prebuilt_scenarios()
    assert len(scens) == 8
    assert "Base Case" in scens
    assert all(s.action_cost_usd == 0.0 for s in scens.values())


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


def test_family_arrays_reconcile_to_totals(base_result):
    """Per-family shipped/demanded must sum to the aggregate arrays the
    past-due curve uses (same measure, decomposed)."""
    np.testing.assert_allclose(base_result.family_demand.sum(axis=2),
                               base_result.units_demanded)
    np.testing.assert_allclose(base_result.family_shipped.sum(axis=2),
                               base_result.units_shipped)


def test_catalog_from_text_matches_file_and_validates():
    """The session what-if path (catalog_text) must load identically to the
    repo file and reject claims the simulator cannot price."""
    from pathlib import Path

    text = (Path(__file__).parent.parent / "config"
            / "management_actions.yaml").read_text(encoding="utf-8")
    from_text = management_actions(catalog_text=text)
    from_file = management_actions()
    assert list(from_text) == list(from_file)
    assert all(from_text[n].overrides == from_file[n].overrides
               for n in from_file)
    bad = ("actions:\n"
           "  - name: Bad claim\n"
           "    description: Uses a knob the simulator does not know.\n"
           "    horizon: Tactical\n"
           "    overrides:\n"
           "      warp_drive_mult: 2.0\n")
    with pytest.raises(ValueError, match="warp_drive_mult"):
        management_actions(catalog_text=bad)


def test_response_package_stacks_actions(data, config, baseline):
    """A response package = merged action overrides + summed decision cost;
    the simulator charges the cost to Q1 operating income and nothing else."""
    from src.market_intelligence import merge_confidence_params

    acts = management_actions()
    names = ["Expedite critical component receipts",
             "Authorize overtime at EMS sites"]
    merged: dict = {}
    for n in names:
        merged = merge_confidence_params(merged, acts[n].overrides)
    cost = sum(acts[n].action_cost_usd for n in names) + 6.0e6
    with_cost = dict(merged, action_cost_usd=cost)
    a = run_simulation(data, config, baseline, params=merged, n_sims=N, seed=7)
    b = run_simulation(data, config, baseline, params=with_cost, n_sims=N,
                       seed=7)
    assert np.isfinite(b.revenue).all()
    np.testing.assert_array_equal(a.revenue, b.revenue)   # cost is P&L-only
    d_oi = (fiscal_year(a.operating_income).mean()
            - fiscal_year(b.operating_income).mean())
    assert d_oi == pytest.approx(cost, rel=1e-6)


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
