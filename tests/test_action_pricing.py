"""Action pricing must difference against a reference at the SAME path count.

Regression tests for the subsample bias found in the 2026-08 external review:
action runs at min(n_sims, 2000) paths were differenced against headline KPIs
at the full path count, which under common random numbers is not a clean
difference — 11 of 14 actions showed an identical spurious +1.61 pt
Δ P(Q1 plan), including actions inert until month 7+.
"""
from __future__ import annotations

import pytest

from src.market_intelligence import CONFIDENCE_SIM_PARAMS, merge_confidence_params
from src.recommendations import build_recommendations
from src.scenarios import kpi_summary, management_actions
from src.sensitivity import binding_components
from src.simulation import run_simulation

# actions whose overrides only take effect from month 7 or later — they
# cannot move Q1, so their Q1 delta against an equal-count CRN reference
# must be exactly zero
INERT_IN_Q1 = [
    "Dual-source the high-end FPGA",              # supply ramp from month 7
    "Commit long-lead component orders",          # receipts months 7+
    "Qualify EMS Eastern Europe for Zenith Compute",  # usable from month 10
    "Expand final-integration headcount",         # capacity from month 7
]


@pytest.fixture(scope="module")
def conf_params():
    return dict(CONFIDENCE_SIM_PARAMS["Moderate"])


@pytest.fixture(scope="module")
def ref_700(data, config, baseline, conf_params):
    """No-action reference at a path count DIFFERENT from the base fixture's
    1,500 — the same shape as the app's 2,000-path action pricing under the
    5,000-path headline."""
    r = run_simulation(data, config, baseline, params=conf_params,
                       n_sims=700, seed=42, scenario_name="reference")
    return kpi_summary(r, baseline, config)


@pytest.fixture(scope="module")
def actions_700(data, config, baseline, conf_params):
    catalog = management_actions()
    out = {}
    for name in INERT_IN_Q1 + ["Authorize overtime at EMS sites"]:
        spec = catalog[name]
        r = run_simulation(data, config, baseline,
                           params=merge_confidence_params(conf_params,
                                                          spec.overrides),
                           n_sims=700, seed=42, scenario_name=name)
        out[name] = (kpi_summary(r, baseline, config), spec)
    return out


def test_inert_actions_show_zero_q1_delta_at_equal_paths(ref_700, actions_700):
    """The P1 acceptance property: actions whose overrides start at month 7+
    show ΔP(Q1 plan) ≈ 0.0 pts (±0.1) against an equal-count reference."""
    for name in INERT_IN_Q1:
        kpi, _ = actions_700[name]
        d_q1 = kpi["p_q1_plan"] - ref_700["p_q1_plan"]
        assert abs(d_q1) <= 0.001, (
            f"{name}: ΔP(Q1 plan) = {d_q1 * 100:+.2f} pts against the "
            f"equal-count reference — the action cannot touch Q1")
        d_q1_rev = kpi["q1_revenue"]["mean"] - ref_700["q1_revenue"]["mean"]
        assert abs(d_q1_rev) < 1e6, (
            f"{name}: Δ Q1 revenue = {d_q1_rev:+,.0f} against the "
            f"equal-count reference")


def test_recommendation_deltas_use_equal_count_reference(
        config, baseline, base_result, ref_700, actions_700):
    """build_recommendations must difference action KPIs against ref_kpi
    (equal path count), not the headline base_kpi, when the two differ."""
    base_kpi = kpi_summary(base_result, baseline, config)  # 1,500 paths
    binding = binding_components(base_result)
    recs = build_recommendations(base_kpi, actions_700, binding,
                                 ref_kpi=ref_700)
    assert recs, "expected at least one recommendation from the sample set"
    for rec in recs:
        kpi, spec = actions_700[rec.title]
        expected_ev = (kpi["fy_gross_profit"]["mean"]
                       - ref_700["fy_gross_profit"]["mean"]
                       - spec.action_cost_usd)
        assert rec.expected_value_usd == pytest.approx(expected_ev)
        # inert-in-Q1 actions must not be credited with a Q1 probability gain
        if rec.title in INERT_IN_Q1:
            d_q1 = kpi["p_q1_plan"] - ref_700["p_q1_plan"]
            assert abs(d_q1) <= 0.001
