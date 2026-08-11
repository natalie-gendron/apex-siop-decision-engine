"""Headless UI verification via streamlit.testing.v1.AppTest.

Slow by design: the first run executes the full app at Standard mode
(5,000 headline paths, 2,000-path action pricing) exactly as a user would
see it. These tests pin the two 2026-08 external-review fixes that only
manifest through the app's own wiring:

1. P1 acceptance — action deltas are differenced against an equal-path-count
   CRN reference, so actions inert until month 7+ show ΔP(Q1 plan) ≈ 0.0 pts
   at Standard mode (the bug showed a uniform spurious +1.61 pts).
2. P2.3 — with a sidebar confidence override plus a scenario active,
   scenario-conditioned action pricing runs at the EFFECTIVE (override)
   level, the same backdrop as the world KPIs it is differenced against.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

# actions whose overrides only take effect from month 7 or later
INERT_IN_Q1 = [
    "Dual-source the high-end FPGA",
    "Commit long-lead component orders",
    "Qualify EMS Eastern Europe for Zenith Compute",
    "Expand final-integration headcount",
]


def _actions_table(at: AppTest):
    """The Management Recommendations actions table (the only frame with an
    Incremental EV column)."""
    for d in at.dataframe:
        v = d.value
        if v is not None and "Incremental EV ($M)" in getattr(v, "columns", []):
            return v
    raise AssertionError("actions table not found in the app tree")


def _select(at: AppTest, label_prefix: str):
    for s in at.selectbox:
        if s.label.startswith(label_prefix):
            return s
    raise AssertionError(f"selectbox '{label_prefix}' not found")


@pytest.fixture(scope="module")
def at_standard():
    """One full app run at the defaults (Standard mode, base world)."""
    at = AppTest.from_file(APP_PATH, default_timeout=1800)
    at.run()
    assert not at.exception, at.exception
    return at


def test_inert_actions_show_zero_q1_delta_at_standard_mode(at_standard):
    """P1 acceptance: at Standard mode (5,000 headline / 2,000 action paths),
    actions whose overrides start at month 7+ show ΔP(Q1 plan) ≈ 0.0 pts
    (±0.1) in the actions table."""
    table = _actions_table(at_standard)
    for name in INERT_IN_Q1:
        assert name in table.index, f"action '{name}' missing from the table"
        d_q1 = float(table.loc[name, "Δ P(Q1 plan) (pts)"])
        assert abs(d_q1) <= 0.1, (
            f"{name}: ΔP(Q1 plan) = {d_q1:+.2f} pts — an action inert until "
            f"month 7+ must not move Q1 (path-count bias regression)")


def test_conditioned_actions_priced_at_override_level(at_standard):
    """P2.3: with scenario + confidence override active, the conditioned
    actions table must be priced at the override level (the same backdrop as
    the world KPIs), not the assessed level."""
    at = at_standard
    _select(at, "Simulation mode").set_value("Quick (1,000)")
    _select(at, "Scenario (the world)").set_value("EMS Malaysia Disruption")
    _select(at, "Demand confidence").set_value("Very Low")
    at.run()
    assert not at.exception, at.exception

    # guard: the override must actually differ from the assessed level,
    # otherwise this test proves nothing
    override_texts = [c.value for c in at.caption] + [m.value for m in at.markdown]
    assert any("Override active" in t for t in override_texts), (
        "expected the sidebar override badge — is the assessed level "
        "already Very Low?")

    # recompute one action's incremental EV independently on the override
    # backdrop; the displayed value must match exactly (same sims, same seed)
    from src.baseline_plan import run_baseline
    from src.config import load_config
    from src.data_generator import load_or_generate
    from src.market_intelligence import (CONFIDENCE_SIM_PARAMS,
                                         merge_confidence_params)
    from src.scenarios import (compare_scenarios, kpi_summary,
                               management_actions, prebuilt_scenarios)
    from src.simulation import run_simulation

    cfg = load_config()
    data = load_or_generate(seed=cfg.random_seed)
    baseline = run_baseline(data, cfg)
    scen = prebuilt_scenarios()["EMS Malaysia Disruption"]
    action_name = "Authorize overtime at EMS sites"
    spec = management_actions()[action_name]
    conf = dict(CONFIDENCE_SIM_PARAMS["Very Low"])

    ref = run_simulation(
        data, cfg, baseline,
        params=merge_confidence_params(conf, scen.overrides),
        n_sims=1000, seed=cfg.random_seed,
        scenario_name="Action-pricing reference")
    act = run_simulation(
        data, cfg, baseline,
        params=merge_confidence_params(
            conf, merge_confidence_params(scen.overrides, spec.overrides)),
        n_sims=1000, seed=cfg.random_seed, scenario_name=action_name)
    expected = compare_scenarios(kpi_summary(ref, baseline, cfg),
                                 kpi_summary(act, baseline, cfg),
                                 spec.action_cost_usd)["incremental_ev"] / 1e6

    table = _actions_table(at)
    shown = float(table.loc[action_name, "Incremental EV ($M)"])
    assert shown == pytest.approx(expected, abs=1e-6), (
        f"conditioned EV for '{action_name}' is {shown:+.3f} $M but the "
        f"override-level backdrop gives {expected:+.3f} $M — conditioned "
        f"pricing is not running at the effective confidence level")
