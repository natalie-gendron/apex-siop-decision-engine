"""Need-conditional expediting: premium is paid to protect builds, not on
every late PO.

Regression tests for the 2026-08 audit finding: a volume-based premium on the
whole delayed pool priced the standing 50%-recovery policy as ~$28M/yr of
waste in the base world, made "Accept shipment risk" (stop expediting) free
money, and taxed every receipts-adding action (dual-source, long-lead
commits) with premium on its own incremental delayed pool.
"""
from __future__ import annotations

import pytest

from src.market_intelligence import CONFIDENCE_SIM_PARAMS, merge_confidence_params
from src.scenarios import kpi_summary, management_actions, prebuilt_scenarios
from src.simulation import run_simulation

N, SEED = 1000, 42


@pytest.fixture(scope="module")
def conf():
    return dict(CONFIDENCE_SIM_PARAMS["Moderate"])


def _kpi(data, config, baseline, params, name):
    r = run_simulation(data, config, baseline, params=params, n_sims=N,
                       seed=SEED, scenario_name=name)
    return kpi_summary(r, baseline, config)


def test_base_world_expedite_spend_is_need_sized(data, config, baseline, conf):
    """The standing policy buys protection where it's needed — it is not a
    ~$28M/yr volume habit."""
    base = _kpi(data, config, baseline, conf, "base")
    spend = base["expedite_cost"]["mean"]
    assert 0.0 < spend < 8e6, f"base expedite spend {spend / 1e6:.1f}M"


def test_stop_expediting_is_not_free_money(data, config, baseline, conf):
    """'Accept shipment risk' trades protection for savings — its base-world
    EV must be ~neutral, not a +$10M windfall."""
    catalog = management_actions()
    spec = catalog["Accept shipment risk (no extraordinary cost)"]
    ref = _kpi(data, config, baseline, conf, "ref")
    act = _kpi(data, config, baseline,
               merge_confidence_params(conf, spec.overrides), spec.name)
    ev = (act["fy_gross_profit"]["mean"] - ref["fy_gross_profit"]["mean"]
          - spec.action_cost_usd)
    assert ev < 2e6, f"stop-expediting EV {ev / 1e6:.1f}M reads as free money"


def test_receipt_adding_actions_not_taxed_with_premium(data, config, baseline,
                                                       conf):
    """Dual-sourcing adds receipts, which reduces need — its expedite spend
    must not exceed the no-action reference's."""
    catalog = management_actions()
    spec = catalog["Dual-source the high-end FPGA"]
    ref = _kpi(data, config, baseline, conf, "ref")
    act = _kpi(data, config, baseline,
               merge_confidence_params(conf, spec.overrides), spec.name)
    assert (act["expedite_cost"]["mean"]
            <= ref["expedite_cost"]["mean"] * 1.05)


def test_expediting_pays_in_the_world_it_exists_for(data, config, baseline,
                                                    conf):
    """Under the FPGA shortage, expediting must price better than standing
    down — the ordering that makes the recommendation engine credible."""
    catalog = management_actions()
    shortage = prebuilt_scenarios()["Critical FPGA Shortage"].overrides
    ref = _kpi(data, config, baseline,
               merge_confidence_params(conf, shortage), "ref-shortage")

    def ev(name):
        spec = catalog[name]
        stacked = merge_confidence_params(shortage, spec.overrides)
        k = _kpi(data, config, baseline,
                 merge_confidence_params(conf, stacked), name)
        return (k["fy_gross_profit"]["mean"] - ref["fy_gross_profit"]["mean"]
                - spec.action_cost_usd)

    assert ev("Expedite critical component receipts") > \
        ev("Accept shipment risk (no extraordinary cost)")
    # shortage-world expedite spend must exceed the base world's (need-driven)
    base_ref = _kpi(data, config, baseline, conf, "ref-base")
    assert (ref["expedite_cost"]["mean"]
            > base_ref["expedite_cost"]["mean"] * 2)
