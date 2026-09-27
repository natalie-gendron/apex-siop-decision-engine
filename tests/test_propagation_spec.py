"""Executable propagation specification: what each lever SHOULD do.

One test per lever or mechanism, stating the expected direction of its effect
on operational outputs (shipments, service, shortages, inventory) and
financial outputs (revenue, COGS, operating income, AP, E&O).

Tests that pass today document behavior the engine already gets right. Tests
marked ``xfail(strict=True)`` document a gap: the reason names the code in
``src/simulation.py`` that causes it. Because they are strict, fixing the gap
turns the xfail into an XPASS failure, which forces the marker to be removed
and the spec to be updated in ``docs/propagation-spec.md``.

All runs share one seed (common random numbers): every stochastic draw in
``run_simulation`` has a shape independent of the lever values, so the
difference between two runs is the lever effect, not sampling noise.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.operations import build_planning_arrays
from src.scenarios import management_actions
from src.simulation import fiscal_year, run_simulation, service_level

N_SIMS = 600
SPEC_SEED = 7


@pytest.fixture(scope="module")
def pa(data):
    return build_planning_arrays(data)


@pytest.fixture(scope="module")
def sim(data, config, baseline):
    """Memoized runner: sim(key, **params) with a fixed seed and n_sims."""
    cache: dict[str, object] = {}

    def _run(key: str, _data=None, _config=None, **params):
        if key not in cache:
            cache[key] = run_simulation(
                _data if _data is not None else data,
                _config if _config is not None else config,
                baseline, params=params or None,
                n_sims=N_SIMS, seed=SPEC_SEED, scenario_name=key)
        return cache[key]

    return _run


def fy_mean(arr: np.ndarray) -> float:
    return float(fiscal_year(arr).mean())


def past_due_fy_end(r) -> float:
    """Cumulative demand not yet shipped at the end of the fiscal year."""
    return float((fiscal_year(r.units_demanded) - fiscal_year(r.units_shipped)).mean())


def implied_ap(r, config) -> np.ndarray:
    """AP recovered from WC = inventory + AR - AP, with AR = revenue * DSO / 30."""
    ar = r.revenue * config.financial.dso_days / 30.0
    return r.inventory + ar - r.working_capital


def _all(names, value):
    return {n: value for n in names}


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------

def test_demand_up_when_ems_binds_raises_past_due_and_lowers_service(sim, pa):
    """+20% demand with EMS capacity binding: shipments rise by less than
    demand, past-due grows, service falls. EMS is tightened 10% so the
    premise holds: in the base world EMS binds in only one FY month, since
    the separate integration stage (the old binding constraint) is folded
    into EMS capacity."""
    tight = _all(pa.site_names, 0.9)
    base = sim("ems_tight", ems_capacity_mult=tight)
    up = sim("ems_tight_demand_up", ems_capacity_mult=tight,
             demand_family_mult=_all(pa.families, 1.2))
    d_dem = fy_mean(up.units_demanded) - fy_mean(base.units_demanded)
    d_ship = fy_mean(up.units_shipped) - fy_mean(base.units_shipped)
    assert d_dem > 0
    assert d_ship < 0.5 * d_dem
    assert past_due_fy_end(up) > past_due_fy_end(base)
    assert service_level(up).mean() < service_level(base).mean() - 0.05
    assert up.capacity_shortfall_units.sum(axis=1).mean() > \
        base.capacity_shortfall_units.sum(axis=1).mean()


def test_demand_down_raises_component_inventory_exposure(sim, pa):
    """-20% demand: open POs inside the lead time keep arriving, so raw
    inventory (cash tied up) builds through the lead-time window and is
    higher on average over the year; revenue falls.

    Restated 2026-09-27 (build step 4, user decision): with buyers
    responding beyond the lead time, the excess is worked off by year end,
    so year-end E&O no longer rises with a demand drop. Lasting E&O from a
    drop belongs to obsolescence risk (step 5), not to this test."""
    base = sim("base")
    down = sim("demand_down", demand_family_mult=_all(pa.families, 0.8))
    assert down.raw_inventory[:, 2].mean() > base.raw_inventory[:, 2].mean()   # Q1 end
    assert down.raw_inventory[:, :12].mean() > base.raw_inventory[:, :12].mean()
    assert fy_mean(down.revenue) < fy_mean(base.revenue)


def test_price_changes_revenue_not_units(sim):
    """ASP +10%: revenue rises about 10%, units and COGS unchanged."""
    base = sim("base")
    price = sim("asp_up", asp_mult=1.1)
    np.testing.assert_allclose(price.units_shipped, base.units_shipped)
    np.testing.assert_allclose(price.cogs, base.cogs)
    ratio = fy_mean(price.revenue) / fy_mean(base.revenue)
    assert ratio == pytest.approx(1.10, abs=0.002)


# ---------------------------------------------------------------------------
# EMS capacity, overtime, yield
# ---------------------------------------------------------------------------

def test_relieving_ems_capacity_exposes_next_constraint(sim, pa):
    """Relieving EMS capacity (final integration and test included) helps,
    then plateaus: critical components become the constraint. With more
    demand on top, component shortage grows."""
    base = sim("base")
    ems15 = sim("ems_x1.5", ems_capacity_mult=_all(pa.site_names, 1.5))
    ems3 = sim("ems_x3", ems_capacity_mult=_all(pa.site_names, 3.0))
    assert fy_mean(ems15.units_shipped) > fy_mean(base.units_shipped)
    # plateau: doubling again adds (almost) nothing
    assert fy_mean(ems3.units_shipped) == pytest.approx(
        fy_mean(ems15.units_shipped), rel=0.005)
    assert ems3.ems_utilization[:, :12].mean() < 0.5
    # the plateau is set by components, not capacity (expected value: since
    # step 4 a rare path can release a component-starved backlog that briefly
    # exceeds even tripled capacity)
    assert ems3.capacity_shortfall_units.sum(axis=1).mean() < 0.1
    assert ems3.component_short_units[:, :12].sum(axis=1).mean() > 10
    assert service_level(ems3).mean() < 0.999

    both = sim("demand_up_all_capacity_relieved",
               demand_family_mult=_all(pa.families, 1.2),
               ems_capacity_mult=_all(pa.site_names, 3.0))
    # components, not capacity, bind: capacity shortfall is under 1% of
    # component shortfall (a rare path's released backlog can briefly exceed
    # even tripled capacity since purchasing responds, build step 4)
    comp_short = both.component_short_units[:, :12].sum(axis=1).mean()
    assert both.capacity_shortfall_units[:, :12].sum(axis=1).mean() < 0.01 * comp_short
    assert comp_short > 100


def test_overtime_raises_shipments_and_conversion_cost(sim, pa):
    """Overtime adds output when EMS binds and carries a conversion premium,
    so COGS per recognized unit rises."""
    base = sim("base")
    ot = sim("overtime_full", overtime_fraction=1.0)
    assert fy_mean(ot.units_shipped) > fy_mean(base.units_shipped)
    assert fy_mean(ot.revenue) > fy_mean(base.revenue)
    cpu_base = fy_mean(base.cogs) / fy_mean(base.family_units.sum(axis=2))
    cpu_ot = fy_mean(ot.cogs) / fy_mean(ot.family_units.sum(axis=2))
    assert cpu_ot > cpu_base


def test_lower_yield_reduces_shipments_when_capacity_binds(sim):
    """FPY -10 pts: rework consumes EMS capacity, so good output falls."""
    base = sim("base")
    low = sim("fpy_down", fpy_delta=-0.10)
    assert fiscal_year(low.rework_cost).mean() > fiscal_year(base.rework_cost).mean()
    assert fy_mean(low.units_shipped) < fy_mean(base.units_shipped) * 0.99


# ---------------------------------------------------------------------------
# Components: lead time, purchasing response, safety stock
# ---------------------------------------------------------------------------

def test_longer_lead_time_raises_shortage_under_demand_increase(sim, pa):
    """Components binding (EMS capacity relieved), demand +20%:
    a 50% longer lead time delays the supply response, so component
    shortage and lost shipments rise."""
    common = dict(demand_family_mult=_all(pa.families, 1.2),
                  ems_capacity_mult=_all(pa.site_names, 3.0))
    normal = sim("demand_up_all_capacity_relieved", **common)
    slow = sim("demand_up_relieved_lt_x1.5", lead_time_mult=1.5, **common)
    assert slow.component_short_units[:, :12].sum(axis=1).mean() > \
        normal.component_short_units[:, :12].sum(axis=1).mean()
    assert fy_mean(slow.units_shipped) < fy_mean(normal.units_shipped)


@pytest.mark.xfail(strict=True, reason=(
    "DEFERRED by the user 2026-09-27 as calibration. Purchases respond since "
    "build step 4 (planned orders beyond the lead time; shortage clears with "
    "no demand noise, and with supply shocks alone). Under lumpy demand the "
    "data's safety stock (~0.6 months) is too thin, so late-horizon shortage "
    "stays above months 5-7: a buffer-sizing question, not a missing mechanism"))
def test_component_purchases_respond_to_sustained_demand_beyond_lead_time(sim, pa):
    """Sustained +20% demand: once past the longest lead time (30 weeks,
    about 7 months) buyers have re-planned, so component shortage in months
    13-18 is no worse than in months 5-7."""
    r = sim("demand_up_all_capacity_relieved",
            demand_family_mult=_all(pa.families, 1.2),
            ems_capacity_mult=_all(pa.site_names, 3.0))
    early = r.component_short_units[:, 4:7].mean()
    late = r.component_short_units[:, 12:18].mean()
    assert late <= early


def test_higher_safety_stock_does_not_reduce_service(sim):
    base = sim("base")
    ss = sim("safety_stock_x2", safety_stock_mult=2.0)
    assert service_level(ss).mean() >= service_level(base).mean()


def test_higher_safety_stock_raises_average_raw_inventory(sim):
    base = sim("base")
    ss = sim("safety_stock_x2", safety_stock_mult=2.0)
    assert ss.raw_inventory[:, :12].mean() > base.raw_inventory[:, :12].mean()


# ---------------------------------------------------------------------------
# Financial mechanics: recurring action cost, AP, E&O
# ---------------------------------------------------------------------------

def test_permanent_capacity_action_carries_recurring_cost(sim, config):
    """Reserved EMS capacity (take-or-pay, online month 3) costs money every
    month it is held, not only in Q1."""
    spec = management_actions()["Reserve additional EMS capacity"]
    r = sim("reserve_capacity", action_cost_usd=spec.action_cost_usd,
            **spec.overrides)
    below_gp_cost = r.gross_profit - r.operating_income - config.financial.opex_monthly_usd
    assert (below_gp_cost[:, 3:].mean(axis=0) > 0).all()


def test_buying_more_components_raises_ap(sim, pa, config):
    """Buy more components (double the safety-stock target, a one-time buy of
    the extra buffer): payables rise by at least a quarter of the extra
    purchase value, averaged over the fiscal year, times DPO / 30.

    Lever changed 2026-09-27 (build step 5, user approval): since step 4
    comp_supply_mult scales supplier capacity and no longer buys anything."""
    base = sim("base")
    buy = sim("safety_stock_x2", safety_stock_mult=2.0)
    extra_purchases_m = float((pa.comp_safety * pa.comp_cost).sum()) / 12.0
    expected = 0.25 * extra_purchases_m * config.financial.dpo_days / 30.0
    d_ap = implied_ap(buy, config)[:, :12].mean() - implied_ap(base, config)[:, :12].mean()
    assert d_ap > expected


def test_eo_reflects_component_obsolescence_risk(sim, data):
    """Same excess stock, higher obsolescence risk: larger E&O reserve."""
    lo, hi = data.components.copy(), data.components.copy()
    lo["obsolescence_risk"] = 0.0
    hi["obsolescence_risk"] = 1.0
    r_lo = sim("obsolescence_low", _data=dataclasses.replace(data, components=lo))
    r_hi = sim("obsolescence_high", _data=dataclasses.replace(data, components=hi))
    assert r_hi.eo_reserve.mean() > r_lo.eo_reserve.mean()


# ---------------------------------------------------------------------------
# Structural: customer dimension, one engine
# ---------------------------------------------------------------------------

def test_simulation_reports_revenue_by_customer(sim, data):
    r = sim("base")
    by_cust = getattr(r, "customer_revenue")
    assert by_cust.shape[-1] == data.demand["customer"].nunique()
    np.testing.assert_allclose(by_cust.sum(axis=-1), r.revenue, rtol=1e-6)


def test_zero_shock_simulation_reconciles_to_baseline(sim, config, baseline):
    """One engine: with shocks switched off the simulation reproduces the
    deterministic baseline FY revenue within 1%."""
    cfg = config.model_copy(deep=True)
    u = cfg.uncertainty
    u.market_demand_sigma = {k: 0.0 for k in u.market_demand_sigma}
    for k in ("customer_idiosyncratic_sigma", "asp_sigma", "material_cost_sigma",
              "fx_cost_sigma", "conversion_cost_sigma", "freight_sigma",
              "ems_labor_sigma", "site_disruption_impact"):
        setattr(u, k, 0.0)
    r = sim("zero_shock", _config=cfg, cancel_prob_mult=0.0,
            pushout_prob_add=-1.0, pullin_prob_add=-1.0, comp_disrupt_mult=0.0)
    base_fy = float(baseline.monthly["revenue_usd"].iloc[:12].sum())
    assert fy_mean(r.revenue) == pytest.approx(base_fy, rel=0.01)


# ---------------------------------------------------------------------------
# Build step 6a: buy-ahead as a non-cancellable purchase; E&O hits the P&L
# ---------------------------------------------------------------------------

FPGA_BUY = {"buy_ahead": {"High-End FPGA": (0, 2.0)}}   # 2 months of cover, ordered now


def test_buy_ahead_cannot_arrive_before_lead_time(sim, pa):
    """An order placed in month 1 arrives one lead time later: nothing
    changes before then."""
    base = sim("base")
    buy = sim("fpga_buy_ahead", **FPGA_BUY)
    c = pa.comp_names.index("High-End FPGA")
    lt = int(np.ceil(pa.comp_lead_time[c] / 4.345))
    np.testing.assert_allclose(buy.raw_inventory[:, :lt], base.raw_inventory[:, :lt])
    assert buy.raw_inventory[:, lt].mean() > base.raw_inventory[:, lt].mean()


def test_non_cancellable_buy_ahead_is_exposed_when_demand_softens(sim, pa):
    """Demand -20%: the committed parts still arrive, so year-end stock and
    E&O are higher than without the commitment."""
    down = dict(demand_family_mult=_all(pa.families, 0.8))
    plain = sim("demand_down", **down)
    buy = sim("demand_down_fpga_buy", **down, **FPGA_BUY)
    assert buy.raw_inventory[:, 11].mean() > plain.raw_inventory[:, 11].mean()
    assert buy.eo_reserve.mean() > plain.eo_reserve.mean()


def test_buy_ahead_protects_shipments_in_a_shortage(sim):
    """FPGA shortage world: parts committed ahead of the cut protect
    shipments."""
    from src.scenarios import prebuilt_scenarios
    world = prebuilt_scenarios()["Critical FPGA Shortage"].overrides
    plain = sim("fpga_shortage", **world)
    buy = sim("fpga_shortage_buy", **world, **FPGA_BUY)
    assert fy_mean(buy.units_shipped) > fy_mean(plain.units_shipped)


def test_eo_provision_is_charged_to_cogs(sim, data):
    """E&O is a P&L charge: the FY provision (year-end reserve less the
    opening reserve) lowers FY gross profit by the same amount. Higher
    obsolescence risk changes nothing operational, only the reserve."""
    lo, hi = data.components.copy(), data.components.copy()
    lo["obsolescence_risk"] = 0.0
    hi["obsolescence_risk"] = 1.0
    r_lo = sim("obsolescence_low", _data=dataclasses.replace(data, components=lo))
    r_hi = sim("obsolescence_high", _data=dataclasses.replace(data, components=hi))
    np.testing.assert_allclose(r_hi.units_shipped, r_lo.units_shipped)
    assert r_hi.eo_reserve.mean() > r_lo.eo_reserve.mean()
    d_provision = r_hi.eo_provision.mean() - r_lo.eo_provision.mean()
    d_gp = fy_mean(r_hi.gross_profit) - fy_mean(r_lo.gross_profit)
    assert d_provision != 0
    assert d_gp == pytest.approx(-d_provision, rel=1e-6)


# ---------------------------------------------------------------------------
# Build step 6b: a second source is independent, not more of the same
# ---------------------------------------------------------------------------

DISRUPTED = {"comp_disrupt_mult": 4.0}
MORE_FPGA = {"comp_supply_ramp": {"High-End FPGA": (0, 1.5)}}
SECOND_SOURCE = {"dual_source": {"High-End FPGA": (0, 0.5)}}


def test_second_source_protects_against_supplier_disruption(sim):
    """With frequent supplier disruptions, half the FPGA volume on an
    independent source ships more than the same extra capacity from the one
    source."""
    same = sim("disrupted_more_fpga", **DISRUPTED, **MORE_FPGA)
    dual = sim("disrupted_dual_fpga", **DISRUPTED, **MORE_FPGA, **SECOND_SOURCE)
    assert fy_mean(dual.units_shipped) > fy_mean(same.units_shipped)
    assert dual.component_short_units[:, :12].sum(axis=1).mean() < \
        same.component_short_units[:, :12].sum(axis=1).mean()


def test_second_source_changes_nothing_without_disruptions(sim):
    """Independence is insurance: with no supplier disruptions, the split
    between sources changes nothing."""
    calm = {"comp_disrupt_mult": 0.0}
    one = sim("calm_more_fpga", **calm, **MORE_FPGA)
    two = sim("calm_dual_fpga", **calm, **MORE_FPGA, **SECOND_SOURCE)
    np.testing.assert_allclose(two.units_shipped, one.units_shipped)
    np.testing.assert_allclose(two.revenue, one.revenue)


# ---------------------------------------------------------------------------
# Build step 6c: customer-specific demand events (D7 upside, D9 push-out)
# ---------------------------------------------------------------------------

TITAN_ASK = {"customer_demand_edit": {"titan_ask": {
    "customer": "Titan Semiconductor", "family": "Zenith Compute Test",
    "month": 3, "units": 12}}}


def _cust(r, name):
    return r.customers.index(name)


def test_customer_upside_adds_demand_and_revenue_to_that_customer(sim):
    base = sim("base")
    ask = sim("titan_ask", **TITAN_ASK)
    i = _cust(base, "Titan Semiconductor")
    d_dem = ask.customer_demand[:, :, i].sum(axis=1).mean() - base.customer_demand[:, :, i].sum(axis=1).mean()
    assert d_dem == pytest.approx(12.0, rel=1e-6)
    others = [j for j in range(len(base.customers)) if j != i]
    np.testing.assert_allclose(ask.customer_demand[:, :, others], base.customer_demand[:, :, others])
    assert ask.customer_revenue[:, :12, i].sum(axis=1).mean() > \
        base.customer_revenue[:, :12, i].sum(axis=1).mean()


def test_customer_upside_displaces_lower_priority_customers(sim, pa):
    """EMS tight, strict priority: a priority-1 customer's upside is served
    ahead of priority-3 demand, which ships less."""
    tight = {"ems_capacity_mult": _all(pa.site_names, 0.85)}
    base = sim("ems_085", **tight)
    ask = sim("ems_085_titan_ask", **tight, **TITAN_ASK)
    j = _cust(base, "Meridian Micro Devices")                  # priority 3
    assert ask.customer_shipped[:, :12, j].sum(axis=1).mean() < \
        base.customer_shipped[:, :12, j].sum(axis=1).mean()


def test_customer_push_out_moves_that_customers_revenue_later(sim):
    """A named customer pushes Zenith systems from month 2 to month 5: it
    moves up to the units asked (a path whose month-2 order already slipped
    has less to move), so Q1 demand falls by at most that and the customer's
    horizon demand is unchanged."""
    push = {"customer_demand_edit": {"kestrel_push": {
        "customer": "Kestrel Compute", "family": "Zenith Compute Test",
        "month": 1, "units": 6, "to_month": 4}}}
    base = sim("base")
    moved = sim("kestrel_push", **push)
    i = _cust(base, "Kestrel Compute")
    q1 = lambda r: r.customer_demand[:, :3, i].sum(axis=1).mean()
    assert 0.0 < q1(base) - q1(moved) <= 6.0 + 1e-9
    assert moved.customer_demand[:, :, i].sum() == pytest.approx(base.customer_demand[:, :, i].sum())


def test_unknown_customer_edit_is_rejected(data, config):
    with pytest.raises(KeyError):
        run_simulation(data, config, n_sims=2, params={"customer_demand_edit": {
            "x": {"customer": "Nobody Inc", "month": 1, "units": 1}}})
