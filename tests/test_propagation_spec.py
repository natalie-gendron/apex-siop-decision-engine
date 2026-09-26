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


def test_demand_down_raises_component_inventory_and_eo(sim, pa):
    """-20% demand with flat receipts: raw inventory builds, E&O rises."""
    base = sim("base")
    down = sim("demand_down", demand_family_mult=_all(pa.families, 0.8))
    assert down.raw_inventory[:, :12].mean() > base.raw_inventory[:, :12].mean()
    assert down.eo_reserve.mean() > base.eo_reserve.mean()
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
    # the plateau is set by components, not capacity
    assert ems3.capacity_shortfall_units.sum() == pytest.approx(0.0, abs=1e-6)
    assert ems3.component_short_units[:, :12].sum(axis=1).mean() > 10
    assert service_level(ems3).mean() < 0.999

    both = sim("demand_up_all_capacity_relieved",
               demand_family_mult=_all(pa.families, 1.2),
               ems_capacity_mult=_all(pa.site_names, 3.0))
    assert both.capacity_shortfall_units.sum() == pytest.approx(0.0, abs=1e-6)
    assert both.component_short_units[:, :12].sum(axis=1).mean() > 100


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


@pytest.mark.xfail(strict=True, reason=(
    "src/simulation.py run_simulation: fpy_eff only feeds rework_cost "
    "(section 6, rework_units = ship * (1 - fpy_eff)); it never reduces "
    "site_cap or shipped units in the section 4 shipment loop"))
def test_lower_yield_reduces_shipments_when_capacity_binds(sim):
    """FPY -10 pts: rework consumes EMS capacity, so good output falls."""
    base = sim("base")
    low = sim("fpy_down", fpy_delta=-0.10)
    assert fiscal_year(low.rework_cost).mean() > fiscal_year(base.rework_cost).mean()
    assert fy_mean(low.units_shipped) < fy_mean(base.units_shipped) * 0.99


# ---------------------------------------------------------------------------
# Components: lead time, purchasing response, safety stock
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "src/simulation.py section 2: lead_time_mult only raises delay_frac, a "
    "one-month receipt slip (received[:, 1:] += delayed[:, :-1]); higher "
    "delay_frac also enlarges delayed_pool, so expediting recovers more and "
    "shortage FALLS. Lead time never gates when supply can respond"))
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
    "src/simulation.py section 2: receipts_nominal is np.tile of "
    "comp_po_monthly (open_po_units_per_month) for all 18 months; only "
    "comp_supply_mult/comp_supply_ramp change it, never realized demand"))
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


@pytest.mark.xfail(strict=True, reason=(
    "src/simulation.py section 2: safety_floor = comp_safety * "
    "safety_stock_mult * 0.3 is subtracted from usable supply (start_avail, "
    "comp_avail) without adding any receipts, so a higher policy removes "
    "supply and service falls slightly"))
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

@pytest.mark.xfail(strict=True, reason=(
    "src/simulation.py section 6: action_cost_m[:3] = action_cost_usd / 3; "
    "the only action cost is one-time in months 1-3. No recurring cost for "
    "permanent capacity (take-or-pay, headcount) after it takes effect"))
def test_permanent_capacity_action_carries_recurring_cost(sim, config):
    """Reserved EMS capacity (take-or-pay, online month 3) costs money every
    month it is held, not only in Q1."""
    spec = management_actions()["Reserve additional EMS capacity"]
    r = sim("reserve_capacity", action_cost_usd=spec.action_cost_usd,
            **spec.overrides)
    below_gp_cost = r.gross_profit - r.operating_income - config.financial.opex_monthly_usd
    assert (below_gp_cost[:, 3:].mean(axis=0) > 0).all()


@pytest.mark.xfail(strict=True, reason=(
    "src/simulation.py section 6: ap = cogs * 0.75 * dpo_days / 30 is tied "
    "to COGS of recognized units, not to component purchases (receipts)"))
def test_buying_more_components_raises_ap(sim, pa, config):
    """+50% receipts on every component: payables rise by at least a quarter
    of the extra monthly purchase value times DPO / 30."""
    base = sim("base")
    buy = sim("comp_supply_x1.5", comp_supply_mult={"__all__": 1.5})
    extra_purchases_m = 0.5 * float((pa.comp_po_monthly * pa.comp_cost).sum())
    expected = 0.25 * extra_purchases_m * config.financial.dpo_days / 30.0
    d_ap = implied_ap(buy, config)[:, :12].mean() - implied_ap(base, config)[:, :12].mean()
    assert d_ap > expected


@pytest.mark.xfail(strict=True, reason=(
    "src/simulation.py section 6: eo_reserve = excess_rm * comp_cost * "
    "eo_reserve_rate + 0.05 * fg; the components.obsolescence_risk column "
    "is never read by build_planning_arrays or run_simulation"))
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

@pytest.mark.xfail(strict=True, reason=(
    "src/simulation.py run_simulation aggregates demand to family x month "
    "(build_planning_arrays groupby month/family); SimulationResult has no "
    "customer dimension, so revenue concentration cannot be read"))
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
