"""One engine: the baseline supply plan IS the engine with zero shocks.

Guards the single-engine design (docs/engine-reconciliation.md). With
`Shocks.zero()` every path is identical, independent of the seed, and equal
to `run_baseline` on units, revenue, cost and inventory.
"""
from __future__ import annotations

import numpy as np
import pytest

import dataclasses

from src.baseline_plan import run_baseline
from src.operations import build_planning_arrays, effective_site_capacity
from src.shocks import Shocks
from src.simulation import run_simulation


@pytest.fixture(scope="module")
def zero(data, config):
    return run_simulation(data, config, n_sims=3, seed=11, shocks=Shocks.zero())


def test_zero_shocks_have_no_spread(zero, data, config):
    """Zero really means zero: no draw moves any output, across paths or seeds."""
    for arr in (zero.revenue, zero.cogs, zero.inventory, zero.units_shipped,
                zero.eo_reserve, zero.cash_flow):
        assert np.ptp(arr, axis=0).max() == 0.0
    other = run_simulation(data, config, n_sims=1, seed=999, shocks=Shocks.zero())
    np.testing.assert_array_equal(other.revenue[0], zero.revenue[0])
    np.testing.assert_array_equal(other.inventory[0], zero.inventory[0])


def test_zero_shock_demand_is_the_demand_plan(zero, baseline):
    np.testing.assert_allclose(zero.family_demand[0], baseline.demand_units, rtol=1e-12)


def test_zero_shock_engine_equals_baseline(zero, baseline):
    np.testing.assert_allclose(zero.family_shipped[0], baseline.supply_units, atol=1e-9)
    for arr, col in [(zero.revenue, "revenue_usd"), (zero.cogs, "cogs_usd"),
                     (zero.inventory, "inventory_usd"),
                     (zero.cash_flow, "cash_flow_usd")]:
        np.testing.assert_allclose(arr[0], baseline.monthly[col], rtol=1e-9)


def test_constraint_log_counts_each_unit_once(baseline):
    """Every unit that misses its requested month is logged once: the log
    total equals demand minus on-time shipments (FIFO within family)."""
    dem = baseline.demand_units
    cum_dem = np.cumsum(dem, axis=0)
    cum_ship = np.cumsum(baseline.supply_units, axis=0)
    prev = np.vstack([np.zeros((1, dem.shape[1])), cum_dem[:-1]])
    first_miss = np.clip(cum_dem - np.maximum(cum_ship, prev), 0, None)
    logged = baseline.constraints["units_lost"].sum()
    assert logged == pytest.approx(first_miss.sum(), abs=0.1 * len(baseline.constraints) + 1)
    assert set(baseline.constraints["type"]) <= {"component", "ems_capacity"}


def test_zero_shock_capacity_is_the_stated_capacity(zero, data):
    """At zero shocks EMS capacity is exactly available x adherence x labor:
    no leftover mean-one lognormal drag from a sigma whose shock is off."""
    np.testing.assert_allclose(zero.site_capacity[0],
                               effective_site_capacity(build_planning_arrays(data)),
                               rtol=1e-12)


def test_backlog_aging_drops_lost_orders(data, config):
    """With lost-after limits, the aging queue matches the engine's unmet
    backlog and holds nothing older than the limit."""
    cfg = config.model_copy(deep=True)
    cfg.customers.lost_after_months = {c: 1 for c in data.demand["customer"].unique()}
    cap = data.ems_capacity.copy()
    cap["available_capacity_units"] *= 0.8
    b = run_baseline(dataclasses.replace(data, ems_capacity=cap), cfg)
    aging = b.backlog_aging
    past_due = aging[aging["age_months"] >= 1].groupby("month")["units"].sum()
    unmet = b.unmet.sum(axis=1)
    for m_prev, m in zip(b.monthly["month"][:-1], b.monthly["month"][1:]):
        assert past_due.get(m, 0.0) == pytest.approx(unmet[m_prev], abs=1e-6)
    assert aging["age_months"].max() <= 1
