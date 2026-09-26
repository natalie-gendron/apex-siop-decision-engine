"""One engine: the baseline supply plan IS the engine with zero shocks.

Guards the single-engine design (docs/engine-reconciliation.md). With
`Shocks.zero()` every path is identical, independent of the seed, and equal
to `run_baseline` on units, revenue, cost and inventory.
"""
from __future__ import annotations

import numpy as np
import pytest

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
    assert set(baseline.constraints["type"]) <= {"component", "ems_capacity", "integration"}
