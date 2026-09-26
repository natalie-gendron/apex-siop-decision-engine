"""Customer dimension: outputs reconcile, and each allocation policy serves
who it says it serves (docs/design-customer-dimension.md)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.customers import concentration, customer_table, policy_comparison
from src.operations import build_planning_arrays
from src.shocks import Shocks
from src.simulation import run_simulation

N = 600
SEED = 7


@pytest.fixture(scope="module")
def pa(data):
    return build_planning_arrays(data)


@pytest.fixture(scope="module")
def stress(pa):
    """EMS at 85%: supply is short often enough that the policy decides who waits."""
    return {"ems_capacity_mult": {s: 0.85 for s in pa.site_names}}


@pytest.fixture(scope="module")
def runs(data, config, stress):
    return {pol: run_simulation(data, config, n_sims=N, seed=SEED,
                                params={**stress, "allocation_policy": pol})
            for pol in ("strict_priority", "priority_first", "proportional", "protect_top_n")}


def _fy_fill(r):
    dem = r.customer_demand[:, :12].sum(axis=1).astype(float)
    shp = r.customer_shipped[:, :12].sum(axis=1).astype(float)
    return (shp / dem).mean(axis=0)


def test_customer_outputs_reconcile(runs, data):
    r = runs["strict_priority"]
    assert r.customers == build_planning_arrays(data).customers
    np.testing.assert_allclose(r.customer_revenue.sum(axis=-1), r.revenue, rtol=1e-9)
    np.testing.assert_allclose(r.customer_shipped.sum(axis=-1), r.units_shipped, rtol=1e-5)
    np.testing.assert_allclose(r.customer_demand.sum(axis=-1), r.units_demanded, rtol=1e-5)


def _group_fill(r, mask):
    dem = r.customer_demand[:, :12].sum(axis=1)[:, mask].sum(axis=1)
    shp = r.customer_shipped[:, :12].sum(axis=1)[:, mask].sum(axis=1)
    return float((shp / dem).mean())


def test_strict_priority_backlog_first(runs, pa):
    """Backlog first, priority within: priority 1 as a group gains and the
    priority-3 customer absorbs the most. A priority-1 customer whose demand
    is mostly forecast can still wait behind others' booked orders."""
    strict, prop = runs["strict_priority"], runs["proportional"]
    p1, p3 = pa.cust_priority == 1, pa.cust_priority == 3
    assert _group_fill(strict, p1) > _group_fill(prop, p1)
    assert _group_fill(strict, p3) < _group_fill(prop, p3)
    assert _fy_fill(strict)[p3].min() == pytest.approx(_fy_fill(strict).min())


def test_priority_first_serves_priority_one_fully_first(runs, pa):
    """Priority first: every priority-1 customer does at least as well as
    under proportional or backlog-first rationing."""
    pf = _fy_fill(runs["priority_first"])
    p1 = pa.cust_priority == 1
    for other in ("proportional", "strict_priority"):
        assert (pf[p1] >= _fy_fill(runs[other])[p1] - 1e-9).all()
    assert pf[pa.cust_priority == 3].min() < _fy_fill(runs["strict_priority"])[pa.cust_priority == 3].min()


def test_protect_top_n_protects_the_top_customers(runs):
    prot, prop = _fy_fill(runs["protect_top_n"]), _fy_fill(runs["proportional"])
    assert (prot[:3] >= prop[:3] - 1e-9).all()
    assert prot[:3].min() > prot[3:].min()


def test_policy_moves_revenue_between_customers_not_totals(runs):
    """The policy is a redistribution: company FY revenue moves by far less
    than the revenue it moves between customers."""
    tot = {k: r.revenue[:, :12].sum(axis=1).mean() for k, r in runs.items()}
    cust = {k: r.customer_revenue[:, :12].sum(axis=1).mean(axis=0) for k, r in runs.items()}
    moved = np.abs(cust["strict_priority"] - cust["proportional"]).sum() / 2
    assert abs(tot["strict_priority"] - tot["proportional"]) < moved


def test_lost_after_n_months_loses_revenue(data, config, stress):
    """A customer whose shorted orders are lost immediately loses revenue;
    by default nothing is lost."""
    base = run_simulation(data, config, n_sims=N, seed=SEED, params=stress)
    victim = "Meridian Micro Devices"      # priority 3: shorted first under strict priority
    lose = run_simulation(data, config, n_sims=N, seed=SEED,
                          params={**stress, "lost_after_months": {victim: 0}})
    assert base.customer_lost_revenue.sum() == 0.0
    i = lose.customers.index(victim)
    assert lose.customer_lost_revenue[:, :, i].sum(axis=1).mean() > 0
    others = [j for j in range(len(lose.customers)) if j != i]
    assert lose.customer_lost_revenue[:, :, others].sum() == 0.0
    assert lose.customer_revenue[:, :12, i].sum(axis=1).mean() < \
        base.customer_revenue[:, :12, i].sum(axis=1).mean()


def test_longer_wait_loses_less(data, config, stress):
    victim = "Meridian Micro Devices"
    lost = [run_simulation(data, config, n_sims=N, seed=SEED,
                           params={**stress, "lost_after_months": {victim: k}}
                           ).customer_lost_revenue.sum(axis=(1, 2)).mean()
            for k in (0, 1, 3)]
    assert lost[0] > lost[1] > lost[2]


def test_unknown_policy_and_customer_are_rejected(data, config):
    with pytest.raises(ValueError):
        run_simulation(data, config, n_sims=2, params={"allocation_policy": "loudest"})
    with pytest.raises(KeyError):
        run_simulation(data, config, n_sims=2, params={"lost_after_months": {"Nobody Inc": 1}})


def test_baseline_uses_the_default_policy(data, config, baseline):
    z = run_simulation(data, config, n_sims=1, shocks=Shocks.zero())
    np.testing.assert_allclose(baseline.customer_revenue.to_numpy(),
                               z.customer_revenue[0], rtol=1e-12)


def test_customer_views_build(runs, baseline):
    r = runs["strict_priority"]
    t = customer_table(r, baseline)
    assert len(t) == len(r.customers)
    assert t["Share of FY revenue"].sum() == pytest.approx(1.0)
    c = concentration(r, baseline)
    assert list(c["Group"]) == ["Top 1", "Top 3", "Top 5"]
    assert c["Share of FY revenue"].is_monotonic_increasing
    cmp = policy_comparison(runs)
    assert cmp.loc[cmp["Policy"] == "proportional", "Δ gross profit vs reference"].item() == 0.0


def test_live_recompute_budget(data, config):
    """5,000 paths under the default policy stay inside the ~3 s budget."""
    t0 = time.perf_counter()
    run_simulation(data, config, n_sims=5000, seed=1)
    assert time.perf_counter() - t0 < 3.0
