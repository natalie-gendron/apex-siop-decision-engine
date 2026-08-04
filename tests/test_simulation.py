"""Monte Carlo engine: reproducibility, bounds, correlation behavior."""
from __future__ import annotations

import numpy as np

from src.correlations import FactorEngine
from src.simulation import fiscal_year, quarterly, run_simulation
from src.validation import check_psd


def test_simulation_reproducible(data, config, baseline):
    a = run_simulation(data, config, baseline, n_sims=500, seed=99)
    b = run_simulation(data, config, baseline, n_sims=500, seed=99)
    np.testing.assert_array_equal(a.revenue, b.revenue)
    np.testing.assert_array_equal(a.inventory, b.inventory)


def test_different_seed_changes_results(data, config, baseline):
    a = run_simulation(data, config, baseline, n_sims=500, seed=1)
    b = run_simulation(data, config, baseline, n_sims=500, seed=2)
    assert not np.array_equal(a.revenue, b.revenue)


def test_outputs_finite_and_bounded(base_result):
    assert np.isfinite(base_result.revenue).all()
    assert (base_result.revenue >= 0).all()
    assert (base_result.inventory >= 0).all()
    gm = base_result.gross_margin
    assert (gm <= 1.0).all() and (gm >= -1.0).all()
    assert (base_result.ems_utilization >= 0).all()
    assert (base_result.expedite_cost >= 0).all()


def test_percentiles_ordered(base_result):
    from src.utils import percentile_stats
    for arr in [fiscal_year(base_result.revenue),
                base_result.inventory[:, 11],
                fiscal_year(base_result.gross_profit)]:
        s = percentile_stats(arr)
        assert s["p5"] <= s["p25"] <= s["median"] <= s["p75"] <= s["p95"]


def test_scenario_changes_outputs(data, config, baseline, base_result):
    from src.scenarios import prebuilt_scenarios
    scen = prebuilt_scenarios()["Critical FPGA Shortage"]
    r = run_simulation(data, config, baseline, params=scen.overrides,
                       n_sims=base_result.n_sims, seed=base_result.seed)
    base_fy = fiscal_year(base_result.revenue).mean()
    scen_fy = fiscal_year(r.revenue).mean()
    assert scen_fy < base_fy  # shortage must reduce expected revenue
    assert (fiscal_year(r.expedite_cost).mean()
            > fiscal_year(base_result.expedite_cost).mean())


def test_ai_surge_raises_demand(data, config, baseline, base_result):
    from src.scenarios import prebuilt_scenarios
    scen = prebuilt_scenarios()["AI Demand Surge"]
    r = run_simulation(data, config, baseline, params=scen.overrides,
                       n_sims=base_result.n_sims, seed=base_result.seed)
    assert (fiscal_year(r.units_demanded).mean()
            > fiscal_year(base_result.units_demanded).mean())


def test_correlations_produce_comovement(base_result):
    """AI factor must correlate positively with Zenith-family revenue."""
    from scipy.stats import spearmanr
    eagle_rev = base_result.family_revenue[:, :12, 0].sum(axis=1)
    rho, _ = spearmanr(base_result.drivers["AI / HPC demand factor"], eagle_rev)
    assert rho > 0.1


def test_factor_correlation_matrix_psd(config):
    engine = FactorEngine(config.factors)
    corr = engine.implied_correlation().to_numpy()
    _, valid = check_psd(corr, repair=False)
    assert valid


def test_quarterly_aggregation(base_result):
    q = quarterly(base_result.revenue)
    assert q.shape == (base_result.n_sims, 6)
    np.testing.assert_allclose(q.sum(axis=1), base_result.revenue.sum(axis=1),
                               rtol=1e-9)


def test_forced_pushout_moves_revenue(data, config, baseline, base_result):
    from src.scenarios import prebuilt_scenarios
    scen = prebuilt_scenarios()["Major Customer Push-Out"]
    r = run_simulation(data, config, baseline, params=scen.overrides,
                       n_sims=base_result.n_sims, seed=base_result.seed)
    base_q = quarterly(base_result.revenue).mean(axis=0)
    scen_q = quarterly(r.revenue).mean(axis=0)
    assert scen_q[0] < base_q[0]                    # Q1 loses revenue
    # the pushed revenue reappears in later quarters (congestion can defer it
    # beyond Q2), and the horizon total is roughly preserved, not destroyed
    assert scen_q[1:].sum() > base_q[1:].sum()
    assert abs(scen_q.sum() - base_q.sum()) / base_q.sum() < 0.01
