"""Guard: the spike copies reproduce src exactly before any flag is flipped.

1. sim_variant.run_variant(flags=AS_IS) == src.simulation.run_simulation
   (full uncertainty and zero-shock world).
2. greedy.greedy_allocate + baseline_financials == src.baseline_plan.run_baseline.
"""
from __future__ import annotations

import numpy as np

from harness import ZERO_PARAMS, load_data, zero_shock
from src.baseline_plan import run_baseline
from src.config import load_config
from src.operations import build_planning_arrays
from src.simulation import run_simulation

from greedy import baseline_financials, greedy_allocate
from sim_variant import run_variant


def main() -> None:
    data = load_data()
    cfg = load_config()
    b = run_baseline(data, cfg)
    for label, (d, c, prm) in {
        "full uncertainty": (data, cfg, None),
        "zero shock": (*zero_shock(data, cfg), ZERO_PARAMS),
    }.items():
        ref = run_simulation(d, c, b, params=prm, n_sims=300, seed=7)
        var = run_variant(d, c, params=prm, n_sims=300, seed=7)
        for k in ("revenue", "cogs", "inventory"):
            np.testing.assert_allclose(var[k], getattr(ref, k), rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(var["ship"], ref.family_shipped, rtol=1e-12, atol=1e-9)
        print(f"sim_variant == src.simulation ({label}): OK")

    d0, c0 = zero_shock(data, cfg)
    b0 = run_baseline(d0, c0)
    pa = build_planning_arrays(d0)
    g = greedy_allocate(d0, pa)
    np.testing.assert_allclose(g["built"], b0.supply_units)
    fin = baseline_financials(d0, c0, pa, g["built"], g["comp_used"])
    np.testing.assert_allclose(fin["revenue"], b0.monthly["revenue_usd"])
    np.testing.assert_allclose(fin["cogs"], b0.monthly["cogs_usd"])
    np.testing.assert_allclose(fin["inventory"], b0.monthly["inventory_usd"])
    print("greedy copy == src.baseline_plan: OK")


if __name__ == "__main__":
    main()
