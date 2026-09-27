"""Mechanisms that are zero in the neutralized world but still separate the engines.

1. Expected-value events the user's zero-shock setup had to switch off by hand
   (cancellations, push/pull timing, component and site disruptions,
   acceptance slip): the baseline has none of them.
2. Installation capacity: carried in PlanningArrays and decremented by the
   baseline (baseline_plan.py L57, L129) but enforced by neither engine.
3. Clip bias of the labor multipliers under the default ems_labor_sigma.
"""
from __future__ import annotations

import numpy as np

from harness import ZERO_PARAMS, fy_metrics_baseline, fy_metrics_sim, load_data, zero_shock
from src.baseline_plan import run_baseline
from src.config import load_config
from src.operations import build_planning_arrays
from src.simulation import _lognormal_mult

from greedy import baseline_financials, greedy_allocate
from sim_variant import run_variant


def main() -> None:
    raw, cfg_full = load_data(), load_config()
    data, cfg = zero_shock(raw, cfg_full)
    ref = fy_metrics_sim(run_variant(data, cfg, params=ZERO_PARAMS, n_sims=200))["rev"]
    print(f"Sim, zero sigma, all events off: FY rev {ref:.1f} $M\n")
    print("| event re-enabled at its default (sigmas still 0) | FY rev delta $M |")
    print("|---|---|")
    events = {
        "cancellations (cancel_prob_mult=1)": ({"cancel_prob_mult": 1.0}, False),
        "push-outs (pushout_prob_add=0)": ({"pushout_prob_add": 0.0}, False),
        "pull-ins (pullin_prob_add=0)": ({"pullin_prob_add": 0.0}, False),
        "component disruptions (comp_disrupt_mult=1)": ({"comp_disrupt_mult": 1.0}, False),
        "acceptance slip (acceptance_delay_add=0)": ({"acceptance_delay_add": 0.0}, False),
        "EMS regional site disruption (data default)": ({}, True),
    }
    all_prm = dict(ZERO_PARAMS)
    for label, (prm, site) in events.items():
        d = raw if site else data
        if site:
            _, c = zero_shock(raw, cfg_full)
        else:
            c = cfg
        v = fy_metrics_sim(run_variant(d, c, params=dict(ZERO_PARAMS, **prm), n_sims=2000))["rev"]
        all_prm.update(prm)
        print(f"| {label} | {v - ref:+.1f} |")
    _, c = zero_shock(raw, cfg_full)
    v = fy_metrics_sim(run_variant(raw, c, params=all_prm, n_sims=2000))["rev"]
    print(f"| all of the above together | {v - ref:+.1f} |")

    pa = build_planning_arrays(data)
    b = fy_metrics_baseline(run_baseline(data, cfg))
    capped = np.minimum(pa.integration_capacity, pa.installation_capacity)
    g = greedy_allocate(data, pa, integ_cap=capped)
    fin = baseline_financials(data, cfg, pa, g["built"], g["comp_used"])
    print(f"\nInstallation capacity {pa.installation_capacity[0]:.0f}/mo vs integration "
          f"{pa.integration_capacity.mean():.0f}/mo. Baseline with installation enforced: FY rev "
          f"{fin['revenue'][:12].sum() / 1e6:.1f} vs {b['rev']:.1f} $M "
          f"({fin['revenue'][:12].sum() / 1e6 - b['rev']:+.1f}); FY built {g['built'][:12].sum():.0f} vs {b['shipped']:.0f}")

    z = np.random.default_rng(0).standard_normal(1_000_000)
    lm = _lognormal_mult(z, cfg_full.uncertainty.ems_labor_sigma)
    print(f"\nLabor multiplier clip bias at sigma={cfg_full.uncertainty.ems_labor_sigma}: "
          f"EMS clip[0.7,1.1] mean {np.clip(lm, 0.7, 1.1).mean():.4f}; "
          f"integration clip[0.8,1.05] mean {np.clip(np.clip(lm, 0.7, 1.1), 0.8, 1.05).mean():.4f}")


if __name__ == "__main__":
    main()
