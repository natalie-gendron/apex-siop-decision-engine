"""How much of P(FY plan) is engine mismatch rather than uncertainty.

Full-uncertainty world (default config, default params), 3000 sims.
Compares P(FY revenue >= FY plan) for:
  - the sim as-is (what the app reports; app.py L873)
  - the sim with the utilization penalty removed (exact rerun)
  - the sim with its unscaled factor shocks removed (exact rerun)
  - the as-is distribution shifted by the zero-shock engine gap
    (baseline minus sim at zero shocks): a first-order estimate of the
    outlook if the sim reconciled to the baseline at zero shocks
Also reports the lever-value distortion: EMS x1.3 and integration x1.3.
"""
from __future__ import annotations

import numpy as np

from harness import ZERO_PARAMS, fy_metrics_baseline, fy_metrics_sim, load_data, zero_shock
from src.baseline_plan import run_baseline
from src.config import load_config

from sim_variant import run_variant

N = 3000
SITES = ("EMS Americas", "EMS Malaysia", "EMS Taiwan", "EMS Eastern Europe")


def main() -> None:
    data, cfg = load_data(), load_config()
    b = run_baseline(data, cfg)
    plan_fy = float(b.revenue_plan_q[:4].sum()) / 1e6
    base_fy = fy_metrics_baseline(b)["rev"]
    d0, c0 = zero_shock(data, cfg)
    gap0 = fy_metrics_baseline(run_baseline(d0, c0))["rev"] \
        - fy_metrics_sim(run_variant(d0, c0, params=ZERO_PARAMS))["rev"]
    print(f"FY revenue plan (financial_plan input): {plan_fy:.1f} $M; baseline FY revenue: {base_fy:.1f} $M")
    print(f"Zero-shock engine gap (baseline - sim): {gap0:.1f} $M\n")

    def fy_rev(flags=None, params=None):
        r = run_variant(data, cfg, params=params, n_sims=N, seed=42, flags=flags)
        return r["revenue"][:, :12].sum(1) / 1e6

    cases = {
        "sim as-is": fy_rev(),
        "no utilization penalty": fy_rev({"util_penalty": False}),
        "no unscaled factor shocks": fy_rev({"factor_shocks": False}),
    }
    asis = cases["sim as-is"]
    cases["as-is + zero-shock gap (level shift)"] = asis + gap0
    print("| case | mean FY rev $M | P10 | P90 | P(FY plan) |")
    print("|---|---|---|---|---|")
    for k, v in cases.items():
        print(f"| {k} | {v.mean():.0f} | {np.percentile(v, 10):.0f} | {np.percentile(v, 90):.0f} "
              f"| {(v >= plan_fy).mean():.1%} |")

    print("\nLever value (mean FY revenue delta, $M, full uncertainty):")
    ems = {"ems_capacity_mult": {s: 1.3 for s in SITES}}
    integ = {"integration_capacity_mult": 1.3}
    for label, flags in (("sim as-is", None), ("no utilization penalty", {"util_penalty": False})):
        base = fy_rev(flags)
        print(f"  {label}: EMS x1.3 {fy_rev(flags, ems).mean() - base.mean():+.1f}; "
              f"integration x1.3 {fy_rev(flags, integ).mean() - base.mean():+.1f}")


if __name__ == "__main__":
    main()
