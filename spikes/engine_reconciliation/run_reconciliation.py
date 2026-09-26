"""Quantify each structural difference between run_simulation and run_baseline.

Zero-shock world (harness.zero_shock). Starts from the simulation as-is and
neutralizes one difference at a time (alone, cumulatively, and "last out":
everything else neutralized). Prints markdown tables.

Usage: python spikes/engine_reconciliation/run_reconciliation.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from harness import (ZERO_PARAMS, binding_probe, fy_metrics_baseline, fy_metrics_sim,
                     load_data, zero_shock)
from src.baseline_plan import run_baseline
from src.config import load_config
from src.operations import build_planning_arrays

from greedy import baseline_financials, greedy_allocate
from sim_variant import AS_IS, run_variant

N = 20  # zero-shock sims; residual variance comes only from unscaled factor shocks

# (key, label, neutralized flag values), in cumulative order
STEPS = [
    ("util_penalty", "Utilization adherence penalty (sim L264-272)", {"util_penalty": False}),
    ("factor_shocks", "Unscaled factor shocks at zero sigma (L157-158, L231)", {"factor_shocks": False}),
    ("lead_time_delay", "Base receipt delay (LT-8)/100 (L174)", {"lead_time_delay": False}),
    ("safety_floor", "30% safety-stock hard floor (L207)", {"safety_floor_frac": 0.0}),
    ("adherence", "Mean vs per-site adherence (L267 vs operations L171)", {"adherence": "per_site"}),
    ("allocator", "Proportional rationing vs greedy priority (L305-355)", {"allocator": "greedy"}),
    ("fpy_penalty", "FPY utilization penalty, cost only (L283-285)", {"fpy_util_penalty": False}),
    ("cogs", "COGS formula: rework/expedite (L415-439 vs baseline L170-175)", {"cogs": "baseline"}),
    ("inv_damping", "RM purchasing-response damping (L452-454)", {"inv_damping": False}),
    ("inv_floor", "RM valuation omits safety floor (L449)", {"inv_floor_excluded": False}),
    ("fg_val", "FG valued without rework/scrap (L460)", {"fg_valuation": "baseline"}),
]
METRICS = ["rev", "shipped", "gp", "gm", "inv"]


def md(df: pd.DataFrame, index: bool = False) -> str:
    """Minimal markdown table (no tabulate dependency)."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(out)


def main() -> None:
    t0 = time.time()
    data, cfg = zero_shock(load_data(), load_config())
    b = run_baseline(data, cfg)
    bm = fy_metrics_baseline(b)

    def run(flags=None, params=None, n=N):
        prm = dict(ZERO_PARAMS, **(params or {}))
        return run_variant(data, cfg, params=prm, n_sims=n, seed=42, flags=flags)

    def met(flags):
        # with factor shocks off every sim is identical: one sim suffices
        n = N if flags.get("factor_shocks", True) else 1
        return fy_metrics_sim(run(flags, n=n))

    as_is = met({})
    all_off = {}
    for _, _, f in STEPS:
        all_off.update(f)

    rows, cum_flags, prev = [], {}, as_is
    for key, label, f in STEPS:
        alone = met(f)
        cum_flags.update(f)
        cum = met(dict(cum_flags))
        last_out = met({k: v for k, v in all_off.items() if k not in f})
        full = met(all_off)
        r = {"key": key, "difference": label}
        for mname in METRICS:
            r[f"alone_{mname}"] = alone[mname] - as_is[mname]
            r[f"cum_{mname}"] = cum[mname] - prev[mname]
            r[f"lastout_{mname}"] = full[mname] - last_out[mname]
        r["cum_rev_level"] = cum["rev"]
        rows.append(r)
        prev = cum
    df = pd.DataFrame(rows)

    print("## Endpoints (FY = months 1-12, zero-shock world)\n")
    end = pd.DataFrame([dict(engine="baseline (src)", **bm),
                        dict(engine="simulation as-is", **as_is),
                        dict(engine="sim, all differences neutralized", **met(all_off))])
    print(md(end.round(3)))

    fmt = {"rev": "{:+.1f}", "shipped": "{:+.1f}", "gp": "{:+.1f}", "gm": "{:+.4f}", "inv": "{:+.1f}"}
    for mode, title in (("alone", "One at a time from sim as-is"),
                        ("cum", "Cumulative, in table order (step contribution)"),
                        ("lastout", "Last out: all else neutralized, then this one")):
        print(f"\n## {title} ($M, units, GM pts as fraction)\n")
        t = df[["difference"] + [f"{mode}_{m}" for m in METRICS]].copy()
        for m in METRICS:
            t[f"{mode}_{m}"] = t[f"{mode}_{m}"].map(fmt[m].format)
        t.columns = ["difference", "FY rev", "FY shipped", "FY GP", "FY GM", "End-FY inv"]
        print(md(t))

    # floor() rounding: a baseline-only artifact, measured inside the greedy engine
    pa = build_planning_arrays(data)
    g_nf = greedy_allocate(data, pa, use_floor=False)
    fin_nf = baseline_financials(data, cfg, pa, g_nf["built"], g_nf["comp_used"])
    nf_rev = fin_nf["revenue"][:12].sum() / 1e6
    print(f"\n## Baseline floor() rounding\n\nFY revenue without floors: {nf_rev:.1f} "
          f"vs {bm['rev']:.1f} (delta {nf_rev - bm['rev']:+.1f} $M); FY built "
          f"{g_nf['built'][:12].sum():.1f} vs {bm['shipped']:.0f}")

    # which constraint binds, at each cumulative stage
    print("\n## Binding constraint probe (FY revenue delta, $M, from +30% capacity)\n")
    probe_rows = []
    stage = {}
    for key, _, f in [("as_is", "", {})] + STEPS[:6]:
        stage.update(f)
        pr = binding_probe(lambda params, flags=None: run(flags, params, n=1),
                           {"params": {}, "flags": dict(stage)})
        m = fy_metrics_sim(run(dict(stage), n=1))
        probe_rows.append({"stage (cumulative through)": key, **{k: round(v, 1) for k, v in pr.items()},
                           "ems_util": round(m["ems_util"], 3), "integ_util": round(m["integ_util"], 3)})
    print(md(pd.DataFrame(probe_rows)))

    # baseline's own binding: probe by editing its inputs
    def base_rev(integ=1.0, ems=1.0):
        d2 = data.__class__(**{**data.__dict__})
        d2.integration_capacity = data.integration_capacity.copy()
        d2.integration_capacity["integration_capacity_units"] *= integ
        d2.ems_capacity = data.ems_capacity.copy()
        d2.ems_capacity["available_capacity_units"] *= ems
        return fy_metrics_baseline(run_baseline(d2, cfg))["rev"]
    br = base_rev()
    print(f"\nBaseline (src): integration x1.3 -> {base_rev(integ=1.3) - br:+.1f} $M; "
          f"EMS x1.3 -> {base_rev(ems=1.3) - br:+.1f} $M; "
          f"both x1.3 -> {base_rev(1.3, 1.3) - br:+.1f} $M")

    # constraint attribution audit on the baseline
    g = greedy_allocate(data, pa)
    c = g["constraints"]
    fy = c[c["month"] < 12]
    print("\n## Baseline constraint log audit (FY rows)\n")
    print(md(fy.groupby(["type", "physical"])["units_lost"].agg(["count", "sum"]).round(1), index=True))
    print(f"\nFY unmet at month 12 (true shortfall): {pa.demand_units[:12].sum() - g['built'][:12].sum():.1f} units;"
          f" sum of logged units_lost: {fy['units_lost'].sum():.1f} (carried units re-logged monthly)")

    # utilization penalty detail
    r = run({}, n=1)
    print("\n## Sim adherence_eff by month (as-is, zero shock)\n")
    print(" ".join(f"{x:.2f}" for x in r["adherence_eff"][0]))
    df.to_csv(__file__.replace("run_reconciliation.py", "results_zero_shock.csv"), index=False)
    print(f"\n(runtime {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
