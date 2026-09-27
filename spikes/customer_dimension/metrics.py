"""Spike: concentration and allocation metrics, and the cost-of-who-gets-shorted table.

Usage:  python spikes/customer_dimension/metrics.py [--paths 5000]

Runs every allocation policy on the same random numbers (common random
numbers, so policy deltas are paired and not sampling noise), in a base world
and in a stressed world (the tightest critical component loses 25% of receipts
from month 3). Prints markdown tables.

Metric definitions (per customer c, FY = months 1-12):
  plan_rev        FY plan-quantity x customer ASP (the demand plan at face value)
  E_rev / P10_rev mean / 10th percentile of FY recognized revenue
  RaR             plan_rev - P10_rev  (revenue at risk, demand + supply)
  supply_gap      E[(FY demand units - FY shipped units)+ x ASP], supply-driven
  P_short         P(FY fill rate < 95%), fill = shipped / realized demand units
  fill            mean FY fill rate
  delay_mo        late unit-months / realized demand units (average lateness)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from kernel import POLICIES, SimOptions, simulate  # noqa: E402
from problem import load_problem  # noqa: E402


def tightest_component(prob) -> int:
    dem_f = np.zeros((prob.firm_base.shape[0], len(prob.fam_names)))
    for li in range(prob.n_lines):
        dem_f[:, prob.line_fam[li]] += prob.firm_base[:, li] + prob.fcst_base[:, li]
    req = np.cumsum(dem_f @ prob.comp_usage.T, axis=0)
    sup = prob.comp_start[None, :] + np.cumsum(np.tile(prob.comp_po, (len(dem_f), 1)), axis=0)
    return int(np.argmin((sup / np.maximum(req, 1e-9)).min(axis=0)))


def by_customer(prob, r):
    """Aggregate line results to customers: dict of (n, N_cust) arrays."""
    Ncus = prob.n_cust
    onehot = np.eye(Ncus)[prob.line_cust]                               # (L, N_cust)
    ship_fy = r.ship_q[:, :4, :].sum(axis=1)                            # (n, L)
    dem_fy = r.dem_q[:, :4, :].sum(axis=1)
    gap_val = np.clip(dem_fy - ship_fy, 0, None) * prob.line_asp[None, :]
    return {
        "rev": r.rev_fy @ onehot,
        "gm": r.gm_fy @ onehot,
        "ship": ship_fy @ onehot,
        "dem": dem_fy @ onehot,
        "gap": gap_val @ onehot,
        "late": r.late_um @ onehot,
    }


def m(x):
    return x / 1e6


def policy_table(prob, results, top):
    base = results["proportional"]
    rev0, gm0 = base.rev_fy.sum(1), base.gm_fy.sum(1)
    rows = ["| policy | E[FY rev] $M | P10 FY rev $M | E[FY GM] $M | GM % | "
            "dGM vs proportional $M (P10, P90) | top-3 E[rev] $M | top-3 P(short) | "
            "customers with P(short) > 25% | late unit-months |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pol, r in results.items():
        c = by_customer(prob, r)
        rev, gm = r.rev_fy.sum(1), r.gm_fy.sum(1)
        d = gm - gm0
        fill = c["ship"] / np.clip(c["dem"], 1e-9, None)
        p_short = (fill < 0.95).mean(axis=0)
        top_fill = c["ship"][:, top].sum(1) / c["dem"][:, top].sum(1)
        rows.append(
            f"| {pol} | {m(rev.mean()):,.0f} | {m(np.percentile(rev, 10)):,.0f} | "
            f"{m(gm.mean()):,.0f} | {gm.mean() / rev.mean():.1%} | "
            f"{m(d.mean()):+,.1f} ({m(np.percentile(d, 10)):+,.1f}, {m(np.percentile(d, 90)):+,.1f}) | "
            f"{m(c['rev'][:, top].sum(1).mean()):,.0f} | {(top_fill < 0.95).mean():.0%} | "
            f"{int((p_short > 0.25).sum())} of {prob.n_cust} | {r.late_um.sum(1).mean():,.0f} |")
    return "\n".join(rows)


def customer_table(prob, results, pols):
    plan = prob.plan_revenue_by_cust()
    order = np.argsort(-plan)
    share = plan / plan.sum()
    head = "| customer | prio | plan share | plan FY $M |"
    sep = "|---|---:|---:|---:|"
    for p in pols:
        head += f" {p}: E rev $M | RaR $M | supply gap $M | P(short) | fill | delay mo |"
        sep += "---:|---:|---:|---:|---:|---:|"
    rows = [head, sep]
    cs = {p: by_customer(prob, results[p]) for p in pols}
    for ci in order:
        row = (f"| {prob.cust_names[ci]} | {prob.cust_priority[ci]} | {share[ci]:.0%} | "
               f"{m(plan[ci]):,.0f} |")
        for p in pols:
            c = cs[p]
            rev = c["rev"][:, ci]
            fill = c["ship"][:, ci] / np.clip(c["dem"][:, ci], 1e-9, None)
            delay = c["late"][:, ci].mean() / max(c["dem"][:, ci].mean(), 1e-9)
            row += (f" {m(rev.mean()):,.0f} | {m(plan[ci] - np.percentile(rev, 10)):,.0f} | "
                    f"{m(c['gap'][:, ci].mean()):,.1f} | {(fill < 0.95).mean():.0%} | "
                    f"{fill.mean():.1%} | {delay:.2f} |")
        rows.append(row)
    return "\n".join(rows)


def concentration_table(prob, r, ns=(1, 3, 5)):
    plan = prob.plan_revenue_by_cust()
    order = np.argsort(-plan)
    c = by_customer(prob, r)
    tot = c["rev"].sum(1)
    rows = ["| group | plan share | E[share of FY rev] | P10 group rev $M | "
            "RaR $M | P(group fill < 95%) |", "|---|---:|---:|---:|---:|---:|"]
    for n in ns:
        g = order[:n]
        grev = c["rev"][:, g].sum(1)
        gfill = c["ship"][:, g].sum(1) / c["dem"][:, g].sum(1)
        rows.append(f"| top-{n} | {plan[g].sum() / plan.sum():.0%} | {(grev / tot).mean():.0%} | "
                    f"{m(np.percentile(grev, 10)):,.0f} | "
                    f"{m(plan[g].sum() - np.percentile(grev, 10)):,.0f} | {(gfill < 0.95).mean():.0%} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=5000)
    args = ap.parse_args()
    prob, cfg = load_problem()
    top = np.argsort(-prob.plan_revenue_by_cust())[:3]
    k = tightest_component(prob)
    worlds = {
        "base": SimOptions(),
        f"stress (component {k} receipts x0.75 from month 3)": SimOptions(comp_stress={k: (2, 0.75)}),
    }
    for wname, opt in worlds.items():
        results = {p: simulate(prob, cfg, p, args.paths, seed=42, opt=opt) for p in POLICIES}
        print(f"\n## {wname}: cost of who gets shorted ({args.paths:,} paths, common random numbers)\n")
        print(policy_table(prob, results, top))
        print(f"\n### {wname}: by customer\n")
        print(customer_table(prob, results, ["proportional", "strict_greedy"]))
        print(f"\n### {wname}: concentration (strict_greedy)\n")
        print(concentration_table(prob, results["strict_greedy"]))

    print("\n## Customer correlation choice (proportional, base world)\n")
    print("| cust_corr | top-3 P10 rev $M | top-3 RaR $M | total P10 rev $M |")
    print("|---:|---:|---:|---:|")
    plan = prob.plan_revenue_by_cust()
    for rho in (0.0, 0.5, 0.9):
        r = simulate(prob, cfg, "proportional", args.paths, opt=SimOptions(cust_corr=rho))
        c = by_customer(prob, r)
        g = c["rev"][:, top].sum(1)
        print(f"| {rho} | {m(np.percentile(g, 10)):,.0f} | {m(plan[top].sum() - np.percentile(g, 10)):,.0f} | "
              f"{m(np.percentile(r.rev_fy.sum(1), 10)):,.0f} |")


if __name__ == "__main__":
    main()
