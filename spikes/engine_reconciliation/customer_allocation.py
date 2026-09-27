"""Who gets shorted: greedy customer priority vs proportional rationing.

Zero-shock world, every other difference neutralized, so the two runs differ
only in the allocator. Proportional rationing is family-level (the sim has no
customer dimension), so customers inside a family are shorted pro rata:
customer FY shipped = customer FY demand x family FY fill (approximation that
ignores carry timing). Revenue here is at each demand row's own ASP
(demand_plan.asp_usd) to show concentration; both engines actually value
units at the family-average ASP (operations.py L86, baseline_plan.py L168).
"""
from __future__ import annotations

import pandas as pd

from harness import ZERO_PARAMS, load_data, zero_shock
from src.config import load_config

from sim_variant import run_variant

ALIGNED = dict(util_penalty=False, factor_shocks=False, lead_time_delay=False,
               safety_floor_frac=0.0, adherence="per_site")


def main() -> None:
    data, cfg = zero_shock(load_data(), load_config())
    months = sorted(data.demand["month"].unique())[:12]
    dem = data.demand[data.demand["month"].isin(months)].copy()
    dem["units"] = dem["base_forecast_units"] + dem["backlog_units"]
    dem["rev_at_row_asp"] = dem["units"] * dem["asp_usd"]
    asp = dem.groupby(["customer", "product_family"]).apply(
        lambda g: (g["rev_at_row_asp"].sum() / max(g["units"].sum(), 1e-9)), include_groups=False)

    g = run_variant(data, cfg, params=ZERO_PARAMS, flags={**ALIGNED, "allocator": "greedy"})
    p = run_variant(data, cfg, params=ZERO_PARAMS, flags=ALIGNED)
    fams = list(data.products["product_family"])

    served = g["greedy"][0]["served"]
    served = served[served["month"] < 12]
    g_cf = served.groupby(["customer", "product_family"])["served"].sum()

    fam_dem = dem.groupby("product_family")["units"].sum()
    fam_ship_p = pd.Series(p["ship"][0, :12].sum(0), index=fams)
    fill_p = (fam_ship_p / fam_dem).clip(upper=1.0)
    cf = dem.groupby(["customer", "product_family", "customer_priority"])["units"].sum().reset_index()
    cf["greedy"] = [g_cf.get((c, f), 0.0) for c, f in zip(cf["customer"], cf["product_family"])]
    cf["proportional"] = cf["units"] * cf["product_family"].map(fill_p)
    cf["asp"] = [asp[(c, f)] for c, f in zip(cf["customer"], cf["product_family"])]

    by_cust = cf.groupby(["customer", "customer_priority"]).apply(lambda x: pd.Series({
        "demand_units": x["units"].sum(),
        "fill_greedy": x["greedy"].sum() / x["units"].sum(),
        "fill_proportional": x["proportional"].sum() / x["units"].sum(),
        "rev_greedy_$M": (x["greedy"] * x["asp"]).sum() / 1e6,
        "rev_prop_$M": (x["proportional"] * x["asp"]).sum() / 1e6,
    }), include_groups=False).reset_index().sort_values("rev_greedy_$M", ascending=False)
    by_cust["share_of_rev"] = by_cust["rev_greedy_$M"] / by_cust["rev_greedy_$M"].sum()
    print("## FY fill and revenue by customer (row ASP)\n")
    print(by_cust.round(3).to_string(index=False))

    tier = cf.groupby("customer_priority")[["units", "greedy", "proportional"]].sum()
    tier["fill_greedy"] = tier["greedy"] / tier["units"]
    tier["fill_proportional"] = tier["proportional"] / tier["units"]
    print("\n## FY fill by priority tier\n")
    print(tier.round(3).to_string())
    top3 = by_cust.head(3)
    print(f"\nTop-3 customers: {top3['share_of_rev'].sum():.0%} of FY revenue; "
          f"shortfall greedy {(top3['demand_units'] * (1 - top3['fill_greedy'])).sum():.1f} units vs "
          f"proportional {(top3['demand_units'] * (1 - top3['fill_proportional'])).sum():.1f} units")


if __name__ == "__main__":
    main()
