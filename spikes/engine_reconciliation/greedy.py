"""Spike copy of the baseline's greedy priority allocator and its financial translation.

Copied from src/baseline_plan.py::run_baseline (L43-202) and parametrized so the
same logic can run on capacity/supply arrays produced by the simulation engine.
Nothing in src/ is modified.

Differences from the original are opt-in only:
  - `use_floor`: the original floors component, integration and per-site
    capacity to whole units (baseline_plan.py L99, L103, L119).
  - `site_cap`, `integ_cap`, `comp_start`, `comp_receipts` replace the
    hard-wired inputs (effective_site_capacity, integration_capacity,
    comp_on_hand, comp_po_monthly).
With the default inputs and use_floor=True it reproduces run_baseline exactly
(checked in check_equivalence.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.baseline_plan import _sorted_demand_queue
from src.operations import PlanningArrays, effective_site_capacity
from src.utils import N_MONTHS


def greedy_allocate(data, pa: PlanningArrays, site_cap=None, integ_cap=None,
                    comp_start=None, comp_receipts=None, use_floor: bool = True,
                    demand_rows: pd.DataFrame | None = None) -> dict:
    """Run the baseline greedy customer-priority allocation.

    site_cap: (S, M) std-units; integ_cap: (M,); comp_start: (C,);
    comp_receipts: (M, C). Returns built (M, F), per-row allocations and the
    constraint log as the baseline writes it, plus slack arrays.
    """
    fl = np.floor if use_floor else (lambda x: x)
    fams = pa.families
    fam_ix = {f: i for i, f in enumerate(fams)}
    n_f, n_s = len(fams), len(pa.site_names)
    dem = _sorted_demand_queue(data) if demand_rows is None else demand_rows
    site_cap = effective_site_capacity(pa) if site_cap is None else site_cap
    integ_cap = pa.integration_capacity if integ_cap is None else integ_cap
    comp_stock = (pa.comp_on_hand if comp_start is None else comp_start).copy()
    if comp_receipts is None:
        comp_receipts = np.tile(pa.comp_po_monthly[None, :], (N_MONTHS, 1))

    site_remaining = site_cap.copy()
    integ_remaining = integ_cap.copy().astype(float)
    built = np.zeros((N_MONTHS, n_f))
    site_load = np.zeros((n_s, N_MONTHS))
    comp_used = np.zeros((len(pa.comp_names), N_MONTHS))
    served_rows: list[dict] = []
    constraint_rows: list[dict] = []
    carry: list[dict] = []

    contention = pa.site_qual.sum(axis=1)
    site_order = np.lexsort((pa.site_cost, contention))

    for m in range(N_MONTHS):
        month = pa.months[m]
        month_rows = dem[dem["month"] == month].copy()
        month_rows["carry_age"] = 0
        month_rows["orig_month"] = m
        queue = pd.concat([pd.DataFrame(carry), month_rows]) if carry else month_rows
        carry = []
        queue = queue.sort_values(
            by=["backlog_units", "customer_priority", "requested_month", "contribution_per_std"],
            ascending=[False, True, True, False])
        comp_stock = comp_stock + comp_receipts[m]

        for _, row in queue.iterrows():
            f = fam_ix[row["product_family"]]
            want = float(row["base_forecast_units"] + row["backlog_units"])
            if want <= 0:
                continue
            usage = pa.comp_usage[:, f]
            with np.errstate(divide="ignore", invalid="ignore"):
                comp_ceiling = np.where(usage > 0, comp_stock / usage, np.inf)
            comp_max = float(fl(comp_ceiling.min()))
            integ_max = float(fl(integ_remaining[m]))
            take = min(want, comp_max, integ_max)
            complexity = pa.family_complexity[f]
            alloc_total = 0.0
            for s in site_order:
                if pa.site_qual[s, f] == 0 or take - alloc_total <= 0:
                    continue
                site_units = min(take - alloc_total, fl(site_remaining[s, m] / complexity))
                if site_units <= 0:
                    continue
                site_remaining[s, m] -= site_units * complexity
                site_load[s, m] += site_units * complexity
                alloc_total += site_units
            alloc_total = float(alloc_total)
            built[m, f] += alloc_total
            integ_remaining[m] -= alloc_total
            comp_stock = comp_stock - usage * alloc_total
            comp_used[:, m] += usage * alloc_total
            served_rows.append({"month": m, "orig_month": int(row["orig_month"]),
                                "customer": row["customer"],
                                "customer_priority": int(row["customer_priority"]),
                                "product_family": row["product_family"],
                                "want": want, "served": alloc_total})
            shortfall = want - alloc_total
            if shortfall > 0.5:
                # attribution exactly as baseline_plan.py L135-140
                if alloc_total < min(want, integ_max) and comp_max <= min(want, integ_max):
                    ctype = "component"
                elif integ_max < want:
                    ctype = "integration"
                else:
                    ctype = "ems_capacity"
                # physical attribution: which ceiling actually stopped this row
                if comp_max <= alloc_total + 1e-9 and comp_max < want:
                    phys = "component"
                elif integ_max <= alloc_total + 1e-9 and integ_max < want:
                    phys = "integration"
                else:
                    phys = "ems_capacity"
                constraint_rows.append({"month": m, "type": ctype, "physical": phys,
                                        "product_family": fams[f], "units_lost": shortfall})
                nxt = row.copy()
                nxt["base_forecast_units"] = shortfall if row["backlog_units"] == 0 else 0.0
                nxt["backlog_units"] = shortfall if row["backlog_units"] > 0 else 0.0
                nxt["carry_age"] = int(row.get("carry_age", 0)) + 1
                if m < N_MONTHS - 1:
                    carry.append(nxt.to_dict())

    return {
        "built": built, "site_load": site_load, "comp_used": comp_used,
        "site_slack": site_remaining, "integ_slack": integ_remaining,
        "served": pd.DataFrame(served_rows),
        "constraints": pd.DataFrame(constraint_rows),
    }


def baseline_financials(data, config, pa: PlanningArrays, built: np.ndarray,
                        comp_used: np.ndarray | None = None,
                        comp_start=None, comp_receipts=None) -> dict:
    """Financial translation copied from baseline_plan.py L156-194."""
    fams = pa.families
    n_f = len(fams)
    rec_units = np.zeros((N_MONTHS, n_f))
    for f in range(n_f):
        lag = int(pa.family_rec_lag[f])
        if lag == 0:
            rec_units[:, f] = built[:, f]
        else:
            rec_units[lag:, f] = built[:-lag, f]
            rec_units[:lag, f] = built[:lag, f].mean()
    revenue_m = (rec_units * pa.family_asp).sum(axis=1)
    prod = data.products.set_index("product_family").loc[fams]
    unit_cogs = (prod["material_cost_usd"] + prod["ems_conversion_cost_usd"]
                 + prod["integration_test_cost_usd"] + prod["freight_cost_usd"]
                 + prod["warranty_reserve_usd"]
                 + prod["rework_prob"] * 0.5 * prod["ems_conversion_cost_usd"]
                 + prod["scrap_prob"] * prod["material_cost_usd"]).to_numpy(float)
    cogs_m = (rec_units * unit_cogs).sum(axis=1)
    if comp_used is None:
        comp_used = (built @ pa.comp_usage.T).T
    comp_start = pa.comp_on_hand if comp_start is None else comp_start
    if comp_receipts is None:
        comp_receipts = np.tile(pa.comp_po_monthly[None, :], (N_MONTHS, 1))
    material_spend_m = (built * pa.family_material_cost).sum(axis=1)
    raw_val = np.zeros(N_MONTHS)
    stock = comp_start.copy()
    for m in range(N_MONTHS):
        stock = stock + comp_receipts[m] - comp_used[:, m]
        raw_val[m] = float((np.clip(stock, 0, None) * pa.comp_cost).sum()) + 0.9 * material_spend_m[m]
    wip_val = (built * (pa.family_material_cost + prod["ems_conversion_cost_usd"].to_numpy(float))
               * prod["build_cycle_months"].to_numpy(float)[None, :] * 0.6).sum(axis=1)
    fg_val = np.clip(np.cumsum(built - rec_units, axis=0), 0, None) @ unit_cogs
    return {"revenue": revenue_m, "cogs": cogs_m, "rec": rec_units,
            "inventory": raw_val + wip_val + fg_val, "raw": raw_val,
            "wip": wip_val, "fg": fg_val}
