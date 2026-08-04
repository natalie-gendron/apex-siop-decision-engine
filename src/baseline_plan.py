"""Deterministic monthly baseline plan.

Allocation heuristic (documented, Version 1):
  1. Firm backlog is served before forecast demand.
  2. Higher customer priority first (1 before 2 before 3).
  3. Earlier requested shipment month first.
  4. Higher contribution margin per standard-equivalent unit breaks remaining ties.
  5. Builds go only to qualified EMS sites, cheapest conversion cost first.
  6. Component availability (on-hand + cumulative PO receipts, above safety stock),
     EMS capacity (adherence- and labor-derated), and final-integration capacity
     are all respected; unmet demand rolls into the next month and ages.

Limitations: the heuristic is greedy month-by-month — it does not pre-build ahead
of demand, trade off future months, or split lots optimally. A mixed-integer
optimizer could improve allocation quality but is intentionally out of scope for
Version 1 (explainability over optimality).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AppConfig
from .models import BaselineResult, InputData
from .operations import PlanningArrays, build_planning_arrays, effective_site_capacity
from .utils import N_MONTHS, PRODUCT_FAMILIES, quarter_of_month


def _sorted_demand_queue(data: InputData) -> pd.DataFrame:
    """Demand rows expanded to (month, family, customer) with heuristic sort keys."""
    dem = data.demand.copy()
    prod = data.products.set_index("product_family")
    dem["contribution_per_std"] = dem.apply(
        lambda r: (r["asp_usd"] - (
            prod.loc[r["product_family"], "material_cost_usd"]
            + prod.loc[r["product_family"], "ems_conversion_cost_usd"]
            + prod.loc[r["product_family"], "integration_test_cost_usd"]
            + prod.loc[r["product_family"], "freight_cost_usd"]
        )) / prod.loc[r["product_family"], "config_complexity"], axis=1)
    return dem


def run_baseline(data: InputData, config: AppConfig) -> BaselineResult:
    """Build the deterministic 18-month baseline plan."""
    pa = build_planning_arrays(data)
    fams = pa.families
    n_f = len(fams)
    fam_ix = {f: i for i, f in enumerate(fams)}
    sites = pa.site_names
    n_s = len(sites)

    dem = _sorted_demand_queue(data)
    site_cap = effective_site_capacity(pa)          # (n_sites, n_months)
    site_remaining = site_cap.copy()
    comp_stock = pa.comp_on_hand.copy()             # above zero; safety stock is a floor target
    integ_remaining = pa.integration_capacity.copy()
    install_remaining = pa.installation_capacity.copy()

    built = np.zeros((N_MONTHS, n_f))
    site_load = np.zeros((n_s, N_MONTHS))
    comp_used = np.zeros((len(pa.comp_names), N_MONTHS))
    unmet = np.zeros((N_MONTHS, n_f))
    constraint_rows: list[dict] = []
    backlog_age_rows: list[dict] = []
    carry: list[dict] = []                          # unmet demand carried forward

    for m in range(N_MONTHS):
        month = pa.months[m]
        month_rows = dem[dem["month"] == month].copy()
        month_rows["carry_age"] = 0
        queue = pd.concat([pd.DataFrame(carry), month_rows]) if carry else month_rows
        carry = []

        # Heuristic sort: backlog first, then priority, requested month, margin
        queue = queue.sort_values(
            by=["backlog_units", "customer_priority", "requested_month", "contribution_per_std"],
            ascending=[False, True, True, False],
        )

        # age tracking for backlog reporting
        for age, g in queue.groupby("carry_age"):
            units = (g["base_forecast_units"] + g["backlog_units"]).sum()
            if units > 0:
                backlog_age_rows.append({"month": month, "age_months": int(age), "units": float(units)})

        comp_receipts = pa.comp_po_monthly
        comp_stock = comp_stock + comp_receipts

        for _, row in queue.iterrows():
            f = fam_ix[row["product_family"]]
            want = float(row["base_forecast_units"] + row["backlog_units"])
            if want <= 0:
                continue

            # component ceiling for this family
            usage = pa.comp_usage[:, f]
            with np.errstate(divide="ignore", invalid="ignore"):
                comp_ceiling = np.where(usage > 0, comp_stock / usage, np.inf)
            comp_max = float(np.floor(comp_ceiling.min()))
            binding_comp = int(np.argmin(comp_ceiling)) if np.isfinite(comp_ceiling.min()) else -1

            # integration ceiling
            integ_max = float(np.floor(integ_remaining[m]))

            take = min(want, comp_max, integ_max)

            # EMS allocation: prefer the least-contested qualified site (fewest
            # other families qualified there), then lowest conversion cost. This
            # keeps flexible multi-family sites (e.g. EMS Taiwan) available for
            # families with no alternative.
            complexity = pa.family_complexity[f]
            contention = pa.site_qual.sum(axis=1)
            site_order = np.lexsort((pa.site_cost, contention))
            alloc_total = 0.0
            for s in site_order:
                if pa.site_qual[s, f] == 0 or take - alloc_total <= 0:
                    continue
                site_units = min(take - alloc_total,
                                 np.floor(site_remaining[s, m] / complexity))
                if site_units <= 0:
                    continue
                site_remaining[s, m] -= site_units * complexity
                site_load[s, m] += site_units * complexity
                alloc_total += site_units

            alloc_total = float(alloc_total)
            built[m, f] += alloc_total
            integ_remaining[m] -= alloc_total
            install_remaining[m] -= alloc_total
            comp_stock = comp_stock - usage * alloc_total
            comp_used[:, m] += usage * alloc_total

            shortfall = want - alloc_total
            if shortfall > 0.5:
                if alloc_total < min(want, integ_max) and comp_max <= min(want, integ_max):
                    ctype, detail = "component", pa.comp_names[binding_comp]
                elif integ_max < want:
                    ctype, detail = "integration", "Final integration capacity"
                else:
                    ctype, detail = "ems_capacity", "Qualified EMS capacity"
                constraint_rows.append({
                    "month": month, "type": ctype, "detail": detail,
                    "product_family": fams[f], "units_lost": round(shortfall, 1),
                })
                unmet[m, f] += shortfall
                nxt = row.copy()
                nxt["base_forecast_units"] = shortfall if row["backlog_units"] == 0 else 0.0
                nxt["backlog_units"] = shortfall if row["backlog_units"] > 0 else 0.0
                nxt["carry_age"] = int(row.get("carry_age", 0)) + 1
                if m < N_MONTHS - 1:
                    carry.append(nxt.to_dict())

    # ------------------------------------------------------------------
    # Financial translation of the deterministic plan
    # ------------------------------------------------------------------
    fin = config.financial
    rec_units = np.zeros((N_MONTHS, n_f))
    for f in range(n_f):
        lag = int(pa.family_rec_lag[f])
        if lag == 0:
            rec_units[:, f] = built[:, f]
        else:
            rec_units[lag:, f] = built[:-lag, f]
            # systems finished before the horizon start (in-transit pipeline)
            # recognize in the first lag month(s) at the steady-state build rate
            rec_units[:lag, f] = built[:lag, f].mean() if lag else 0.0

    revenue_fm = rec_units * pa.family_asp          # (n_months, n_fam)
    prod = data.products.set_index("product_family").loc[fams]
    unit_cogs = (prod["material_cost_usd"] + prod["ems_conversion_cost_usd"]
                 + prod["integration_test_cost_usd"] + prod["freight_cost_usd"]
                 + prod["warranty_reserve_usd"]
                 + prod["rework_prob"] * 0.5 * prod["ems_conversion_cost_usd"]
                 + prod["scrap_prob"] * prod["material_cost_usd"]).to_numpy(float)
    cogs_fm = rec_units * unit_cogs
    revenue_m = revenue_fm.sum(axis=1)
    cogs_m = cogs_fm.sum(axis=1)
    gp_m = revenue_m - cogs_m

    # inventory build-up. The component master covers critical items only; the
    # remaining (non-critical) raw material is held at ~0.9 months of material
    # consumption beyond the critical-component value (documented simplification).
    material_spend_m = (built * pa.family_material_cost).sum(axis=1)
    raw_val = np.zeros(N_MONTHS)
    comp_stock2 = pa.comp_on_hand.copy()
    for m in range(N_MONTHS):
        comp_stock2 = comp_stock2 + pa.comp_po_monthly - comp_used[:, m]
        crit_val = float((np.clip(comp_stock2, 0, None) * pa.comp_cost).sum())
        raw_val[m] = crit_val + 0.9 * material_spend_m[m]
    wip_val = (built * (pa.family_material_cost + prod["ems_conversion_cost_usd"].to_numpy(float))
               * prod["build_cycle_months"].to_numpy(float)[None, :] * 0.6).sum(axis=1)
    awaiting = built - rec_units                    # shipped/built, not yet recognized
    fg_val = np.clip(np.cumsum(awaiting, axis=0), 0, None) @ unit_cogs
    inventory_m = raw_val + wip_val + fg_val

    ar = revenue_m * fin.dso_days / 30.0
    ap = cogs_m * 0.75 * fin.dpo_days / 30.0
    wc_m = inventory_m + ar - ap
    oi_m = gp_m - fin.opex_monthly_usd
    ebitda_m = oi_m + fin.depreciation_monthly_usd
    dwc = np.diff(wc_m, prepend=wc_m[0])
    cash_m = ebitda_m - dwc - fin.capex_monthly_usd - np.clip(oi_m, 0, None) * fin.tax_rate

    monthly = pd.DataFrame({
        "month": pa.months,
        "quarter": [f"Q{quarter_of_month(m)}" for m in range(N_MONTHS)],
        "units_demand": pa.demand_units.sum(axis=1),
        "units_built": built.sum(axis=1),
        "units_recognized": rec_units.sum(axis=1),
        "units_unmet": unmet.sum(axis=1),
        "revenue_usd": revenue_m,
        "revenue_plan_usd": pa.revenue_plan_m,
        "cogs_usd": cogs_m,
        "gross_profit_usd": gp_m,
        "gross_margin": np.where(revenue_m > 0, gp_m / revenue_m, 0.0),
        "operating_income_usd": oi_m,
        "ebitda_usd": ebitda_m,
        "cash_flow_usd": cash_m,
        "raw_inventory_usd": raw_val,
        "wip_inventory_usd": wip_val,
        "fg_inventory_usd": fg_val,
        "inventory_usd": inventory_m,
        "working_capital_usd": wc_m,
        "ems_utilization": site_load.sum(axis=0) / np.clip(site_cap.sum(axis=0), 1e-9, None),
        "integration_utilization": built.sum(axis=1) / np.clip(pa.integration_capacity, 1e-9, None),
    })

    plan_q = np.array([pa.revenue_plan_m[q * 3:(q + 1) * 3].sum() for q in range(6)])

    comp_supply_cum = pa.comp_on_hand[:, None] + pa.comp_po_monthly[:, None] * np.arange(1, N_MONTHS + 1)[None, :]
    component_usage = pd.DataFrame({
        "component": np.repeat(pa.comp_names, N_MONTHS),
        "month": pa.months * len(pa.comp_names),
        "consumed_units": comp_used.flatten(),
        "cumulative_supply_units": comp_supply_cum.flatten(),
        "cumulative_consumed_units": np.cumsum(comp_used, axis=1).flatten(),
    })

    return BaselineResult(
        monthly=monthly,
        family_units=pd.DataFrame(rec_units, index=pa.months, columns=fams),
        family_revenue=pd.DataFrame(revenue_fm, index=pa.months, columns=fams),
        site_load=pd.DataFrame(site_load, index=sites, columns=pa.months),
        site_capacity=pd.DataFrame(site_cap, index=sites, columns=pa.months),
        integration_load=pd.DataFrame({
            "month": pa.months, "load_units": built.sum(axis=1),
            "capacity_units": pa.integration_capacity,
        }),
        component_usage=component_usage,
        constraints=pd.DataFrame(constraint_rows) if constraint_rows else pd.DataFrame(
            columns=["month", "type", "detail", "product_family", "units_lost"]),
        unmet=pd.DataFrame(unmet, index=pa.months, columns=fams),
        backlog_aging=pd.DataFrame(backlog_age_rows) if backlog_age_rows else pd.DataFrame(
            columns=["month", "age_months", "units"]),
        demand_units=pa.demand_units,
        supply_units=built,
        revenue_plan_q=plan_q,
        revenue_plan_m=pa.revenue_plan_m,
    )
