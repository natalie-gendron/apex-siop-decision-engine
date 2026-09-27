"""Baseline supply plan: the engine run with zero shocks and one path.

One engine (see `docs/engine-reconciliation.md`): the baseline is not a second
allocator. It is `run_simulation(shocks=Shocks.zero(), n_sims=1)`, so the plan
of record and the Monte Carlo cannot disagree on allocation, cost or
inventory. This module only packages that single path into the
plan-of-record tables the views and the export read.

Reporting conventions:
  - Past-due demand ages first-in, first-out within a product family; orders
    lost after waiting (customers with a lost-after limit) leave the queue
    oldest first, as they do in the engine.
  - The constraint log records each unit once, in the month it first misses
    its requested month, against the ceiling that cut it that month
    (a named critical component or qualified EMS capacity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AppConfig
from .models import BaselineResult, InputData
from .operations import build_planning_arrays
from .shocks import Shocks
from .simulation import run_simulation
from .utils import N_MONTHS, quarter_of_month

_LIMITS = [("component", None), ("ems_capacity", "Qualified EMS capacity")]


def _fifo_outstanding(cum_dem: np.ndarray, shipped_to_date: float, o: int) -> float:
    """Units from origin month o still unserved after `shipped_to_date`
    cumulative shipments, serving the oldest demand first."""
    prev = cum_dem[o - 1] if o > 0 else 0.0
    return float(max(cum_dem[o] - max(shipped_to_date, prev), 0.0))


def run_baseline(data: InputData, config: AppConfig) -> BaselineResult:
    """The deterministic 18-month baseline supply plan (zero-shock engine,
    default allocation policy: strict customer priority)."""
    r = run_simulation(data, config, n_sims=1, seed=0, shocks=Shocks.zero(),
                       scenario_name="Baseline supply plan", keep_component_paths=True)
    pa = build_planning_arrays(data)
    fams, sites, months = pa.families, pa.site_names, pa.months

    demand = r.family_demand[0]                     # (M, F); equals the demand plan
    built = r.family_shipped[0]
    rec_units = r.family_units[0]
    unmet = r.family_backlog[0]
    site_load = r.site_load[0]                      # (S, M)
    site_cap = r.site_capacity[0]
    limits = r.limit_units[:, 0]                    # (3, M, F)

    backlog_age_rows: list[dict] = []
    constraint_rows: list[dict] = []
    cum_dem = np.cumsum(demand, axis=0)
    cum_ship = np.cumsum(built, axis=0)
    # lost at the end of month m: gone from the queue from month m + 1
    cum_lost = np.cumsum(r.family_lost[0], axis=0)
    for f, fam in enumerate(fams):
        for m in range(N_MONTHS):
            # demand queue at the start of month m, by age
            shipped_before = (cum_ship[m - 1, f] + cum_lost[m - 1, f]) if m > 0 else 0.0
            for o in range(m + 1):
                units = (demand[m, f] if o == m
                         else _fifo_outstanding(cum_dem[:, f], shipped_before, o))
                if units > 1e-9:
                    backlog_age_rows.append({"month": months[m], "age_months": m - o,
                                             "units": units})
            # units of month m's demand that miss month m: logged once, here
            first_miss = _fifo_outstanding(
                cum_dem[:, f], cum_ship[m, f] + (cum_lost[m - 1, f] if m > 0 else 0.0), m)
            cut = limits[:, m, f]
            if first_miss > 0.05 and cut.sum() > 1e-9:
                for k, (ctype, detail) in enumerate(_LIMITS):
                    units = first_miss * cut[k] / cut.sum()
                    if units <= 0.05:
                        continue
                    if detail is None:
                        c = int(r.binding_component[0, m, f])
                        detail = pa.comp_names[c] if c >= 0 else "Critical component"
                    constraint_rows.append({
                        "month": months[m], "type": ctype, "detail": detail,
                        "product_family": fam, "units_lost": round(units, 1)})

    monthly = pd.DataFrame({
        "month": months,
        "quarter": [f"Q{quarter_of_month(m)}" for m in range(N_MONTHS)],
        "units_demand": demand.sum(axis=1),
        "units_built": built.sum(axis=1),
        "units_recognized": rec_units.sum(axis=1),
        "units_unmet": unmet.sum(axis=1),
        "revenue_usd": r.revenue[0],
        "revenue_plan_usd": pa.revenue_plan_m,
        "cogs_usd": r.cogs[0],
        "gross_profit_usd": r.gross_profit[0],
        "gross_margin": r.gross_margin[0],
        "operating_income_usd": r.operating_income[0],
        "ebitda_usd": r.ebitda[0],
        "cash_flow_usd": r.cash_flow[0],
        "raw_inventory_usd": r.raw_inventory[0],
        "wip_inventory_usd": r.wip_inventory[0],
        "fg_inventory_usd": r.fg_inventory[0],
        "inventory_usd": r.inventory[0],
        "working_capital_usd": r.working_capital[0],
        "ems_utilization": site_load.sum(axis=0) / np.clip(site_cap.sum(axis=0), 1e-9, None),
    })

    plan_q = np.array([pa.revenue_plan_m[q * 3:(q + 1) * 3].sum() for q in range(6)])

    consumed = r.component_consumed[0].T            # (C, M)
    component_usage = pd.DataFrame({
        "component": np.repeat(pa.comp_names, N_MONTHS),
        "month": months * len(pa.comp_names),
        "consumed_units": consumed.flatten(),
        "cumulative_supply_units": r.component_usable_supply[0].T.flatten(),
        "cumulative_consumed_units": np.cumsum(consumed, axis=1).flatten(),
    })

    return BaselineResult(
        monthly=monthly,
        family_units=pd.DataFrame(rec_units, index=months, columns=fams),
        family_revenue=pd.DataFrame(r.family_revenue[0], index=months, columns=fams),
        site_load=pd.DataFrame(site_load, index=sites, columns=months),
        site_capacity=pd.DataFrame(site_cap, index=sites, columns=months),
        component_usage=component_usage,
        constraints=pd.DataFrame(constraint_rows) if constraint_rows else pd.DataFrame(
            columns=["month", "type", "detail", "product_family", "units_lost"]),
        unmet=pd.DataFrame(unmet, index=months, columns=fams),
        backlog_aging=pd.DataFrame(backlog_age_rows) if backlog_age_rows else pd.DataFrame(
            columns=["month", "age_months", "units"]),
        demand_units=pa.demand_units,
        supply_units=built,
        revenue_plan_q=plan_q,
        revenue_plan_m=pa.revenue_plan_m,
        customer_revenue=pd.DataFrame(r.customer_revenue[0], index=months,
                                      columns=r.customers),
    )
