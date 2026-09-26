"""Operational planning aggregates shared by the baseline plan and the simulator.

Converts the generated input tables into dense numpy structures:
demand by family/month, EMS capacity by site/month (final integration and test
included), and component supply pipelines. All units follow the input tables; EMS capacity uses
"standard-equivalent systems" where a family's configuration complexity weights
its capacity consumption.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import InputData
from .utils import EMS_SITES, N_MONTHS, PRODUCT_FAMILIES, month_labels


@dataclass
class PlanningArrays:
    """Dense arrays derived from the input tables."""

    months: list[str]
    families: list[str]
    demand_units: np.ndarray          # (n_months, n_fam) backlog + forecast units
    backlog_units: np.ndarray         # (n_months, n_fam) firm backlog portion
    family_asp: np.ndarray            # (n_fam,) demand-weighted average selling price
    family_unit_cost: np.ndarray      # (n_fam,) standard full unit cost
    family_material_cost: np.ndarray  # (n_fam,)
    family_complexity: np.ndarray     # (n_fam,) std-equivalent weight
    family_rec_lag: np.ndarray        # (n_fam,) revenue-recognition lag in months (0/1)
    family_accept_prob_delay: np.ndarray  # (n_fam,) probability an accepted system slips a month
    site_names: list[str]
    site_capacity: np.ndarray         # (n_sites, n_months) available std-equivalent units
    site_overtime: np.ndarray         # (n_sites, n_months) max overtime std-units
    site_adherence: np.ndarray        # (n_sites, n_months)
    site_labor: np.ndarray            # (n_sites, n_months)
    site_fpy: np.ndarray              # (n_sites, n_months)
    site_cost: np.ndarray             # (n_sites,) conversion cost per std unit
    site_ot_premium: np.ndarray       # (n_sites,) overtime premium as % of conversion cost
    site_disrupt_prob: np.ndarray     # (n_sites,)
    site_qual: np.ndarray             # (n_sites, n_fam) 1 if family qualified
    comp_names: list[str]
    comp_on_hand: np.ndarray          # (n_comp,)
    comp_po_monthly: np.ndarray       # (n_comp,)
    comp_safety: np.ndarray           # (n_comp,)
    comp_usage: np.ndarray            # (n_comp, n_fam) units consumed per system
    comp_cost: np.ndarray             # (n_comp,)
    comp_disrupt: np.ndarray          # (n_comp,) monthly disruption probability
    comp_alloc_risk: np.ndarray       # (n_comp,)
    comp_lead_time: np.ndarray        # (n_comp,) weeks
    comp_expedite_prem: np.ndarray    # (n_comp,) premium as % of unit cost
    comp_expedite_ok: np.ndarray      # (n_comp,) bool
    revenue_plan_m: np.ndarray        # (n_months,)
    pushout_prob: np.ndarray          # (n_fam,) demand-weighted monthly push-out prob
    pullin_prob: np.ndarray           # (n_fam,)
    cancel_prob: np.ndarray           # (n_fam,)
    site_readiness: np.ndarray        # (n_fam,) demand-weighted readiness probability
    # demand lines: one per customer x family (the unit of allocation)
    customers: list[str]              # sorted by FY demand-plan revenue, largest first
    cust_priority: np.ndarray         # (n_cust,) 1 = highest
    cust_group: np.ndarray            # (n_cust,) group index (correlated demand shocks)
    line_cust: np.ndarray             # (n_lines,) customer index
    line_fam: np.ndarray              # (n_lines,) family index
    line_backlog: np.ndarray          # (n_months, n_lines) booked backlog units
    line_forecast: np.ndarray         # (n_months, n_lines) unbooked forecast units
    line_asp: np.ndarray              # (n_lines,)
    line_push: np.ndarray             # (n_months, n_lines) push-out probability
    line_pull: np.ndarray             # (n_months, n_lines) pull-in probability
    line_cancel: np.ndarray           # (n_months, n_lines) cancellation probability
    line_readiness: np.ndarray        # (n_lines,) customer site-readiness probability


def build_planning_arrays(data: InputData) -> PlanningArrays:
    months = month_labels()
    fams = PRODUCT_FAMILIES
    n_f = len(fams)
    fam_ix = {f: i for i, f in enumerate(fams)}
    m_ix = {m: i for i, m in enumerate(months)}

    dem = data.demand.copy()
    dem["units"] = dem["base_forecast_units"] + dem["backlog_units"]
    demand_units = np.zeros((N_MONTHS, n_f))
    backlog_units = np.zeros((N_MONTHS, n_f))
    for (month, fam), g in dem.groupby(["month", "product_family"]):
        demand_units[m_ix[month], fam_ix[fam]] = g["units"].sum()
        backlog_units[m_ix[month], fam_ix[fam]] = g["backlog_units"].sum()

    def wavg(col: str) -> np.ndarray:
        out = np.zeros(n_f)
        for fam, g in dem.groupby("product_family"):
            w = g["units"].clip(lower=0.01)
            out[fam_ix[fam]] = np.average(g[col], weights=w)
        return out

    family_asp = wavg("asp_usd")
    prod = data.products.set_index("product_family").loc[fams]
    unit_cost = (prod["material_cost_usd"] + prod["ems_conversion_cost_usd"]
                 + prod["integration_test_cost_usd"] + prod["freight_cost_usd"]
                 + prod["warranty_reserve_usd"]).to_numpy(float)
    accept_lag = prod["acceptance_lag_months"].to_numpy(float)

    sites = EMS_SITES
    s_ix = {s: i for i, s in enumerate(sites)}
    cap = data.ems_capacity
    site_capacity = np.zeros((len(sites), N_MONTHS))
    site_overtime = np.zeros((len(sites), N_MONTHS))
    site_adherence = np.zeros((len(sites), N_MONTHS))
    site_labor = np.zeros((len(sites), N_MONTHS))
    site_fpy = np.zeros((len(sites), N_MONTHS))
    for _, r in cap.iterrows():
        i, j = s_ix[r["ems_site"]], m_ix[r["month"]]
        site_capacity[i, j] = r["available_capacity_units"]
        site_overtime[i, j] = r["max_overtime_units"]
        site_adherence[i, j] = r["schedule_adherence"]
        site_labor[i, j] = r["labor_availability"]
        site_fpy[i, j] = r["first_pass_yield"]

    site_master = data.ems_sites.set_index("ems_site").loc[sites]
    site_cost = site_master["cost_per_std_unit_usd"].to_numpy(float)
    site_ot_premium = site_master["overtime_premium_pct"].to_numpy(float)
    site_disrupt = site_master["regional_disruption_prob_monthly"].to_numpy(float)
    site_qual = np.zeros((len(sites), n_f))
    for fam in fams:
        qualified = str(prod.loc[fam, "qualified_ems_sites"]).split(";")
        for s in qualified:
            site_qual[s_ix[s], fam_ix[fam]] = 1.0


    comp = data.components
    comp_usage = np.zeros((len(comp), n_f))
    for k, (_, r) in enumerate(comp.iterrows()):
        for fam in str(r["products_using"]).split(";"):
            if fam in fam_ix:
                comp_usage[k, fam_ix[fam]] = r["usage_per_system"]

    # demand lines (customer x family); customers ordered by FY plan revenue
    dem["fy_rev"] = np.where(dem["month"].map(m_ix) < 12, dem["units"] * dem["asp_usd"], 0.0)
    customers = (dem.groupby("customer")["fy_rev"].sum()
                 .sort_values(ascending=False, kind="stable").index.tolist())
    c_ix = {c: i for i, c in enumerate(customers)}
    cmaster = dem.groupby("customer").agg(priority=("customer_priority", "first"),
                                          group=("customer_group", "first")).loc[customers]
    groups = sorted(cmaster["group"].unique())
    lines = sorted({(c_ix[c], fam_ix[f]) for c, f in zip(dem["customer"], dem["product_family"])})
    l_ix = {key: i for i, key in enumerate(lines)}
    n_l = len(lines)
    line_backlog = np.zeros((N_MONTHS, n_l))
    line_forecast = np.zeros((N_MONTHS, n_l))
    line_push = np.zeros((N_MONTHS, n_l))
    line_pull = np.zeros((N_MONTHS, n_l))
    line_cancel = np.zeros((N_MONTHS, n_l))
    line_asp = np.zeros(n_l)
    line_ready_num = np.zeros(n_l)
    line_ready_den = np.zeros(n_l)
    for r in dem.itertuples(index=False):
        li = l_ix[(c_ix[r.customer], fam_ix[r.product_family])]
        j = m_ix[r.month]
        line_backlog[j, li] += r.backlog_units
        line_forecast[j, li] += r.base_forecast_units
        line_push[j, li] = r.push_out_prob
        line_pull[j, li] = r.pull_in_prob
        line_cancel[j, li] = r.cancel_prob
        line_asp[li] = r.asp_usd
        w = max(r.units, 0.01)
        line_ready_num[li] += r.site_readiness_prob * w
        line_ready_den[li] += w

    fin = data.financial_plan.set_index("month").reindex(months)
    revenue_plan_m = fin["revenue_plan_usd"].to_numpy(float)

    return PlanningArrays(
        months=months, families=list(fams),
        demand_units=demand_units, backlog_units=backlog_units,
        family_asp=family_asp, family_unit_cost=unit_cost,
        family_material_cost=prod["material_cost_usd"].to_numpy(float),
        family_complexity=prod["config_complexity"].to_numpy(float),
        family_rec_lag=(accept_lag >= 0.75).astype(float),
        family_accept_prob_delay=np.clip(accept_lag / 3.0, 0.05, 0.5),
        site_names=list(sites), site_capacity=site_capacity,
        site_overtime=site_overtime, site_adherence=site_adherence,
        site_labor=site_labor, site_fpy=site_fpy, site_cost=site_cost,
        site_ot_premium=site_ot_premium,
        site_disrupt_prob=site_disrupt, site_qual=site_qual,
        comp_names=comp["component"].tolist(),
        comp_on_hand=comp["on_hand_units"].to_numpy(float),
        comp_po_monthly=comp["open_po_units_per_month"].to_numpy(float),
        comp_safety=comp["safety_stock_units"].to_numpy(float),
        comp_usage=comp_usage, comp_cost=comp["unit_cost_usd"].to_numpy(float),
        comp_disrupt=comp["disruption_prob_monthly"].to_numpy(float),
        comp_alloc_risk=comp["allocation_risk"].to_numpy(float),
        comp_lead_time=comp["lead_time_weeks"].to_numpy(float),
        comp_expedite_prem=comp["expedite_premium_pct"].to_numpy(float),
        comp_expedite_ok=comp["expedite_available"].to_numpy(bool),
        revenue_plan_m=revenue_plan_m,
        pushout_prob=wavg("push_out_prob"), pullin_prob=wavg("pull_in_prob"),
        cancel_prob=wavg("cancel_prob"), site_readiness=wavg("site_readiness_prob"),
        customers=customers,
        cust_priority=cmaster["priority"].to_numpy(int),
        cust_group=np.array([groups.index(g) for g in cmaster["group"]]),
        line_cust=np.array([c for c, _ in lines]), line_fam=np.array([f for _, f in lines]),
        line_backlog=line_backlog, line_forecast=line_forecast, line_asp=line_asp,
        line_push=line_push, line_pull=line_pull, line_cancel=line_cancel,
        line_readiness=line_ready_num / line_ready_den,
    )


def effective_site_capacity(pa: PlanningArrays, overtime: bool = False) -> np.ndarray:
    """Deterministic effective capacity (n_sites, n_months): available capacity
    derated by schedule adherence and labor availability."""
    cap = pa.site_capacity * pa.site_adherence * pa.site_labor
    if overtime:
        cap = cap + pa.site_overtime
    return cap


def standard_unit_cost(data: InputData, families: list[str],
                       fpy: float) -> dict[str, np.ndarray]:
    """The one unit-cost policy, per family: standard cost build-up plus scrap
    and expected rework (units failing first pass reworked at half conversion
    cost). COGS applies cost shocks to the components; finished goods are
    valued at `standard`."""
    prod = data.products.set_index("product_family").loc[families]
    out = {
        "material": prod["material_cost_usd"].to_numpy(float),
        "conversion": prod["ems_conversion_cost_usd"].to_numpy(float),
        "integration": prod["integration_test_cost_usd"].to_numpy(float),
        "freight": prod["freight_cost_usd"].to_numpy(float),
        "warranty": prod["warranty_reserve_usd"].to_numpy(float),
        "scrap": (prod["scrap_prob"] * prod["material_cost_usd"]).to_numpy(float),
    }
    out["rework"] = (1.0 - fpy) * 0.5 * out["conversion"]
    out["standard"] = sum(out[k] for k in ("material", "conversion", "integration",
                                           "freight", "warranty", "scrap", "rework"))
    return out
