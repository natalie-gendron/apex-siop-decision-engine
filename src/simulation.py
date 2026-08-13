"""Correlated Monte Carlo simulation of the 18-month SIOP plan.

Hybrid granularity (documented simplification): the deterministic baseline plan
runs at customer/site/component detail; the Monte Carlo engine simulates the
decision-relevant aggregates — demand by product family and month, EMS capacity
by site, supply for all 30 critical components, integration capacity, and the
financial translation. Within each month, scarce components and integration
capacity are rationed proportionally to demand (a vectorized approximation of
the baseline's priority ordering). Results reconcile to the baseline when all
shocks are set to zero.

All distributions are bounded: multiplicative shocks are lognormal (positive),
probabilities are clipped to [0, 1], capacities floored at zero, and discrete
events are Bernoulli. Correlations come exclusively from the factor model in
`correlations.py`, so no impossible covariance can arise.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import AppConfig
from .correlations import FactorEngine
from .models import BaselineResult, InputData, SimulationResult
from .operations import PlanningArrays, build_planning_arrays
from .utils import FAMILY_MARKET, N_MONTHS, PRODUCT_FAMILIES


def default_params() -> dict[str, Any]:
    """All scenario-adjustable knobs with base-case values."""
    return {
        # demand
        "demand_market_mult": {},        # market -> multiplier (applied to families in market)
        "demand_family_mult": {},        # family -> multiplier
        "demand_sigma_mult": 1.0,        # scales demand volatility (Demand Confidence driver)
        "pushout_prob_add": 0.0,
        "pullin_prob_add": 0.0,
        "cancel_prob_mult": 1.0,
        "asp_mult": 1.0,
        "forced_pushout": None,          # {"family":, "from_month":, "to_month":, "units":}
        # components
        "lead_time_mult": 1.0,           # scales delay fractions
        "comp_disrupt_mult": 1.0,
        "comp_supply_mult": {},          # component -> receipts multiplier
        "comp_supply_ramp": {},          # component -> (start_month, multiplier) after qual lag
        "safety_stock_mult": 1.0,
        "expedite_recovery": 0.5,        # fraction of delayed receipts recoverable by expediting
        "expedite_recovery_by_comp": {}, # component -> recovery fraction (targeted expediting)
        "expedite_premium_mult": 1.0,
        # EMS
        "ems_capacity_mult": {},         # site -> multiplier
        "ems_capacity_add": {},          # site -> std-units added (e.g. reserved capacity)
        "ems_capacity_add_ramp": {},     # site -> (start_month, std-units) added from that month
        "ems_window_mult": {},           # site -> (start_m, end_m, multiplier) temporary event
        "adherence_delta": 0.0,
        "fpy_delta": 0.0,
        "overtime_fraction": 0.0,        # fraction of max overtime authorized
        "overtime_start_month": 0,       # overtime effective from this month (decision latency)
        "add_qualification": [],         # [(site, family, start_month)] new EMS qualifications
        # integration / acceptance
        "integration_capacity_mult": 1.0,
        "integration_capacity_ramp": None,  # (start_month, multiplier) delayed capacity add
        "acceptance_delay_add": 0.0,     # added probability that revenue slips a month
        # finance
        "freight_mult": 1.0,
        "material_cost_mult": 1.0,
        "action_cost_usd": 0.0,          # one-time decision cost, spread over first quarter
    }


def _lognormal_mult(shock: np.ndarray, sigma: float) -> np.ndarray:
    """Mean-one lognormal multiplier driven by a standard-normal shock."""
    return np.exp(sigma * shock - 0.5 * sigma ** 2)


def run_simulation(data: InputData, config: AppConfig, baseline: BaselineResult,
                   params: dict[str, Any] | None = None, n_sims: int = 5000,
                   seed: int = 42, scenario_name: str = "Base Case",
                   progress_cb=None) -> SimulationResult:
    """Run the vectorized correlated Monte Carlo simulation."""
    p = default_params()
    if params:
        p.update(params)

    pa = build_planning_arrays(data)
    unc = config.uncertainty
    fin = config.financial
    engine = FactorEngine(config.factors)
    rng = np.random.default_rng(seed)

    fams = pa.families
    n_f = len(fams)
    n_c = len(pa.comp_names)
    n_s = len(pa.site_names)
    M = N_MONTHS

    if progress_cb:
        progress_cb(0.05, "Drawing correlated factor paths")
    factors = engine.draw_factor_paths(rng, n_sims, M)   # (n, M, K)

    # ------------------------------------------------------------------
    # 1. Demand: correlated market multipliers + timing events
    # ------------------------------------------------------------------
    market_shock = {mkt: engine.shock(mkt, factors, rng) for mkt in unc.market_demand_sigma}
    sigma_mult = float(p["demand_sigma_mult"])
    demand = np.empty((n_sims, M, n_f))
    for f, fam in enumerate(fams):
        mkt = FAMILY_MARKET[fam]
        mult = _lognormal_mult(market_shock[mkt],
                               unc.market_demand_sigma[mkt] * sigma_mult)
        idio = _lognormal_mult(rng.standard_normal((n_sims, 1)),
                               unc.customer_idiosyncratic_sigma * sigma_mult)
        scen = p["demand_market_mult"].get(mkt, 1.0) * p["demand_family_mult"].get(fam, 1.0)
        scen_path = np.asarray(scen, dtype=float)
        if scen_path.ndim == 0:
            scen_path = np.full(M, float(scen_path))
        demand[:, :, f] = pa.demand_units[None, :, f] * mult * idio * scen_path[None, :]

    # cancellations (remove), push-outs (shift +1/+2), pull-ins (shift -1)
    cancel_p = np.clip(pa.cancel_prob[None, None, :] * p["cancel_prob_mult"], 0, 1)
    push_p = np.clip(pa.pushout_prob[None, None, :] + p["pushout_prob_add"], 0, 1)
    pull_p = np.clip(pa.pullin_prob[None, None, :] + p["pullin_prob_add"], 0, 1)
    # sim-level heterogeneity in timing behavior (beta-like via lognormal clipping)
    timing_noise = _lognormal_mult(rng.standard_normal((n_sims, 1, 1)), 0.35)
    push_p = np.clip(push_p * timing_noise, 0, 0.6)
    pull_p = np.clip(pull_p * timing_noise ** 0.5, 0, 0.3)

    demand = demand * (1 - cancel_p)
    pushed = demand * push_p
    pulled = demand * pull_p
    demand = demand - pushed - pulled
    demand[:, 1:, :] += 0.7 * pushed[:, :-1, :]
    demand[:, 2:, :] += 0.3 * pushed[:, :-2, :]
    # pushes from the final months partially leave the 18-month horizon; in
    # steady state a similar inflow arrives from orders pushed before the
    # horizon start, so months 0-1 receive the mirrored inflow
    demand[:, 0, :] += 0.7 * pushed[:, 0, :]
    demand[:, 1, :] += 0.3 * pushed[:, 0, :]
    demand[:, :-1, :] += pulled[:, 1:, :]
    demand[:, 0, :] += pulled[:, 0, :]  # cannot pull before horizon; stays in month

    if p["forced_pushout"]:
        fp = p["forced_pushout"]
        f = fams.index(fp["family"])
        units = np.minimum(demand[:, fp["from_month"], f], fp["units"])
        demand[:, fp["from_month"], f] -= units
        demand[:, fp["to_month"], f] += units

    demand = np.clip(demand, 0, None)

    # ------------------------------------------------------------------
    # 2. Component supply pipelines (all critical components, vectorized)
    # ------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.25, "Simulating component supply")
    tight_shock = engine.shock("Component tightness", factors, rng)      # (n, M)
    logistics_shock = engine.shock("Logistics disruption", factors, rng)

    receipts_nominal = np.tile(pa.comp_po_monthly[None, None, :], (n_sims, M, 1))
    for comp, mult in p["comp_supply_mult"].items():
        if comp == "__all__":
            receipts_nominal *= mult
        else:
            receipts_nominal[:, :, pa.comp_names.index(comp)] *= mult
    for comp, (start_m, mult) in p["comp_supply_ramp"].items():
        if comp == "__all__":
            receipts_nominal[:, start_m:, :] *= mult
        else:
            receipts_nominal[:, start_m:, pa.comp_names.index(comp)] *= mult

    # delay fraction rises with supply tightness, logistics disruption and lead time
    lt_weeks = pa.comp_lead_time[None, None, :] * p["lead_time_mult"]
    base_delay = np.clip((lt_weeks - 8.0) / 100.0, 0.01, 0.25)
    delay_frac = np.clip(
        base_delay
        + 0.06 * np.clip(tight_shock, 0, None)[:, :, None] * pa.comp_alloc_risk[None, None, :] * 2.0
        + 0.03 * np.clip(logistics_shock, 0, None)[:, :, None]
        + (p["lead_time_mult"] - 1.0) * 0.25,
        0.0, 0.6,
    )
    disrupt_p = np.clip(pa.comp_disrupt[None, None, :] * p["comp_disrupt_mult"]
                        * (1 + 0.8 * np.clip(tight_shock, 0, None))[:, :, None], 0, 0.5)
    disrupted = rng.random((n_sims, M, n_c)) < disrupt_p

    received = receipts_nominal * (1 - delay_frac)
    received[:, 1:, :] += (receipts_nominal * delay_frac)[:, :-1, :]
    received = received * np.where(disrupted, 0.35, 1.0)

    # Expediting recovers part of delayed receipts at a premium — but only
    # where the receipts are NEEDED: units are expedited up to the projected
    # cumulative shortfall against the demand plan's component requirement
    # to date (the planner's view when the broker is paid — requested dates,
    # not realized shipments). The recovery fractions cap how much of the
    # delayed pool CAN be recovered; need caps how much IS.
    # (A volume-based premium on every late PO priced the standing policy as
    # ~$28M/yr of waste in the base world, made "stop expediting" free money,
    # and taxed every receipts-adding action with premium on its own delayed
    # pool — 2026-08 audit finding.)
    delayed_pool = receipts_nominal * delay_frac + receipts_nominal * np.where(disrupted, 0.65, 0.0)
    recovery_vec = np.full(n_c, float(p["expedite_recovery"]))
    for comp, frac in p["expedite_recovery_by_comp"].items():
        recovery_vec[pa.comp_names.index(comp)] = float(frac)
    recovery = recovery_vec * np.where(pa.comp_expedite_ok, 1.0, 0.0)   # (C,)
    # safety stock is usable in a pinch: only a fraction is treated as a hard
    # floor; the policy level mainly shows up in average raw-material inventory
    safety_floor = pa.comp_safety * p["safety_stock_mult"] * 0.3
    start_avail = np.clip(pa.comp_on_hand - safety_floor, 0, None)      # (C,)
    cum_req = np.cumsum(np.einsum("nmf,cf->nmc", demand, pa.comp_usage), axis=1)
    recovered = np.zeros_like(received)
    cum_supply = np.tile(start_avail[None, :], (n_sims, 1))             # (n, C)
    for m in range(M):
        cum_supply = cum_supply + received[:, m, :]
        need = np.clip(cum_req[:, m, :] - cum_supply, 0, None)
        recovered[:, m, :] = np.minimum(delayed_pool[:, m, :] * recovery[None, :],
                                        need)
        cum_supply = cum_supply + recovered[:, m, :]
    received = received + recovered
    expedite_cost_comp = (recovered * pa.comp_cost[None, None, :]
                          * pa.comp_expedite_prem[None, None, :]
                          * p["expedite_premium_mult"]).sum(axis=2)   # (n, M)

    comp_avail = np.clip(pa.comp_on_hand[None, None, :] - safety_floor[None, None, :], 0, None) \
        + np.cumsum(received, axis=1)                                  # usable cumulative supply

    # ------------------------------------------------------------------
    # 3. EMS + integration capacity
    # ------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.45, "Simulating EMS and integration capacity")
    ems_shock = engine.shock("EMS execution", factors, rng)             # (n, M)
    labor_mult = np.clip(_lognormal_mult(ems_shock, unc.ems_labor_sigma), 0.7, 1.1)

    site_cap = np.empty((n_sims, n_s, M))
    site_disrupted: dict[str, np.ndarray] = {}
    for s, site in enumerate(pa.site_names):
        base = pa.site_capacity[s] * pa.site_labor[s]                   # (M,)
        mult = p["ems_capacity_mult"].get(site, 1.0)
        add = p["ems_capacity_add"].get(site, 0.0)
        cap = base[None, :] * mult + add
        if site in p["ems_capacity_add_ramp"]:
            start_m, units = p["ems_capacity_add_ramp"][site]
            cap = cap.copy()
            cap[:, start_m:] += units
        if site in p["ems_window_mult"]:
            m0, m1, wmult = p["ems_window_mult"][site]
            cap = cap.copy()
            cap[:, m0:m1] *= wmult
        events = rng.random((n_sims, M)) < pa.site_disrupt_prob[s]
        site_disrupted[site] = events.any(axis=1)
        cap = cap * np.where(events, 1 - unc.site_disruption_impact, 1.0)
        cap = cap * labor_mult
        cap_no_ot = np.clip(cap, 0, None)
        ot_mask = np.ones(M)
        ot_mask[:int(p["overtime_start_month"])] = 0.0
        cap = cap + p["overtime_fraction"] * pa.site_overtime[s][None, :] * ot_mask[None, :]
        site_cap[:, s, :] = np.clip(cap, 0, None)
        if s == 0:
            base_cap_total = cap_no_ot.copy()
        else:
            base_cap_total += cap_no_ot

    # utilization feedback: high load erodes schedule adherence (and thus output)
    demand_std = (demand * pa.family_complexity[None, None, :]).sum(axis=2)  # (n, M)
    total_cap_raw = site_cap.sum(axis=1)
    util_prelim = demand_std / np.clip(total_cap_raw, 1e-9, None)
    adherence_base = (pa.site_adherence.mean(axis=0)[None, :] + p["adherence_delta"])
    adherence_eff = np.clip(
        adherence_base - unc.utilization_adherence_penalty * np.clip(util_prelim - 0.92, 0, None) / 0.10,
        0.6, 1.0,
    )
    site_cap = site_cap * adherence_eff[:, None, :]

    integ_mult_path = np.full(M, p["integration_capacity_mult"])
    if p["integration_capacity_ramp"]:
        ramp_start, ramp_mult = p["integration_capacity_ramp"]
        integ_mult_path[ramp_start:] *= ramp_mult
    integ_cap = (pa.integration_capacity[None, :] * integ_mult_path[None, :]
                 * np.clip(labor_mult, 0.8, 1.05))

    # first-pass yield: good output = builds * fpy_eff; rework recovered at cost
    fpy_base = float((pa.site_fpy * pa.site_capacity).sum() / pa.site_capacity.sum())
    fpy_eff = np.clip(fpy_base + p["fpy_delta"]
                      - 0.03 * np.clip(util_prelim - 0.9, 0, None) / 0.1
                      - 0.02 * np.clip(-ems_shock, 0, None), 0.7, 0.99)

    # ------------------------------------------------------------------
    # 4. Monthly shipment loop (vectorized across sims and families)
    # ------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.6, "Allocating supply against demand")
    # site-family qualification may change mid-horizon (new-site qualification)
    qual_by_month = np.tile(pa.site_qual[None, :, :], (M, 1, 1))
    for q_site, q_family, q_start in p["add_qualification"]:
        qual_by_month[q_start:, pa.site_names.index(q_site), fams.index(q_family)] = 1.0

    ship = np.zeros((n_sims, M, n_f))
    backlog = np.zeros((n_sims, n_f))
    cum_consumed = np.zeros((n_sims, n_c))
    comp_short = np.zeros((n_sims, M))
    cap_short = np.zeros((n_sims, M))
    comp_binding_count = np.zeros((n_sims, n_c))
    usage = pa.comp_usage                                              # (C, F)

    for m in range(M):
        want = demand[:, m, :] + backlog                               # (n, F)

        # component rationing: proportional scale per component, min across components
        avail = np.clip(comp_avail[:, m, :] - cum_consumed, 0, None)   # (n, C)
        req = want @ usage.T                                           # (n, C)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale_c = np.where(req > 1e-9, np.minimum(1.0, avail / req), 1.0)
        # a family's scale is the worst of its components
        fam_scale = np.ones((n_sims, n_f))
        for f in range(n_f):
            used = usage[:, f] > 0
            if used.any():
                fam_scale[:, f] = scale_c[:, used].min(axis=1)
        binding = scale_c < 0.999
        comp_binding_count += binding & (req > 1e-9)
        after_comp = want * fam_scale

        # EMS capacity by family: iterative water-filling. Each round, every
        # site allocates its remaining capacity across qualified families in
        # proportion to their remaining standard-equivalent demand; three
        # rounds recover nearly all slack a single proportional pass strands.
        after_cap = np.zeros((n_sims, n_f))
        site_rem = site_cap[:, :, m].copy()                            # (n, S)
        for _ in range(3):
            unmet_u = np.clip(after_comp - after_cap, 0, None)
            unmet_std = unmet_u * pa.family_complexity[None, :]
            if unmet_std.sum() < 1e-6:
                break
            for s in range(n_s):
                qual = qual_by_month[m, s]                             # (F,)
                q_std = unmet_std * qual[None, :]
                denom = q_std.sum(axis=1, keepdims=True)
                share = np.divide(q_std, denom, out=np.zeros_like(q_std),
                                  where=denom > 1e-9)
                give_std = np.minimum(q_std, site_rem[:, s][:, None] * share)
                after_cap += give_std / pa.family_complexity[None, :]
                site_rem[:, s] -= give_std.sum(axis=1)
                unmet_std = np.clip(unmet_std - give_std, 0, None)
        after_cap = np.minimum(after_cap, after_comp)

        # integration capacity: proportional company-level scale
        total_after = after_cap.sum(axis=1)
        integ_scale = np.minimum(1.0, integ_cap[:, m] / np.clip(total_after, 1e-9, None))
        shipped = after_cap * integ_scale[:, None]

        ship[:, m, :] = shipped
        backlog = want - shipped
        cum_consumed += shipped @ usage.T
        comp_short[:, m] = (want - after_comp).sum(axis=1)
        cap_short[:, m] = (after_comp - shipped).sum(axis=1)

    ems_load_std = (ship * pa.family_complexity[None, None, :]).sum(axis=2)
    total_cap = site_cap.sum(axis=1)
    ems_util = ems_load_std / np.clip(total_cap, 1e-9, None)
    integ_util = ship.sum(axis=2) / np.clip(integ_cap, 1e-9, None)

    # ------------------------------------------------------------------
    # 5. Revenue recognition (with acceptance / site-readiness slip)
    # ------------------------------------------------------------------
    accept_delay_p = np.clip(
        pa.family_accept_prob_delay[None, :] + p["acceptance_delay_add"]
        + (1 - pa.site_readiness[None, :]), 0, 0.9)                    # (1, F)
    slip_frac = accept_delay_p[None, :, :] * np.ones((n_sims, 1, 1))
    slip_noise = np.clip(rng.beta(4, 6, size=(n_sims, 1, 1)) * 2.0, 0.3, 1.7)
    slip_frac = np.clip(slip_frac * slip_noise, 0, 0.9)

    rec = np.zeros_like(ship)
    for f in range(n_f):
        lag = int(pa.family_rec_lag[f])
        base_rec = np.zeros((n_sims, M))
        if lag == 0:
            base_rec[:] = ship[:, :, f]
        else:
            base_rec[:, lag:] = ship[:, :-lag, f]
            base_rec[:, :lag] = ship[:, :lag, f].mean(axis=1, keepdims=True)
        sf = slip_frac[:, 0, f][:, None]
        rec_f = base_rec * (1 - sf)
        rec_f[:, 1:] += (base_rec * sf)[:, :-1]
        # steady-state inflow from systems that slipped acceptance before the horizon
        rec_f[:, 0] += base_rec[:, 0] * sf[:, 0]
        rec[:, :, f] = rec_f

    # ------------------------------------------------------------------
    # 6. Financial translation
    # ------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.8, "Translating to financial outcomes")
    asp_shock = rng.standard_normal((n_sims, 1, n_f))
    asp_mult = _lognormal_mult(asp_shock, unc.asp_sigma) * p["asp_mult"]
    revenue_fm = rec * pa.family_asp[None, None, :] * asp_mult
    revenue = revenue_fm.sum(axis=2)

    ppv_mult = _lognormal_mult(rng.standard_normal((n_sims, 1)), unc.material_cost_sigma)
    fx_mult = _lognormal_mult(rng.standard_normal((n_sims, 1)), unc.fx_cost_sigma)
    tight_cost = 1 + 0.02 * np.clip(tight_shock, 0, None)              # tight supply raises cost
    material_mult = ppv_mult * fx_mult * tight_cost * p["material_cost_mult"]  # (n, M)
    conv_mult = _lognormal_mult(rng.standard_normal((n_sims, 1)), unc.conversion_cost_sigma)
    freight_mult = (_lognormal_mult(rng.standard_normal((n_sims, 1)), unc.freight_sigma)
                    * p["freight_mult"]
                    * (1 + 0.08 * np.clip(logistics_shock, 0, None)))

    prod = data.products.set_index("product_family").loc[fams]
    mat_c = prod["material_cost_usd"].to_numpy(float)
    conv_c = prod["ems_conversion_cost_usd"].to_numpy(float)
    integ_c = prod["integration_test_cost_usd"].to_numpy(float)
    freight_c = prod["freight_cost_usd"].to_numpy(float)
    warr_c = prod["warranty_reserve_usd"].to_numpy(float)
    scrap_c = (prod["scrap_prob"] * prod["material_cost_usd"]).to_numpy(float)

    cogs = (rec * mat_c[None, None, :]).sum(axis=2) * material_mult \
        + (rec * conv_c[None, None, :]).sum(axis=2) * conv_mult \
        + (rec * integ_c[None, None, :]).sum(axis=2) \
        + (rec * freight_c[None, None, :]).sum(axis=2) * freight_mult \
        + (rec * (warr_c + scrap_c)[None, None, :]).sum(axis=2)

    rework_units = ship.sum(axis=2) * (1 - fpy_eff)
    rework_cost = rework_units * 0.5 * float(conv_c.mean())
    # overtime conversion premium: std-units produced above base capacity carry
    # the weighted-average overtime premium on EMS conversion cost
    if p["overtime_fraction"] > 0:
        adj_base_cap = base_cap_total * adherence_eff
        ot_used_std = np.clip(ems_load_std - adj_base_cap, 0, None)
        # overtime-capacity-weighted average of each site's contracted premium
        # (overtime_premium_pct in the EMS site table)
        ot_weights = pa.site_overtime.mean(axis=1)
        ot_weights = ot_weights / max(ot_weights.sum(), 1e-9)
        ot_premium = float((pa.site_cost * pa.site_ot_premium * ot_weights).sum())
        overtime_cost = ot_used_std * ot_premium
    else:
        overtime_cost = np.zeros_like(rework_cost)
    # capacity-driven expediting when past-due backlog exists
    expedite_cost = expedite_cost_comp + comp_short * 0.02 * float(mat_c.mean()) \
        * p["expedite_premium_mult"]
    cogs = cogs + rework_cost + expedite_cost + overtime_cost

    gross_profit = revenue - cogs
    action_cost_m = np.zeros(M)
    action_cost_m[:3] = p["action_cost_usd"] / 3.0
    operating_income = gross_profit - fin.opex_monthly_usd - action_cost_m[None, :]
    ebitda = operating_income + fin.depreciation_monthly_usd

    # inventory: critical-component RM + non-critical RM + WIP + FG awaiting acceptance
    cum_consumed_path = _cum_consumed_path(ship, usage)                # (n, M, C)
    stock_path = np.clip(comp_avail - cum_consumed_path, 0, None)     # (n, M, C)
    # purchasing response: buyers defer/reschedule roughly half of any stock
    # beyond two months of forward requirement, damping runaway inventory
    monthly_req_units = pa.comp_usage @ pa.demand_units.mean(axis=0)   # (C,)
    excess_path = np.clip(stock_path - 2.0 * monthly_req_units[None, None, :], 0, None)
    held_stock = stock_path - 0.5 * excess_path
    rm_crit = (held_stock * pa.comp_cost[None, None, :]).sum(axis=2)
    mat_spend = (ship * mat_c[None, None, :]).sum(axis=2)
    rm = rm_crit + 0.9 * mat_spend
    cycle = prod["build_cycle_months"].to_numpy(float)
    wip = (ship * ((mat_c + conv_c) * cycle * 0.6)[None, None, :]).sum(axis=2)
    unit_cogs_std = mat_c + conv_c + integ_c + freight_c + warr_c
    fg = np.clip(np.cumsum(ship - rec, axis=1), 0, None) @ unit_cogs_std
    inventory = rm + wip + fg

    ar = revenue * fin.dso_days / 30.0
    ap = cogs * 0.75 * fin.dpo_days / 30.0
    working_capital = inventory + ar - ap
    dwc = np.diff(working_capital, axis=1, prepend=working_capital[:, :1])
    cash_flow = ebitda - dwc - fin.capex_monthly_usd \
        - np.clip(operating_income, 0, None) * fin.tax_rate

    # E&O: excess critical-component stock above 2.5 months of usage at FY end,
    # reserved at the configured rate, plus obsolescence-weighted aged FG
    excess_rm = np.clip(stock_path[:, 11, :] - 2.5 * monthly_req_units[None, :], 0, None)
    eo_reserve = (excess_rm * pa.comp_cost[None, :]).sum(axis=1) * fin.eo_reserve_rate \
        + fg[:, 11] * 0.05

    if progress_cb:
        progress_cb(0.95, "Collecting outputs")

    drivers = {
        "Global semicap factor": factors[:, :3, 0].mean(axis=1),
        "AI / HPC demand factor": factors[:, :3, 1].mean(axis=1),
        "Memory cycle factor": factors[:, :3, 2].mean(axis=1),
        "Mobile cycle factor": factors[:, :3, 3].mean(axis=1),
        "Auto & industrial factor": factors[:, :3, 4].mean(axis=1),
        "Component tightness factor": factors[:, :3, 5].mean(axis=1),
        "Logistics disruption factor": factors[:, :3, 6].mean(axis=1),
        "EMS labor & execution factor": factors[:, :3, 7].mean(axis=1),
        "Push-out intensity": timing_noise[:, 0, 0],
        "Realized ASP multiplier": asp_mult[:, 0, :].mean(axis=1),
        "Material cost multiplier": material_mult[:, :3].mean(axis=1),
        "Freight cost multiplier": freight_mult[:, :3].mean(axis=1),
        "First-pass yield": fpy_eff[:, :3].mean(axis=1),
        "Component delay fraction": delay_frac[:, :3, :].mean(axis=(1, 2)),
        "Acceptance slip fraction": slip_frac[:, 0, :].mean(axis=1),
    }

    comp_binding = {name: comp_binding_count[:, i] > 0
                    for i, name in enumerate(pa.comp_names)}

    return SimulationResult(
        n_sims=n_sims, seed=seed, scenario_name=scenario_name,
        revenue=revenue, cogs=cogs, gross_profit=gross_profit,
        operating_income=operating_income, ebitda=ebitda, cash_flow=cash_flow,
        inventory=inventory, raw_inventory=rm, wip_inventory=wip, fg_inventory=fg,
        working_capital=working_capital, expedite_cost=expedite_cost,
        rework_cost=rework_cost, eo_reserve=eo_reserve,
        family_revenue=revenue_fm, family_units=rec,
        family_shipped=ship, family_demand=demand,
        units_shipped=ship.sum(axis=2), units_demanded=demand.sum(axis=2),
        ems_utilization=ems_util, integration_utilization=integ_util,
        capacity_shortfall_units=cap_short, component_short_units=comp_short,
        component_binding=comp_binding, site_disrupted=site_disrupted,
        drivers=drivers, params=p,
    )


def _cum_consumed_path(ship: np.ndarray, usage: np.ndarray) -> np.ndarray:
    """Cumulative component consumption path, shape (n_sims, M, C)."""
    monthly = np.einsum("nmf,cf->nmc", ship, usage)
    return np.cumsum(monthly, axis=1)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def quarterly(arr: np.ndarray) -> np.ndarray:
    """Sum an (n_sims, 18) monthly array into (n_sims, 6) quarters."""
    return arr.reshape(arr.shape[0], 6, 3).sum(axis=2)


def fiscal_year(arr: np.ndarray) -> np.ndarray:
    """Sum the first 12 months (the fiscal year)."""
    return arr[:, :12].sum(axis=1)


def service_level(result: SimulationResult) -> np.ndarray:
    """FY fill rate: units shipped / units demanded (capped at 1)."""
    shipped = fiscal_year(result.units_shipped)
    demanded = np.clip(fiscal_year(result.units_demanded), 1e-9, None)
    return np.clip(shipped / demanded, 0, 1)
