"""The APEX engine: correlated Monte Carlo of the 18-month SIOP plan.

One engine. The deterministic baseline supply plan is this function run with
`Shocks.zero()` and one path (`baseline_plan.run_baseline`); there is no
second allocator and no second financial translation.

Granularity: demand by product family and month, EMS capacity by site, supply
for all 30 critical components, and the financial translation. Final
integration and test is performed at the EMS, so it is part of EMS site
capacity, not a separate stage. Within each month, scarce components are
rationed proportionally within a family; EMS capacity is water-filled across
sites, least-contested site first, then cheapest.

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
from .operations import REWORK_SHARE, PlanningArrays, build_planning_arrays, standard_unit_cost
from .shocks import Shocks
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
        "customer_demand_edit": {},      # name -> {"customer", "family" (optional), "month",
                                         # "units", "to_month" (optional)}: a known customer
                                         # event; no to_month = upside ask, else pull-in/push-out
        "lost_after_months": {},         # customer -> months before a shorted order is lost
                                         # (overrides config.customers per key; what-if only)
        # allocation policy (SIOP-owned, response axis)
        "allocation_policy": "strict_priority",  # see ALLOCATION_POLICIES
        "protect_top_n": 3,              # customers protected under protect_top_n
        # components
        "lead_time_mult": 1.0,           # scales delay fractions
        "comp_disrupt_mult": 1.0,
        "comp_supply_mult": {},          # component -> receipts multiplier
        "comp_supply_ramp": {},          # component -> (start_month, multiplier) after qual lag
        "dual_source": {},               # component -> (start_month, share) on an alternate
                                         # source with its own, independent disruptions
        "buy_ahead": {},                 # component -> (order_month, months_of_cover):
                                         # a non-cancellable order, arrives one lead time later
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
        # acceptance
        "acceptance_delay_add": 0.0,     # added probability that revenue slips a month
        # finance
        "freight_mult": 1.0,
        "material_cost_mult": 1.0,
        "action_cost_usd": 0.0,          # one-time decision cost, spread over first quarter
        "recurring_cost": {},            # action -> (start_month, usd_per_month) while held
    }


# Engine constants (listed in the variable map; no owner yet)
FIRM_CANCEL_SHARE = 0.5     # booked backlog cancels at half the forecast rate
CUSTOMER_GROUP_SHARE = 0.5  # share of customer demand-shock variance from its group


# allocation policy: a SIOP-owned decision (response axis). It moves revenue
# between customers and months; it changes margin only through mix.
ALLOCATION_POLICIES = {
    "strict_priority": "Strict priority, backlog first",
    "priority_first": "Strict priority, priority first",
    "proportional": "Proportional (fair share)",
    "protect_top_n": "Protect top customers",
}


def allocation_tiers(policy: str, pa: PlanningArrays, top_n: int = 3) -> list[np.ndarray]:
    """Slice indices per tier, in serving order. Slices 0..L-1 are each line's
    backlog (booked orders plus carried past-due), L..2L-1 its forecast.

    strict_priority  every customer's backlog before anyone's forecast; customer
                     priority (1 first) within each (default: honors commitments)
    priority_first   priority 1 entirely (backlog, then forecast), then priority 2...
    proportional     one tier: everyone gets the same fill ratio within a family
    protect_top_n    the top-N customers by FY plan revenue (all slices), then
                     everyone else's backlog, then everyone else's forecast
    """
    n_l = len(pa.line_fam)
    firm_ix, fcst_ix = np.arange(n_l), np.arange(n_l, 2 * n_l)
    prio = pa.cust_priority[pa.line_cust]
    if policy == "proportional":
        tiers = [np.arange(2 * n_l)]
    elif policy == "strict_priority":
        levels = np.unique(prio)
        tiers = [firm_ix[prio == lv] for lv in levels] + [fcst_ix[prio == lv] for lv in levels]
    elif policy == "priority_first":
        tiers = [t for lv in np.unique(prio) for t in (firm_ix[prio == lv], fcst_ix[prio == lv])]
    elif policy == "protect_top_n":
        top = pa.line_cust < top_n          # customers are ordered by FY plan revenue
        tiers = [np.r_[firm_ix[top], fcst_ix[top]], firm_ix[~top], fcst_ix[~top]]
    else:
        raise ValueError(f"unknown allocation policy {policy!r}; "
                         f"choose from {sorted(ALLOCATION_POLICIES)}")
    return [t for t in tiers if len(t)]


SUPPLIER_UPSIDE_FLEX = 0.25  # suppliers deliver up to 125% of the open-PO rate
WEEKS_PER_MONTH = 4.345


def _lognormal_mult(shock: np.ndarray, sigma: float) -> np.ndarray:
    """Mean-one lognormal multiplier driven by a standard-normal shock."""
    return np.exp(sigma * shock - 0.5 * sigma ** 2)


def run_simulation(data: InputData, config: AppConfig,
                   baseline: BaselineResult | None = None,
                   params: dict[str, Any] | None = None, n_sims: int = 5000,
                   seed: int = 42, scenario_name: str = "Base Case",
                   progress_cb=None, shocks: Shocks | None = None,
                   keep_component_paths: bool = False) -> SimulationResult:
    """Run the vectorized correlated Monte Carlo simulation.

    `shocks` scales every source of randomness (default: as calibrated).
    With `Shocks.zero()` every path is identical and equals the baseline
    supply plan. `baseline` is accepted for call-site compatibility and is
    not read. `keep_component_paths` stores per-component consumption and
    usable supply (large; the baseline views need them, the app does not)."""
    p = default_params()
    if params:
        p.update(params)
    sh = shocks or Shocks()

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
    # 1. Demand by line (customer x family). Backlog is booked: timing risk
    #    only. Forecast takes market and customer volume shocks too.
    # ------------------------------------------------------------------
    lfam, lcust = pa.line_fam, pa.line_cust
    n_l, n_cu = len(lfam), len(pa.customers)
    fam_onehot = np.eye(n_f)[lfam]                                     # (L, F)
    cust_onehot = np.eye(n_cu)[lcust]                                  # (L, Cu)

    market_shock = {mkt: engine.shock(mkt, factors, rng) for mkt in unc.market_demand_sigma}
    sigma_mult = float(p["demand_sigma_mult"]) * sh.demand
    fam_mult = np.empty((n_sims, M, n_f))
    scen = np.empty((M, n_f))
    for f, fam in enumerate(fams):
        mkt = FAMILY_MARKET[fam]
        fam_mult[:, :, f] = _lognormal_mult(market_shock[mkt],
                                            unc.market_demand_sigma[mkt] * sigma_mult)
        scen[:, f] = np.broadcast_to(np.asarray(
            p["demand_market_mult"].get(mkt, 1.0), dtype=float), (M,)) \
            * np.broadcast_to(np.asarray(p["demand_family_mult"].get(fam, 1.0), dtype=float), (M,))
    # persistent customer shock; customers in one group share part of it
    group_z = rng.standard_normal((n_sims, int(pa.cust_group.max()) + 1))
    own_z = rng.standard_normal((n_sims, n_cu))
    cust_z = (np.sqrt(CUSTOMER_GROUP_SHARE) * group_z[:, pa.cust_group]
              + np.sqrt(1 - CUSTOMER_GROUP_SHARE) * own_z)
    cust_mult = _lognormal_mult(cust_z, unc.customer_idiosyncratic_sigma * sigma_mult)

    # calibrated event rates scale with the timing shock; lever deltas
    # (pushout_prob_add etc.) apply on top, so they survive a zero-shock run
    ev = sh.timing_events
    cancel_p = np.clip(pa.line_cancel[None] * ev * p["cancel_prob_mult"], 0, 1)
    push_p = np.clip(pa.line_push[None] * ev + p["pushout_prob_add"], 0, 1)
    pull_p = np.clip(pa.line_pull[None] * ev + p["pullin_prob_add"], 0, 1)
    # path-level heterogeneity in timing behavior (beta-like via lognormal clipping)
    timing_noise = _lognormal_mult(rng.standard_normal((n_sims, 1, 1)), 0.35 * ev)
    push_p = np.clip(push_p * timing_noise, 0, 0.6)
    pull_p = np.clip(pull_p * timing_noise ** 0.5, 0, 0.3)

    scen_l = scen[None][:, :, lfam]                                    # (1, M, L)
    firm = pa.line_backlog[None] * scen_l * (1 - FIRM_CANCEL_SHARE * cancel_p)
    fcst = (pa.line_forecast[None] * scen_l * (1 - cancel_p)
            * fam_mult[:, :, lfam] * cust_mult[:, None, lcust])

    # push-outs move the whole line-month order: one uniform per line-month
    # decides it (u < p pushes; u < 0.3p slips two months); pushed orders
    # return as backlog
    u = rng.random((n_sims, M, n_l))
    pushed = u < push_p
    two = u < 0.3 * push_p
    tot = firm + fcst
    firm = np.where(pushed, 0.0, firm)
    fcst = np.where(pushed, 0.0, fcst)
    firm[:, 1:] += (tot * (pushed & ~two))[:, :-1]
    firm[:, 2:] += (tot * two)[:, :-2]
    # pushes from the final months leave the horizon; in steady state a
    # similar inflow arrives from orders pushed before it, so months 0-1
    # receive the expected month-0 pushes
    inflow = tot[:, 0] * push_p[:, 0]
    firm[:, 0] += 0.7 * inflow
    firm[:, 1] += 0.3 * inflow
    # pull-ins: an expected share of next month's orders books a month early
    pulled = (firm + fcst) * pull_p
    firm, fcst = firm * (1 - pull_p), fcst * (1 - pull_p)
    firm[:, :-1] += pulled[:, 1:]
    firm[:, 0] += pulled[:, 0]  # cannot pull before horizon; stays in month

    if p["forced_pushout"]:
        fp = p["forced_pushout"]
        in_fam = lfam == fams.index(fp["family"])
        a, b = fp["from_month"], fp["to_month"]
        tot_a = (firm[:, a] + fcst[:, a]) * in_fam
        fam_tot = tot_a.sum(axis=1, keepdims=True)
        share = np.divide(np.minimum(fam_tot, fp["units"]), fam_tot,
                          out=np.zeros_like(fam_tot), where=fam_tot > 1e-12)
        firm[:, b] += tot_a * share
        firm[:, a] -= firm[:, a] * in_fam * share
        fcst[:, a] -= fcst[:, a] * in_fam * share

    # known customer events: an upside ask arrives as booked orders; a
    # pull-in or push-out moves up to the units asked, landing as booked
    for label, e in p["customer_demand_edit"].items():
        if e["customer"] not in pa.customers:
            raise KeyError(f"customer_demand_edit '{label}': unknown customer {e['customer']!r}")
        mask = lcust == pa.customers.index(e["customer"])
        if e.get("family"):
            mask &= lfam == fams.index(e["family"])
        if not mask.any():
            raise KeyError(f"customer_demand_edit '{label}': no demand line for "
                           f"{e['customer']!r} / {e.get('family')!r}")
        m0, units = int(e["month"]), float(e["units"])
        if e.get("to_month") is None:
            plan_m = (pa.line_backlog[m0] + pa.line_forecast[m0]) * mask
            w = plan_m / plan_m.sum() if plan_m.sum() > 0 else mask / mask.sum()
            firm[:, m0] += units * w[None, :]
        else:
            to = int(e["to_month"])
            here = (firm[:, m0] + fcst[:, m0]) * mask
            total = here.sum(axis=1, keepdims=True)
            share = np.divide(np.minimum(total, units), total,
                              out=np.zeros_like(total), where=total > 1e-12)
            firm[:, to] += here * share
            firm[:, m0] -= firm[:, m0] * mask * share
            fcst[:, m0] -= fcst[:, m0] * mask * share

    firm = np.clip(firm, 0, None)
    fcst = np.clip(fcst, 0, None)
    demand = (firm + fcst) @ fam_onehot                                # (n, M, F)

    # ------------------------------------------------------------------
    # 2. Component supply setup. Inside each part's lead time receipts are
    #    the open POs; beyond it they are planned orders that buyers place
    #    each month (section 4) and that arrive one lead time later.
    # ------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.25, "Simulating component supply")
    tight_shock = engine.shock("Component tightness", factors, rng) * sh.component_tightness
    logistics_shock = engine.shock("Logistics disruption", factors, rng) * sh.logistics

    # supplier capacity by part and month: comp_supply_mult / _ramp scale it
    # (a supplier cut, a second source, a capacity commitment)
    cap_mult = np.ones((M, n_c))
    for comp, mult in p["comp_supply_mult"].items():
        cols = slice(None) if comp == "__all__" else pa.comp_names.index(comp)
        cap_mult[:, cols] *= mult
    for comp, (start_m, mult) in p["comp_supply_ramp"].items():
        cols = slice(None) if comp == "__all__" else pa.comp_names.index(comp)
        cap_mult[start_m:, cols] *= mult
    # a supplier can deliver its committed (open-PO) rate plus upside flex; a
    # cut supplier has no flex. Capacity is cumulative: a light month's unused
    # capacity carries forward (the supplier builds ahead against the rate)
    cap_rate = pa.comp_po_monthly[None, :] * cap_mult * np.where(
        cap_mult >= 1.0, 1 + SUPPLIER_UPSIDE_FLEX, 1.0)                             # (M, C)
    cum_cap = np.cumsum(cap_rate, axis=0)

    comp_ix_all = list(range(n_c))
    lt_weeks = pa.comp_lead_time * p["lead_time_mult"]
    lt_months = np.clip(np.ceil(lt_weeks / WEEKS_PER_MONTH).astype(int), 1, M)        # (C,)
    # open POs cover the lead-time window; a supplier cut applies to them at
    # once, an increase cannot arrive before the lead time
    month_ix = np.arange(M)
    in_window = month_ix[:, None] < lt_months[None, :]                              # (M, C)
    nominal = np.zeros((n_sims, M, n_c))
    nominal[:] = np.where(in_window, pa.comp_po_monthly[None, :] * np.minimum(cap_mult, 1.0),
                          0.0)[None]
    # non-cancellable buy-ahead: months of cover (FY-average plan usage)
    # ordered in a given month, arriving one lead time later; buyers net it
    # out of later orders but cannot cancel it
    committed = np.zeros((M, n_c))
    fy_usage = pa.demand_units[:12].mean(axis=0) @ pa.comp_usage.T               # (C,)
    for comp, (order_m, cover) in p["buy_ahead"].items():
        cols = comp_ix_all if comp == "__all__" else [pa.comp_names.index(comp)]
        for c in cols:
            a = int(order_m) + int(lt_months[c])
            if a < M:
                committed[a, c] += float(cover) * fy_usage[c]
    nominal += committed[None]

    # delay fraction rises with supply tightness, logistics disruption and lead time
    base_delay = np.clip((lt_weeks[None, None, :] - 8.0) / 100.0, 0.01, 0.25) * sh.receipt_delay
    delay_frac = np.clip(
        base_delay
        + 0.06 * np.clip(tight_shock, 0, None)[:, :, None] * pa.comp_alloc_risk[None, None, :] * 2.0
        + 0.03 * np.clip(logistics_shock, 0, None)[:, :, None]
        + (p["lead_time_mult"] - 1.0) * 0.25,
        0.0, 0.6,
    )
    disrupt_p = np.clip(pa.comp_disrupt[None, None, :] * p["comp_disrupt_mult"]
                        * sh.component_disruption
                        * (1 + 0.8 * np.clip(tight_shock, 0, None))[:, :, None], 0, 0.5)
    disrupted = rng.random((n_sims, M, n_c)) < disrupt_p
    # an alternate source has its own disruption events (same odds, drawn
    # independently); a separate stream keeps every other draw unchanged
    alt_rng = np.random.default_rng([seed, 1])
    alt_disrupted = alt_rng.random((n_sims, M, n_c)) < disrupt_p
    alt_share = np.zeros((M, n_c))
    for comp, (start_m, share) in p["dual_source"].items():
        cols = comp_ix_all if comp == "__all__" else [pa.comp_names.index(comp)]
        alt_share[int(start_m):, cols] = float(share)
    # share of a month's receipts that arrives: a disrupted source delivers 35%
    delivered = ((1 - alt_share)[None] * np.where(disrupted, 0.35, 1.0)
                 + alt_share[None] * np.where(alt_disrupted, 0.35, 1.0))    # (n, M, C)

    # Expediting recovers part of delayed receipts at a premium, but only
    # where the receipts are NEEDED: units are expedited up to the projected
    # cumulative shortfall against the component requirement of realized
    # demand to date. The recovery fractions cap how much of the delayed
    # pool CAN be recovered; need caps how much IS. (2026-08 audit: a
    # volume-based premium on every late PO priced the standing policy as
    # ~$28M/yr of waste.)
    recovery_vec = np.full(n_c, float(p["expedite_recovery"]))
    for comp, frac in p["expedite_recovery_by_comp"].items():
        recovery_vec[pa.comp_names.index(comp)] = float(frac)
    recovery = recovery_vec * np.where(pa.comp_expedite_ok, 1.0, 0.0)   # (C,)
    # the safety-stock policy is the buffer target buyers order toward, and
    # the buffer is fully usable when parts run short (that is its job)
    ss_target = pa.comp_safety * p["safety_stock_mult"]
    start_avail = pa.comp_on_hand.copy()                                # (C,)
    cum_req = np.cumsum(np.einsum("nmf,cf->nmc", demand, pa.comp_usage), axis=1)
    plan_cum = np.vstack([np.zeros((1, n_f)), np.cumsum(pa.demand_units, axis=0)])  # (M+1, F)

    received = np.zeros((n_sims, M, n_c))
    recovered = np.zeros((n_sims, M, n_c))
    comp_avail = np.zeros((n_sims, M, n_c))         # usable cumulative supply
    cum_supply = np.tile(start_avail[None, :], (n_sims, 1))             # (n, C)
    cum_received = np.zeros((n_sims, n_c))
    pending_delay = np.zeros((n_sims, n_c))         # slipped receipts arriving next month
    comp_ix = np.arange(n_c)

    # ------------------------------------------------------------------
    # 3. EMS capacity (final integration and test happen at the EMS)
    # ------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.45, "Simulating EMS capacity")
    ems_raw = engine.shock("EMS execution", factors, rng)                # (n, M)
    ems_shock = ems_raw * sh.ems_execution
    # scale the sigma, not the shock: a mean-one lognormal with the shock at
    # zero but sigma on would still derate capacity by exp(-sigma^2 / 2)
    labor_mult = np.clip(_lognormal_mult(ems_raw, unc.ems_labor_sigma * sh.ems_execution),
                         0.7, 1.1)

    # each site runs at its own scheduled adherence (EMS scorecard input);
    # overtime capacity is derated by the same adherence as base capacity
    adherence = np.clip(pa.site_adherence + p["adherence_delta"], 0.0, 1.0)   # (S, M)
    site_cap = np.empty((n_sims, n_s, M))
    base_cap_total = np.zeros((n_sims, M))
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
        events = rng.random((n_sims, M)) < pa.site_disrupt_prob[s] * sh.site_disruption
        site_disrupted[site] = events.any(axis=1)
        cap = cap * np.where(events, 1 - unc.site_disruption_impact, 1.0)
        cap = cap * labor_mult
        base_cap_total += np.clip(cap, 0, None) * adherence[s][None, :]
        ot_mask = np.ones(M)
        ot_mask[:int(p["overtime_start_month"])] = 0.0
        cap = cap + p["overtime_fraction"] * pa.site_overtime[s][None, :] * ot_mask[None, :]
        site_cap[:, s, :] = np.clip(cap, 0, None) * adherence[s][None, :]

    # first-pass yield: a unit failing first pass is reworked, taking
    # REWORK_SHARE of a build slot, so each good unit loads the EMS by
    # (1 + REWORK_SHARE x (1 - FPY)) and costs rework
    fpy_base = float((pa.site_fpy * pa.site_capacity).sum() / pa.site_capacity.sum())
    fpy_eff = np.clip(fpy_base + p["fpy_delta"]
                      - 0.02 * np.clip(-ems_shock, 0, None), 0.7, 0.99)
    load_per_unit = pa.family_complexity[None, None, :] \
        * (1 + REWORK_SHARE * (1 - fpy_eff))[:, :, None]                # (n, M, F) std-units

    # ------------------------------------------------------------------
    # 4. Monthly shipment loop (vectorized across sims and families)
    # ------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.6, "Allocating supply against demand")
    # site-family qualification may change mid-horizon (new-site qualification)
    qual_by_month = np.tile(pa.site_qual[None, :, :], (M, 1, 1))
    for q_site, q_family, q_start in p["add_qualification"]:
        qual_by_month[q_start:, pa.site_names.index(q_site), fams.index(q_family)] = 1.0

    # fill least-contested sites first (fewest qualified families), then the
    # cheapest, so flexible multi-family sites stay available for families
    # with no alternative
    site_order = [np.lexsort((pa.site_cost, qual_by_month[m].sum(axis=1)))
                  for m in range(M)]

    # allocation: slices are (line, backlog) and (line, forecast). The policy
    # orders them into tiers; each tier is rationed pro-rata within a family
    # against what earlier tiers left
    tiers = allocation_tiers(p["allocation_policy"], pa, int(p["protect_top_n"]))
    slice_fam = np.r_[lfam, lfam]
    slice_onehot = np.r_[fam_onehot, fam_onehot]                        # (K, F)
    tier_onehot = [slice_onehot[t] for t in tiers]

    # shorted orders wait as backlog unless their customer loses them after N
    # months; ages are tracked in cohorts only when some customer has a limit
    lost_after = {**config.customers.lost_after_months, **p["lost_after_months"]}
    unknown = set(lost_after) - set(pa.customers)
    if unknown:
        raise KeyError(f"lost_after_months names unknown customers: {sorted(unknown)}")
    never = M + 1
    line_limit = np.array([lost_after.get(pa.customers[c], never) for c in lcust])
    finite = line_limit[line_limit < never]
    n_age = int(min(finite.max(), M)) + 2 if finite.size else 1
    ages = np.arange(1, n_age + 1)[:, None]                             # cohort a holds age a+1
    lose = ages > line_limit[None, :]                                   # (A, L)

    ship_l = np.zeros((n_sims, M, n_l))
    carry = np.zeros((n_age, n_sims, n_l))          # unmet by age cohort
    lost_l = np.zeros((n_sims, M, n_l))
    late_um_l = np.zeros((n_sims, n_l))             # FY late unit-months
    backlog_path = np.zeros((n_sims, M, n_f))      # end-of-month unmet (past-due)
    cum_consumed = np.zeros((n_sims, n_c))
    comp_short = np.zeros((n_sims, M))
    cap_short = np.zeros((n_sims, M))
    comp_binding_count = np.zeros((n_sims, n_c))
    site_load = np.zeros((n_sims, n_s, M))         # std-units actually built
    limit_units = np.zeros((2, n_sims, M, n_f))    # cut by component / EMS capacity
    binding_comp = np.full((n_sims, M, n_f), -1)   # component that set each family's ceiling
    usage = pa.comp_usage                                              # (C, F)
    fam_uses = [usage[:, f] > 0 for f in range(n_f)]

    for m in range(M):
        # receipts: this month's nominal less slippage, plus last month's slip;
        # a disrupted supplier delivers 35%; expedite recovers what is needed
        nom = nominal[:, m]
        slip = nom * delay_frac[:, m]
        rec_m = (nom - slip + pending_delay) * delivered[:, m]
        pool = slip + nom * (1.0 - delivered[:, m])
        pending_delay = slip
        cum_supply += rec_m
        need = np.clip(cum_req[:, m] - cum_supply, 0, None)
        recov = np.minimum(pool * recovery[None, :], need)
        cum_supply += recov
        received[:, m] = rec_m + recov
        recovered[:, m] = recov
        cum_received += received[:, m]
        comp_avail[:, m] = cum_supply

        firm_new = firm[:, m]
        want_k = np.concatenate([firm_new + carry.sum(axis=0), fcst[:, m]], axis=1)
        alloc_k = np.zeros_like(want_k)
        comp_res = np.clip(comp_avail[:, m, :] - cum_consumed, 0, None)  # (n, C)
        site_rem = site_cap[:, :, m].copy()                             # (n, S)
        binding_m = np.zeros((n_sims, n_c), bool)

        for idx, oh in zip(tiers, tier_onehot):
            w = want_k[:, idx]
            if not w.any():
                continue
            fam_w = w @ oh                                              # (n, F)
            # components: proportional scale per component, family takes its worst
            req = fam_w @ usage.T                                       # (n, C)
            with np.errstate(divide="ignore", invalid="ignore"):
                scale_c = np.where(req > 1e-9, np.minimum(1.0, comp_res / req), 1.0)
            fam_scale = np.ones((n_sims, n_f))
            for f in range(n_f):
                used = fam_uses[f]
                if used.any():
                    sc = np.where(used[None, :], scale_c, np.inf)
                    fam_scale[:, f] = sc.min(axis=1)
                    cut = (fam_scale[:, f] < 0.999) & (fam_w[:, f] > 1e-9)
                    binding_comp[:, m, f] = np.where(cut, sc.argmin(axis=1),
                                                     binding_comp[:, m, f])
            binding_m |= (scale_c < 0.999) & (req > 1e-9)
            after_comp = fam_w * fam_scale

            # EMS capacity: iterative water-filling. Each round, every site
            # (least-contested first) splits its remaining capacity across
            # qualified families in proportion to remaining std-equivalent
            # demand; three rounds recover nearly all slack one pass strands.
            after_cap = np.zeros((n_sims, n_f))
            for _ in range(3):
                unmet_std = np.clip(after_comp - after_cap, 0, None) * load_per_unit[:, m]
                if unmet_std.sum() < 1e-6:
                    break
                for s in site_order[m]:
                    q_std = unmet_std * qual_by_month[m, s][None, :]
                    denom = q_std.sum(axis=1, keepdims=True)
                    share = np.divide(q_std, denom, out=np.zeros_like(q_std),
                                      where=denom > 1e-9)
                    give_std = np.minimum(q_std, site_rem[:, s][:, None] * share)
                    after_cap += give_std / load_per_unit[:, m]
                    site_rem[:, s] -= give_std.sum(axis=1)
                    unmet_std = np.clip(unmet_std - give_std, 0, None)
            shipped_f = np.minimum(after_cap, after_comp)

            comp_res -= shipped_f @ usage.T
            ratio = np.divide(shipped_f, fam_w, out=np.zeros_like(fam_w), where=fam_w > 1e-12)
            alloc_k[:, idx] = w * ratio[:, slice_fam[idx]]
            limit_units[0, :, m] += fam_w - after_comp
            limit_units[1, :, m] += after_comp - shipped_f

        comp_binding_count += binding_m
        ship_firm, ship_fcst = alloc_k[:, :n_l], alloc_k[:, n_l:]
        ship_l[:, m] = ship_firm + ship_fcst
        shipped = ship_l[:, m] @ fam_onehot
        cum_consumed += shipped @ usage.T
        site_load[:, :, m] = site_cap[:, :, m] - site_rem

        # backlog: shipments serve the oldest orders first; this month's
        # unserved orders become age-1 backlog; over-age backlog is lost
        rem = ship_firm.copy()
        for a in range(n_age - 1, -1, -1):
            take = np.minimum(carry[a], rem)
            carry[a] -= take
            rem -= take
        unmet_new = np.clip((firm_new - rem) + (fcst[:, m] - ship_fcst), 0, None)
        if n_age > 1:
            oldest = carry[-1] + carry[-2]      # the last cohort holds every older age
            carry[1:-1] = carry[:-2].copy()
            carry[-1] = oldest
            carry[0] = unmet_new
        else:
            carry[0] += unmet_new
        lost_now = (carry * lose[:, None, :]).sum(axis=0)
        carry *= ~lose[:, None, :]
        lost_l[:, m] = lost_now

        backlog_line = carry.sum(axis=0)
        backlog_path[:, m] = backlog_line @ fam_onehot
        if m < 12:
            late_um_l += backlog_line
        comp_short[:, m] = limit_units[0, :, m].sum(axis=1)
        cap_short[:, m] = limit_units[1, :, m].sum(axis=1)

        # buyers: order up to forecast need over the lead time, the backlog's
        # parts and the safety-stock target, less stock and what is on order.
        # The forecast is the plan scaled by how demand is running (trailing
        # three months vs plan). Orders arrive one lead time later, within
        # supplier capacity.
        lo = max(0, m - 2)
        plan_recent = pa.demand_units[lo:m + 1].sum(axis=0)[None, :]
        run_rate = np.clip(np.divide(demand[:, lo:m + 1].sum(axis=1), plan_recent,
                                     out=np.ones((n_sims, n_f)), where=plan_recent > 1e-9),
                           0.5, 2.0)
        if m == 11:
            fy_end_run_rate = run_rate
        arrive = m + lt_months
        live = arrive < M
        if live.any():
            # plan units for months m+1 .. arrival month, inclusive
            window = plan_cum[np.minimum(arrive + 1, M)] - plan_cum[m + 1]   # (C, F)
            # einsum, not @: identical paths must give bit-identical orders
            fcst_need = np.einsum("nf,cf->nc", run_rate, usage * window)  # (n, C)
            backlog_need = np.einsum("nf,cf->nc", backlog_path[:, m], usage)
            stock = np.clip(pa.comp_on_hand[None, :] + cum_received - cum_consumed, 0, None)
            due = (month_ix[None, :] > m) & (month_ix[None, :] < arrive[:, None])  # (C, M)
            on_order = pending_delay + np.einsum("nmc,cm->nc", nominal, due.astype(float))
            # a committed buy-ahead landing in the arrival month is on order too
            due_commit = committed[np.minimum(arrive, M - 1), comp_ix] * live
            order = np.clip(fcst_need + backlog_need + ss_target[None, :] - stock - on_order
                            - due_commit[None, :], 0, None)
            c_live, a_live = comp_ix[live], arrive[live]
            before = (month_ix[None, :] < arrive[:, None]).astype(float)      # (C, M)
            delivered_before = np.einsum("nmc,cm->nc", nominal, before)
            headroom = np.clip(cum_cap[a_live, c_live][None, :] - delivered_before[:, live]
                               - committed[a_live, c_live][None, :], 0, None)
            # round to a millionth of a unit: the purchasing loop feeds back,
            # so last-digit float noise must not grow into path differences
            nominal[:, a_live, c_live] = committed[a_live, c_live][None, :] + np.round(
                np.minimum(order[:, live], headroom), 6)

    ship = ship_l @ fam_onehot                                         # (n, M, F)
    expedite_cost_comp = (recovered * pa.comp_cost[None, None, :]
                          * pa.comp_expedite_prem[None, None, :]
                          * p["expedite_premium_mult"]).sum(axis=2)   # (n, M)
    ems_load_std = (ship * load_per_unit).sum(axis=2)                  # incl. rework slots
    total_cap = site_cap.sum(axis=1)
    ems_util = ems_load_std / np.clip(total_cap, 1e-9, None)

    # ------------------------------------------------------------------
    # 5. Revenue recognition by line (acceptance / site-readiness slip)
    # ------------------------------------------------------------------
    # calibrated slip odds scale with the shock; a lever's added slip stays
    accept_delay_p = np.clip(
        (pa.family_accept_prob_delay[lfam] + (1 - pa.line_readiness))
        * sh.acceptance_slip + p["acceptance_delay_add"], 0, 0.9)      # (L,)
    slip_noise = np.clip(rng.beta(4, 6, size=(n_sims, 1, 1)) * 2.0, 0.3, 1.7)
    slip_noise = 1.0 + sh.acceptance_slip * (slip_noise - 1.0)
    sf = np.clip(accept_delay_p[None, None, :] * slip_noise, 0, 0.9)   # (n, 1, L)

    # one-month recognition lag for acceptance-based families; the first
    # month recognizes systems finished before the horizon at the month-1 rate
    lagged = np.empty_like(ship_l)
    lagged[:, 1:] = ship_l[:, :-1]
    lagged[:, :1] = ship_l[:, :1]
    base_rec = np.where((pa.family_rec_lag[lfam] >= 1)[None, None, :], lagged, ship_l)
    rec_l = base_rec * (1 - sf)
    rec_l[:, 1:] += (base_rec * sf)[:, :-1]
    # steady-state inflow from systems that slipped acceptance before the horizon
    rec_l[:, 0] += base_rec[:, 0] * sf[:, 0]
    rec = rec_l @ fam_onehot                                           # (n, M, F)

    # ------------------------------------------------------------------
    # 6. Financial translation
    # ------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.8, "Translating to financial outcomes")
    # revenue per line at the customer's ASP, with a price shock per customer
    asp_shock = rng.standard_normal((n_sims, 1, n_cu))
    pc = sh.price_cost
    asp_mult = _lognormal_mult(asp_shock, unc.asp_sigma * pc)[:, :, lcust] * p["asp_mult"]
    revenue_l = rec_l * pa.line_asp[None, None, :] * asp_mult          # (n, M, L)
    revenue_fm = revenue_l @ fam_onehot
    revenue = revenue_fm.sum(axis=2)

    ppv_mult = _lognormal_mult(rng.standard_normal((n_sims, 1)), unc.material_cost_sigma * pc)
    fx_mult = _lognormal_mult(rng.standard_normal((n_sims, 1)), unc.fx_cost_sigma * pc)
    tight_cost = 1 + 0.02 * np.clip(tight_shock, 0, None)              # tight supply raises cost
    material_mult = ppv_mult * fx_mult * tight_cost * p["material_cost_mult"]  # (n, M)
    conv_mult = _lognormal_mult(rng.standard_normal((n_sims, 1)), unc.conversion_cost_sigma * pc)
    freight_mult = (_lognormal_mult(rng.standard_normal((n_sims, 1)), unc.freight_sigma * pc)
                    * p["freight_mult"]
                    * (1 + 0.08 * np.clip(logistics_shock, 0, None)))

    # one unit-cost policy: the same standard build-up values COGS and FG
    uc = standard_unit_cost(data, fams, fpy_base)
    mat_c, conv_c, integ_c = uc["material"], uc["conversion"], uc["integration"]
    freight_c, warr_c, scrap_c = uc["freight"], uc["warranty"], uc["scrap"]
    prod = data.products.set_index("product_family").loc[fams]

    cogs = (rec * mat_c[None, None, :]).sum(axis=2) * material_mult \
        + (rec * conv_c[None, None, :]).sum(axis=2) * conv_mult \
        + (rec * integ_c[None, None, :]).sum(axis=2) \
        + (rec * freight_c[None, None, :]).sum(axis=2) * freight_mult \
        + (rec * (warr_c + scrap_c)[None, None, :]).sum(axis=2)

    # standard cost per recognized unit by line: customer contribution at standard
    unit_cost_l = (mat_c[lfam][None, None, :] * material_mult[:, :, None]
                   + conv_c[lfam][None, None, :] * conv_mult[:, :, None]
                   + (integ_c + warr_c + scrap_c)[lfam][None, None, :]
                   + freight_c[lfam][None, None, :] * freight_mult[:, :, None])

    # rework: units failing first pass are reworked at half conversion cost
    rework_cost = (ship * (REWORK_SHARE * conv_c)[None, None, :]).sum(axis=2) * (1 - fpy_eff)
    # overtime conversion premium: std-units produced above base capacity carry
    # the weighted-average overtime premium on EMS conversion cost
    if p["overtime_fraction"] > 0:
        ot_used_std = np.clip(ems_load_std - base_cap_total, 0, None)
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

    # inventory: critical-component RM + non-critical RM + WIP + FG awaiting acceptance
    # critical stock is valued as physically held: on-hand plus receipts less
    # consumption
    cum_consumed_path = _cum_consumed_path(ship, usage)                # (n, M, C)
    stock_path = np.clip(pa.comp_on_hand[None, None, :] + np.cumsum(received, axis=1)
                         - cum_consumed_path, 0, None)                 # (n, M, C)
    # buyers respond to demand (section 4), so stock is valued as held
    rm_crit = (stock_path * pa.comp_cost[None, None, :]).sum(axis=2)
    mat_spend = (ship * mat_c[None, None, :]).sum(axis=2)
    rm = rm_crit + 0.9 * mat_spend
    cycle = prod["build_cycle_months"].to_numpy(float)
    wip = (ship * ((mat_c + conv_c) * cycle * 0.6)[None, None, :]).sum(axis=2)
    fg = np.clip(np.cumsum(ship - rec, axis=1), 0, None) @ uc["standard"]
    inventory = rm + wip + fg

    # E&O: excess critical-component stock at FY end above 2.5 months of the
    # usage buyers expect (next quarter's plan at the demand run rate). Each
    # part's excess is reserved at the policy rate plus its obsolescence
    # probability on the remainder (risk 0: policy rate; risk 1: written
    # off), plus 5% of aged finished goods
    fwd_plan = pa.demand_units[12:15].mean(axis=0)                    # (F,)
    fwd_usage = np.einsum("nf,cf->nc", fy_end_run_rate * fwd_plan[None, :], usage)
    excess_rm = np.clip(stock_path[:, 11, :] - 2.5 * fwd_usage, 0, None)
    eo_rate = fin.eo_reserve_rate + pa.comp_obsolescence * (1 - fin.eo_reserve_rate)  # (C,)
    eo_reserve = (excess_rm * pa.comp_cost[None, :] * eo_rate[None, :]).sum(axis=1) \
        + fg[:, 11] * 0.05
    # the reserve is a P&L charge: the FY provision (year-end reserve less the
    # opening reserve on today's stock) hits COGS in the FY's last month, and
    # inventory is carried net of the reserve (non-cash, so cash is unchanged)
    open_usage = pa.demand_units[0:3].mean(axis=0) @ usage.T              # (C,)
    open_excess = np.clip(pa.comp_on_hand - 2.5 * open_usage, 0, None)
    eo_opening = float((open_excess * pa.comp_cost * eo_rate).sum())
    eo_rm_end = (excess_rm * pa.comp_cost[None, :] * eo_rate[None, :]).sum(axis=1)
    eo_provision = eo_reserve - eo_opening                               # (n,)
    cogs[:, 11] += eo_provision
    rm[:, :11] -= eo_opening
    rm[:, 11:] -= eo_rm_end[:, None]
    fg[:, 11:] -= (fg[:, 11] * 0.05)[:, None]
    inventory = rm + wip + fg

    gross_profit = revenue - cogs
    action_cost_m = np.zeros(M)
    action_cost_m[:3] = p["action_cost_usd"] / 3.0
    # permanent actions (take-or-pay, headcount) cost money every month held
    for start_m, usd in p["recurring_cost"].values():
        action_cost_m[int(start_m):] += float(usd)
    operating_income = gross_profit - fin.opex_monthly_usd - action_cost_m[None, :]
    ebitda = operating_income + fin.depreciation_monthly_usd

    ar = revenue * fin.dso_days / 30.0
    # payables follow purchases: critical parts as received, other material
    # as consumed, EMS conversion and freight as billed on builds
    crit_bought = (received * pa.comp_cost[None, None, :]).sum(axis=2)
    crit_used = (np.diff(cum_consumed_path, axis=1, prepend=0.0)
                 * pa.comp_cost[None, None, :]).sum(axis=2)
    purchases = (crit_bought + np.clip(mat_spend * material_mult - crit_used, 0, None)
                 + (ship * conv_c[None, None, :]).sum(axis=2) * conv_mult
                 + (ship * freight_c[None, None, :]).sum(axis=2) * freight_mult)
    ap = purchases * fin.dpo_days / 30.0
    working_capital = inventory + ar - ap
    dwc = np.diff(working_capital, axis=1, prepend=working_capital[:, :1])
    cash_flow = ebitda - dwc - fin.capex_monthly_usd \
        - np.clip(operating_income, 0, None) * fin.tax_rate

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
        "Acceptance slip fraction": sf[:, 0, :].mean(axis=1),
    }

    comp_binding = {name: comp_binding_count[:, i] > 0
                    for i, name in enumerate(pa.comp_names)}

    return SimulationResult(
        n_sims=n_sims, seed=seed, scenario_name=scenario_name,
        revenue=revenue, cogs=cogs, gross_profit=gross_profit,
        operating_income=operating_income, ebitda=ebitda, cash_flow=cash_flow,
        inventory=inventory, raw_inventory=rm, wip_inventory=wip, fg_inventory=fg,
        working_capital=working_capital, expedite_cost=expedite_cost,
        rework_cost=rework_cost, eo_reserve=eo_reserve, eo_provision=eo_provision,
        family_revenue=revenue_fm, family_units=rec,
        family_shipped=ship, family_demand=demand,
        units_shipped=ship.sum(axis=2), units_demanded=demand.sum(axis=2),
        ems_utilization=ems_util,
        capacity_shortfall_units=cap_short, component_short_units=comp_short,
        component_binding=comp_binding, site_disrupted=site_disrupted,
        drivers=drivers, params=p,
        customers=list(pa.customers),
        customer_revenue=revenue_l @ cust_onehot,
        customer_gross_profit=((revenue_l - rec_l * unit_cost_l) @ cust_onehot).astype(np.float32),
        customer_shipped=(ship_l @ cust_onehot).astype(np.float32),
        customer_demand=((firm + fcst) @ cust_onehot).astype(np.float32),
        customer_lost_revenue=((lost_l * pa.line_asp[None, None, :]) @ cust_onehot
                               ).astype(np.float32),
        customer_late_unit_months=(late_um_l @ cust_onehot).astype(np.float32),
        family_lost=lost_l @ fam_onehot,
        family_backlog=backlog_path, site_load=site_load, site_capacity=site_cap,
        limit_units=limit_units,
        binding_component=binding_comp,
        component_consumed=(np.einsum("nmf,cf->nmc", ship, usage)
                            if keep_component_paths else None),
        component_usable_supply=comp_avail if keep_component_paths else None,
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
