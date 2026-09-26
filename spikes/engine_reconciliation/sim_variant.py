"""Spike copy of src/simulation.py::run_simulation with switchable structural differences.

Every switch defaults to the current src behavior (AS_IS), so
`run_variant(..., flags=AS_IS)` reproduces src.simulation.run_simulation
bit-for-bit (checked in check_equivalence.py). Each flag neutralizes one
structural difference against src/baseline_plan.py::run_baseline. Line refs
are to src/simulation.py. Nothing in src/ is modified.

The random-number call order is preserved in every variant, so switching a
flag never reshuffles the draws of unrelated variables.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.correlations import FactorEngine
from src.operations import build_planning_arrays
from src.simulation import _cum_consumed_path, _lognormal_mult, default_params
from src.utils import FAMILY_MARKET, N_MONTHS

from greedy import greedy_allocate

AS_IS: dict[str, Any] = {
    "factor_shocks": True,       # L157-158, L231: tightness/logistics/EMS shocks not scaled by any sigma
    "util_penalty": True,        # L264-272: adherence penalty when unconstrained demand/cap > 0.92
    "fpy_util_penalty": True,    # L283-285: FPY penalty (cost only, via rework)
    "lead_time_delay": True,     # L174: base delay (LT-8)/100 clipped [0.01, 0.25]
    "safety_floor_frac": 0.3,    # L207: 30% of safety stock is a hard floor (baseline: 0)
    "adherence": "site_mean",    # L267: unweighted mean adherence across sites (baseline: per site)
    "allocator": "proportional",  # L305-355 (baseline: greedy customer priority)
    "greedy_floor": True,        # baseline_plan.py L99/L103/L119 integer floors
    "cogs": "sim",               # L415-439 (baseline: baseline_plan.py L170-175)
    "inv_damping": True,         # L452-454: purchasing response damps RM valuation
    "inv_floor_excluded": True,  # L449: RM valuation omits the safety floor still on the shelf
    "fg_valuation": "sim",       # L460 (baseline unit_cogs includes rework + scrap)
    "util_penalty_cap": None,    # cap the utilization driver (e.g. 1.0: load cannot exceed capacity)
    "waterfill_rounds": 3,       # L329: rounds of proportional EMS water-filling
    "greedy_demand_rows": None,  # optional re-keyed demand queue for the greedy allocator
}


def run_variant(data, config, params: dict | None = None, n_sims: int = 1,
                seed: int = 42, flags: dict | None = None) -> dict:
    fl = dict(AS_IS)
    if flags:
        unknown = set(flags) - set(AS_IS)
        if unknown:
            raise KeyError(f"unknown flags {unknown}")
        fl.update(flags)
    p = default_params()
    if params:
        p.update(params)

    pa = build_planning_arrays(data)
    unc = config.uncertainty
    engine = FactorEngine(config.factors)
    rng = np.random.default_rng(seed)
    fams = pa.families
    n_f, n_c, n_s, M = len(fams), len(pa.comp_names), len(pa.site_names), N_MONTHS

    factors = engine.draw_factor_paths(rng, n_sims, M)

    # 1. demand (unchanged)
    market_shock = {mkt: engine.shock(mkt, factors, rng) for mkt in unc.market_demand_sigma}
    sigma_mult = float(p["demand_sigma_mult"])
    demand = np.empty((n_sims, M, n_f))
    for f, fam in enumerate(fams):
        mkt = FAMILY_MARKET[fam]
        mult = _lognormal_mult(market_shock[mkt], unc.market_demand_sigma[mkt] * sigma_mult)
        idio = _lognormal_mult(rng.standard_normal((n_sims, 1)),
                               unc.customer_idiosyncratic_sigma * sigma_mult)
        scen = p["demand_market_mult"].get(mkt, 1.0) * p["demand_family_mult"].get(fam, 1.0)
        scen_path = np.asarray(scen, dtype=float)
        if scen_path.ndim == 0:
            scen_path = np.full(M, float(scen_path))
        demand[:, :, f] = pa.demand_units[None, :, f] * mult * idio * scen_path[None, :]
    cancel_p = np.clip(pa.cancel_prob[None, None, :] * p["cancel_prob_mult"], 0, 1)
    push_p = np.clip(pa.pushout_prob[None, None, :] + p["pushout_prob_add"], 0, 1)
    pull_p = np.clip(pa.pullin_prob[None, None, :] + p["pullin_prob_add"], 0, 1)
    timing_noise = _lognormal_mult(rng.standard_normal((n_sims, 1, 1)), 0.35)
    push_p = np.clip(push_p * timing_noise, 0, 0.6)
    pull_p = np.clip(pull_p * timing_noise ** 0.5, 0, 0.3)
    demand = demand * (1 - cancel_p)
    pushed = demand * push_p
    pulled = demand * pull_p
    demand = demand - pushed - pulled
    demand[:, 1:, :] += 0.7 * pushed[:, :-1, :]
    demand[:, 2:, :] += 0.3 * pushed[:, :-2, :]
    demand[:, 0, :] += 0.7 * pushed[:, 0, :]
    demand[:, 1, :] += 0.3 * pushed[:, 0, :]
    demand[:, :-1, :] += pulled[:, 1:, :]
    demand[:, 0, :] += pulled[:, 0, :]
    if p["forced_pushout"]:
        fp = p["forced_pushout"]
        f = fams.index(fp["family"])
        units = np.minimum(demand[:, fp["from_month"], f], fp["units"])
        demand[:, fp["from_month"], f] -= units
        demand[:, fp["to_month"], f] += units
    demand = np.clip(demand, 0, None)

    # 2. component supply
    tight_shock = engine.shock("Component tightness", factors, rng)
    logistics_shock = engine.shock("Logistics disruption", factors, rng)
    if not fl["factor_shocks"]:
        tight_shock = np.zeros_like(tight_shock)
        logistics_shock = np.zeros_like(logistics_shock)
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
    lt_weeks = pa.comp_lead_time[None, None, :] * p["lead_time_mult"]
    if fl["lead_time_delay"]:
        base_delay = np.clip((lt_weeks - 8.0) / 100.0, 0.01, 0.25)
    else:
        base_delay = np.zeros_like(lt_weeks)
    delay_frac = np.clip(
        base_delay
        + 0.06 * np.clip(tight_shock, 0, None)[:, :, None] * pa.comp_alloc_risk[None, None, :] * 2.0
        + 0.03 * np.clip(logistics_shock, 0, None)[:, :, None]
        + (p["lead_time_mult"] - 1.0) * 0.25,
        0.0, 0.6)
    disrupt_p = np.clip(pa.comp_disrupt[None, None, :] * p["comp_disrupt_mult"]
                        * (1 + 0.8 * np.clip(tight_shock, 0, None))[:, :, None], 0, 0.5)
    disrupted = rng.random((n_sims, M, n_c)) < disrupt_p
    received = receipts_nominal * (1 - delay_frac)
    received[:, 1:, :] += (receipts_nominal * delay_frac)[:, :-1, :]
    received = received * np.where(disrupted, 0.35, 1.0)
    delayed_pool = receipts_nominal * delay_frac + receipts_nominal * np.where(disrupted, 0.65, 0.0)
    recovery_vec = np.full(n_c, float(p["expedite_recovery"]))
    for comp, frac in p["expedite_recovery_by_comp"].items():
        recovery_vec[pa.comp_names.index(comp)] = float(frac)
    recovery = recovery_vec * np.where(pa.comp_expedite_ok, 1.0, 0.0)
    safety_floor = pa.comp_safety * p["safety_stock_mult"] * fl["safety_floor_frac"]
    start_avail = np.clip(pa.comp_on_hand - safety_floor, 0, None)
    cum_req = np.cumsum(np.einsum("nmf,cf->nmc", demand, pa.comp_usage), axis=1)
    recovered = np.zeros_like(received)
    cum_supply = np.tile(start_avail[None, :], (n_sims, 1))
    for m in range(M):
        cum_supply = cum_supply + received[:, m, :]
        need = np.clip(cum_req[:, m, :] - cum_supply, 0, None)
        recovered[:, m, :] = np.minimum(delayed_pool[:, m, :] * recovery[None, :], need)
        cum_supply = cum_supply + recovered[:, m, :]
    received = received + recovered
    expedite_cost_comp = (recovered * pa.comp_cost[None, None, :]
                          * pa.comp_expedite_prem[None, None, :]
                          * p["expedite_premium_mult"]).sum(axis=2)
    comp_avail = np.clip(pa.comp_on_hand[None, None, :] - safety_floor[None, None, :], 0, None) \
        + np.cumsum(received, axis=1)

    # 3. EMS + integration capacity
    ems_shock = engine.shock("EMS execution", factors, rng)
    if not fl["factor_shocks"]:
        ems_shock = np.zeros_like(ems_shock)
    labor_mult = np.clip(_lognormal_mult(ems_shock, unc.ems_labor_sigma), 0.7, 1.1)
    site_cap = np.empty((n_sims, n_s, M))
    raw_cap = np.empty((n_sims, n_s, M))   # before any adherence (penalty denominator)
    site_disrupted = {}
    for s, site in enumerate(pa.site_names):
        base = pa.site_capacity[s] * pa.site_labor[s]
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
        raw_cap[:, s, :] = site_cap[:, s, :]
        if s == 0:
            base_cap_total = cap_no_ot.copy()
        else:
            base_cap_total += cap_no_ot
    demand_std = (demand * pa.family_complexity[None, None, :]).sum(axis=2)
    total_cap_raw = raw_cap.sum(axis=1)
    util_prelim = demand_std / np.clip(total_cap_raw, 1e-9, None)
    adherence_base = (pa.site_adherence.mean(axis=0)[None, :] + p["adherence_delta"])
    util_drv = util_prelim if fl["util_penalty_cap"] is None else np.minimum(util_prelim, fl["util_penalty_cap"])
    pen = unc.utilization_adherence_penalty * np.clip(util_drv - 0.92, 0, None) / 0.10
    if not fl["util_penalty"]:
        pen = np.zeros_like(pen)
    if fl["adherence"] == "site_mean":
        adherence_eff = np.clip(adherence_base - pen, 0.6, 1.0)
        site_cap = site_cap * adherence_eff[:, None, :]
    else:  # per-site adherence, as effective_site_capacity() in operations.py L171
        site_adh = np.clip(pa.site_adherence[None, :, :] + p["adherence_delta"]
                           - pen[:, None, :], 0.6, 1.0)
        site_cap = site_cap * site_adh
        adherence_eff = np.clip(adherence_base - pen, 0.6, 1.0)
    integ_mult_path = np.full(M, p["integration_capacity_mult"])
    if p["integration_capacity_ramp"]:
        ramp_start, ramp_mult = p["integration_capacity_ramp"]
        integ_mult_path[ramp_start:] *= ramp_mult
    integ_cap = (pa.integration_capacity[None, :] * integ_mult_path[None, :]
                 * np.clip(labor_mult, 0.8, 1.05))
    fpy_base = float((pa.site_fpy * pa.site_capacity).sum() / pa.site_capacity.sum())
    fpy_pen = 0.03 * np.clip(util_prelim - 0.9, 0, None) / 0.1
    if not fl["fpy_util_penalty"]:
        fpy_pen = np.zeros_like(fpy_pen)
    fpy_eff = np.clip(fpy_base + p["fpy_delta"] - fpy_pen
                      - 0.02 * np.clip(-ems_shock, 0, None), 0.7, 0.99)

    # 4. allocation
    qual_by_month = np.tile(pa.site_qual[None, :, :], (M, 1, 1))
    for q_site, q_family, q_start in p["add_qualification"]:
        qual_by_month[q_start:, pa.site_names.index(q_site), fams.index(q_family)] = 1.0
    ship = np.zeros((n_sims, M, n_f))
    comp_short = np.zeros((n_sims, M))
    cap_short = np.zeros((n_sims, M))
    usage = pa.comp_usage
    greedy_out = None
    if fl["allocator"] == "proportional":
        backlog = np.zeros((n_sims, n_f))
        cum_consumed = np.zeros((n_sims, n_c))
        for m in range(M):
            want = demand[:, m, :] + backlog
            avail = np.clip(comp_avail[:, m, :] - cum_consumed, 0, None)
            req = want @ usage.T
            with np.errstate(divide="ignore", invalid="ignore"):
                scale_c = np.where(req > 1e-9, np.minimum(1.0, avail / req), 1.0)
            fam_scale = np.ones((n_sims, n_f))
            for f in range(n_f):
                used = usage[:, f] > 0
                if used.any():
                    fam_scale[:, f] = scale_c[:, used].min(axis=1)
            after_comp = want * fam_scale
            after_cap = np.zeros((n_sims, n_f))
            site_rem = site_cap[:, :, m].copy()
            for _ in range(int(fl["waterfill_rounds"])):
                unmet_u = np.clip(after_comp - after_cap, 0, None)
                unmet_std = unmet_u * pa.family_complexity[None, :]
                if unmet_std.sum() < 1e-6:
                    break
                for s in range(n_s):
                    qual = qual_by_month[m, s]
                    q_std = unmet_std * qual[None, :]
                    denom = q_std.sum(axis=1, keepdims=True)
                    share = np.divide(q_std, denom, out=np.zeros_like(q_std), where=denom > 1e-9)
                    give_std = np.minimum(q_std, site_rem[:, s][:, None] * share)
                    after_cap += give_std / pa.family_complexity[None, :]
                    site_rem[:, s] -= give_std.sum(axis=1)
                    unmet_std = np.clip(unmet_std - give_std, 0, None)
            after_cap = np.minimum(after_cap, after_comp)
            total_after = after_cap.sum(axis=1)
            integ_scale = np.minimum(1.0, integ_cap[:, m] / np.clip(total_after, 1e-9, None))
            shipped = after_cap * integ_scale[:, None]
            ship[:, m, :] = shipped
            backlog = want - shipped
            cum_consumed += shipped @ usage.T
            comp_short[:, m] = (want - after_comp).sum(axis=1)
            cap_short[:, m] = (after_comp - shipped).sum(axis=1)
    else:
        # the baseline's greedy customer-priority allocator on the sim's own
        # capacity and supply arrays (zero-shock use: demand == demand plan rows)
        greedy_out = []
        for n in range(n_sims):
            g = greedy_allocate(data, pa, site_cap=site_cap[n], integ_cap=integ_cap[n],
                                comp_start=start_avail, comp_receipts=received[n],
                                use_floor=fl["greedy_floor"],
                                demand_rows=fl["greedy_demand_rows"])
            ship[n] = g["built"]
            greedy_out.append(g)

    ems_load_std = (ship * pa.family_complexity[None, None, :]).sum(axis=2)
    total_cap = site_cap.sum(axis=1)
    ems_util = ems_load_std / np.clip(total_cap, 1e-9, None)
    integ_util = ship.sum(axis=2) / np.clip(integ_cap, 1e-9, None)

    # 5. revenue recognition (unchanged)
    accept_delay_p = np.clip(pa.family_accept_prob_delay[None, :] + p["acceptance_delay_add"]
                             + (1 - pa.site_readiness[None, :]), 0, 0.9)
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
        rec_f[:, 0] += base_rec[:, 0] * sf[:, 0]
        rec[:, :, f] = rec_f

    # 6. financial translation
    asp_shock = rng.standard_normal((n_sims, 1, n_f))
    asp_mult = _lognormal_mult(asp_shock, unc.asp_sigma) * p["asp_mult"]
    revenue = (rec * pa.family_asp[None, None, :] * asp_mult).sum(axis=2)
    ppv_mult = _lognormal_mult(rng.standard_normal((n_sims, 1)), unc.material_cost_sigma)
    fx_mult = _lognormal_mult(rng.standard_normal((n_sims, 1)), unc.fx_cost_sigma)
    tight_cost = 1 + 0.02 * np.clip(tight_shock, 0, None)
    material_mult = ppv_mult * fx_mult * tight_cost * p["material_cost_mult"]
    conv_mult = _lognormal_mult(rng.standard_normal((n_sims, 1)), unc.conversion_cost_sigma)
    freight_mult = (_lognormal_mult(rng.standard_normal((n_sims, 1)), unc.freight_sigma)
                    * p["freight_mult"] * (1 + 0.08 * np.clip(logistics_shock, 0, None)))
    prod = data.products.set_index("product_family").loc[fams]
    mat_c = prod["material_cost_usd"].to_numpy(float)
    conv_c = prod["ems_conversion_cost_usd"].to_numpy(float)
    integ_c = prod["integration_test_cost_usd"].to_numpy(float)
    freight_c = prod["freight_cost_usd"].to_numpy(float)
    warr_c = prod["warranty_reserve_usd"].to_numpy(float)
    scrap_c = (prod["scrap_prob"] * prod["material_cost_usd"]).to_numpy(float)
    rework_bl = (prod["rework_prob"] * 0.5 * prod["ems_conversion_cost_usd"]).to_numpy(float)
    if fl["cogs"] == "sim":
        cogs = (rec * mat_c).sum(axis=2) * material_mult \
            + (rec * conv_c).sum(axis=2) * conv_mult \
            + (rec * integ_c).sum(axis=2) \
            + (rec * freight_c).sum(axis=2) * freight_mult \
            + (rec * (warr_c + scrap_c)).sum(axis=2)
        rework_cost = ship.sum(axis=2) * (1 - fpy_eff) * 0.5 * float(conv_c.mean())
        if p["overtime_fraction"] > 0:
            adj_base_cap = base_cap_total * adherence_eff
            ot_used_std = np.clip(ems_load_std - adj_base_cap, 0, None)
            ot_weights = pa.site_overtime.mean(axis=1)
            ot_weights = ot_weights / max(ot_weights.sum(), 1e-9)
            ot_premium = float((pa.site_cost * pa.site_ot_premium * ot_weights).sum())
            overtime_cost = ot_used_std * ot_premium
        else:
            overtime_cost = np.zeros_like(rework_cost)
        expedite_cost = expedite_cost_comp + comp_short * 0.02 * float(mat_c.mean()) \
            * p["expedite_premium_mult"]
        cogs = cogs + rework_cost + expedite_cost + overtime_cost
    else:  # baseline_plan.py L170-175: family rework_prob on recognized units, no expedite
        unit_cogs_bl = mat_c + conv_c + integ_c + freight_c + warr_c + rework_bl + scrap_c
        cogs = (rec * unit_cogs_bl).sum(axis=2)
        rework_cost = (rec * rework_bl).sum(axis=2)
        expedite_cost = np.zeros_like(cogs)
    gross_profit = revenue - cogs

    cum_consumed_path = _cum_consumed_path(ship, usage)
    stock_path = np.clip(comp_avail - cum_consumed_path, 0, None)
    if not fl["inv_floor_excluded"]:
        stock_path = stock_path + np.minimum(safety_floor, pa.comp_on_hand)[None, None, :]
    monthly_req_units = pa.comp_usage @ pa.demand_units.mean(axis=0)
    if fl["inv_damping"]:
        excess_path = np.clip(stock_path - 2.0 * monthly_req_units[None, None, :], 0, None)
        held_stock = stock_path - 0.5 * excess_path
    else:
        held_stock = stock_path
    rm = (held_stock * pa.comp_cost).sum(axis=2) + 0.9 * (ship * mat_c).sum(axis=2)
    cycle = prod["build_cycle_months"].to_numpy(float)
    wip = (ship * ((mat_c + conv_c) * cycle * 0.6)).sum(axis=2)
    if fl["fg_valuation"] == "sim":
        unit_fg = mat_c + conv_c + integ_c + freight_c + warr_c
    else:
        unit_fg = mat_c + conv_c + integ_c + freight_c + warr_c + rework_bl + scrap_c
    fg = np.clip(np.cumsum(ship - rec, axis=1), 0, None) @ unit_fg
    inventory = rm + wip + fg

    return {
        "revenue": revenue, "cogs": cogs, "gross_profit": gross_profit,
        "ship": ship, "rec": rec, "demand": demand, "inventory": inventory,
        "rm": rm, "wip": wip, "fg": fg, "site_cap": site_cap, "integ_cap": integ_cap,
        "ems_util": ems_util, "integ_util": integ_util, "adherence_eff": adherence_eff,
        "comp_short": comp_short, "cap_short": cap_short, "rework_cost": rework_cost,
        "expedite_cost": expedite_cost, "comp_avail": comp_avail, "greedy": greedy_out,
        "delay_frac": delay_frac,
    }
