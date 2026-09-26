"""Spike: vectorized Monte Carlo allocation kernel over demand lines.

Shape of the problem: (n_sims, months, demand_lines). Each line is a customer x
product family. Each month's demand is split into two slices per line:
  firm   = booked backlog + carried-over unmet + pushed-in orders
  fcst   = unbooked forecast (the only slice that takes volume shocks)
so the kernel allocates over K = 2L slices.

Scarce resources, rationed to slices each month:
  components  cumulative (unused supply carries forward), usage per family
  EMS sites   monthly std-equivalent capacity, qualification by family
  integration optional company-level monthly unit cap (may collapse into EMS)

Allocation policies (the lever):
  proportional     one pass, pro-rata within family (today's engine logic)
  priority_tiered  tiers (firm first) x customer priority; pro-rata within tier
  strict_greedy    exact sequential greedy in baseline order
                   (firm first, priority, contribution per std unit)
  margin_greedy    exact sequential greedy by contribution per std unit (firm first)
  margin_tiered    firm first, then contribution-per-std-unit quartile bands
  protect_top_n    top-N customers (all slices) first, then rest firm, rest fcst

"Tiered" policies run the proportional pass once per tier on residual capacity
(T passes). "Greedy" policies loop over slices in a static order, vectorized
across simulations (K passes of small ops). Both are exact for their stated rule.

Exploratory code only. Demand/supply shocks are simplified stand-ins for the
full engine's; the point is the shape, cost and behavior of the allocation.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.correlations import FactorEngine  # noqa: E402

from problem import Problem  # noqa: E402

POLICIES = ["proportional", "priority_tiered", "strict_greedy", "margin_greedy",
            "margin_tiered", "protect_top_n"]
FY = 12  # months 0-11 are the fiscal year


def _lognormal(z, sigma):
    return np.exp(sigma * z - 0.5 * sigma ** 2)


@dataclass
class SimOptions:
    shocks: bool = True
    cust_corr: float = 0.5          # share of customer shock variance from its group factor
    cust_sigma: float | None = None  # default: config customer_idiosyncratic_sigma
    fam_sigma: float = 0.05
    line_noise_sigma: float = 0.15  # month-to-month order lumpiness (forecast slice only)
    firm_cancel_mult: float = 0.5   # firm backlog cancels at half the forecast rate
    comp_stress: dict = field(default_factory=dict)  # comp index -> (start_month, receipts mult)
    use_integration: bool = True
    top_n: int = 3
    margin_bands: int = 4
    waterfill_rounds: int = 3
    waterfill_least_contested_first: bool = True  # False = site index order (as src/ today)
    dtype: str = "float64"          # "float32" halves memory and speeds up the kernel
    prop_passes: int = 1            # re-run each tier's pro-rata pass on residual demand;
                                    # >1 recovers capacity a single pass strands


@dataclass
class SimResult:
    policy: str
    n_sims: int
    ship_q: np.ndarray        # (n, 6, L) units shipped by quarter
    dem_q: np.ndarray         # (n, 6, L) units requested (new demand) by quarter
    rev_fy: np.ndarray        # (n, L) FY recognized revenue
    gm_fy: np.ndarray         # (n, L) FY gross margin dollars
    late_um: np.ndarray       # (n, L) late unit-months (sum of carried backlog)
    end_backlog: np.ndarray   # (n, L) unmet at horizon end
    t_total: float = 0.0
    t_alloc: float = 0.0


# ----------------------------------------------------------------------------
# Static precomputation
# ----------------------------------------------------------------------------

class _Static:
    def __init__(self, prob: Problem, opt: SimOptions):
        L = prob.n_lines
        F = len(prob.fam_names)
        self.L, self.K, self.F = L, 2 * L, F
        self.slice_line = np.concatenate([np.arange(L), np.arange(L)])
        self.slice_firm = np.concatenate([np.ones(L, bool), np.zeros(L, bool)])
        self.slice_fam = prob.line_fam[self.slice_line]
        self.onehot = np.eye(F)[self.slice_fam]                          # (K, F)
        U = prob.comp_usage
        self.fam_comp = [np.where(U[:, f] > 0)[0] for f in range(F)]
        self.fam_use = [U[self.fam_comp[f], f] for f in range(F)]
        self.fam_sites = [np.array([s for s in prob.site_order if prob.site_qual[s, f] > 0])
                          for f in range(F)]
        cps = prob.line_contrib_per_std()[self.slice_line]
        prio = prob.cust_priority[prob.line_cust][self.slice_line]
        firm = self.slice_firm
        K = self.K
        allk = np.arange(K)
        self.order = {
            # np.lexsort: last key is primary
            "strict_greedy": np.lexsort((-cps, prio, ~firm)),
            "margin_greedy": np.lexsort((-cps, ~firm)),
        }
        tiers = {"proportional": [allk]}
        tiers["priority_tiered"] = [allk[(firm == fl) & (prio == p)]
                                    for fl in (True, False) for p in np.unique(prio)]
        edges = np.quantile(cps, np.linspace(0, 1, opt.margin_bands + 1)[1:-1])
        band = np.digitize(cps, edges)                                 # 0 = lowest margin
        tiers["margin_tiered"] = [allk[(firm == fl) & (band == b)]
                                  for fl in (True, False)
                                  for b in range(opt.margin_bands - 1, -1, -1)]
        plan = prob.plan_revenue_by_cust()
        top = np.argsort(-plan)[:opt.top_n]
        is_top = np.isin(prob.line_cust[self.slice_line], top)
        tiers["protect_top_n"] = [allk[is_top], allk[~is_top & firm], allk[~is_top & ~firm]]
        self.tiers = {k: [t for t in v if len(t)] for k, v in tiers.items()}
        self.top_customers = top
        # Permute slices so each tier is a contiguous block: one gather in, one
        # scatter out per month, and every tier pass works on views.
        self.tier_perm, self.tier_bounds, self.tier_identity = {}, {}, {}
        for k, v in self.tiers.items():
            perm = np.concatenate(v)
            ends = np.cumsum([len(t) for t in v])
            self.tier_perm[k] = perm
            self.tier_bounds[k] = list(zip(np.r_[0, ends[:-1]], ends))
            self.tier_identity[k] = bool(np.array_equal(perm, allk))


# ----------------------------------------------------------------------------
# Allocation passes (operate in place on resource state)
#
# Layout: every per-path array is feature-major, (features, n_sims), so the
# reductions over small axes (families, sites, a family's components) are
# row adds over contiguous n-vectors. The sims-last layout was ~3x faster than
# (n_sims, features) in this spike.
# ----------------------------------------------------------------------------

def _prop_pass(want, fam_of, onehot_t, st, prob, comp_res, site_res, integ_res, rounds):
    """Pro-rata allocation of `want` (k, n) against residual resources.

    Pro-rata within a family is exact at family level, so the heavy work runs on
    (F, n) arrays and each line receives its family's fill ratio.
    """
    fam_want = onehot_t @ want                                         # (F, n)
    req = st.U @ fam_want                                              # (C, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_c = np.where(req > 1e-9, np.minimum(1.0, comp_res / req), 1.0)
    fam_scale = np.ones_like(fam_want)
    for f in range(st.F):
        if len(st.fam_comp[f]):
            fam_scale[f] = scale_c[st.fam_comp[f]].min(axis=0)
    fam_after = fam_want * fam_scale

    cx = st.cx_col
    alloc = np.zeros_like(fam_want)
    give_site = np.zeros_like(site_res)
    for _ in range(rounds):
        unmet_std = np.clip(fam_after - alloc, 0, None) * cx
        if unmet_std.max(initial=0.0) < 1e-9:
            break
        for s in st.site_seq:                                          # least contested first
            fq = st.site_fams[s]
            q = unmet_std[fq]                                          # (Fq, n)
            denom = q.sum(axis=0)
            share = np.divide(q, denom, out=np.zeros_like(q), where=denom > 1e-9)
            g = np.minimum(q, site_res[s] * share)
            alloc[fq] += g / cx[fq]
            gs = g.sum(axis=0)
            site_res[s] -= gs
            give_site[s] += gs
            unmet_std[fq] -= g
    np.minimum(alloc, fam_after, out=alloc)
    if integ_res is not None:
        tot = alloc.sum(axis=0)
        iscale = np.minimum(1.0, integ_res / np.clip(tot, 1e-9, None))
        alloc *= iscale
        site_res += give_site * (1 - iscale)                           # return unused EMS
        integ_res -= alloc.sum(axis=0)
    comp_res -= st.U @ alloc
    ratio = np.divide(alloc, fam_want, out=np.zeros_like(alloc), where=fam_want > 1e-12)
    return want * ratio[fam_of]


def _tiered(want, policy, st, prob, comp_res, site_res, integ_res, rounds, passes=1):
    perm, ident = st.tier_perm[policy], st.tier_identity[policy]
    wp = want if ident else want[perm]
    fam_p = st.slice_fam[perm]
    oh_p = st.onehot_t_perm[policy]
    outp = np.zeros_like(wp)
    for a, b in st.tier_bounds[policy]:
        for it in range(passes):
            w = wp[a:b] if it == 0 else wp[a:b] - outp[a:b]
            if it and w.max(initial=0.0) <= 1e-9:
                break
            outp[a:b] += _prop_pass(w, fam_p[a:b], oh_p[:, a:b], st, prob,
                                    comp_res, site_res, integ_res, rounds)
    if ident:
        return outp
    out = np.empty_like(want)
    out[perm] = outp
    return out


def _greedy(want, order, st, prob, comp_res, site_res, integ_res):
    """Exact sequential greedy over slices in `order`, vectorized across sims."""
    out = np.zeros_like(want)
    cx = prob.fam_cx
    for k in order:
        f = st.slice_fam[k]
        ci, cu = st.fam_comp[f], st.fam_use_col[f]
        sites = st.fam_sites[f]
        take = np.minimum(want[k], site_res[sites].sum(axis=0) / float(cx[f]))
        if len(ci):
            np.minimum(take, (comp_res[ci] / cu).min(axis=0), out=take)
        if integ_res is not None:
            np.minimum(take, integ_res, out=take)
        np.clip(take, 0, None, out=take)
        if integ_res is not None:
            integ_res -= take
        if len(ci):
            comp_res[ci] -= cu * take
        rem = take * float(cx[f])
        for s in sites:                                                # fill in fixed order
            d = np.minimum(rem, site_res[s])
            site_res[s] -= d
            rem -= d
        out[k] = take
    return out


# ----------------------------------------------------------------------------
# Monte Carlo driver
# ----------------------------------------------------------------------------

def simulate(prob: Problem, config, policy: str, n_sims: int = 5000, seed: int = 42,
             opt: SimOptions | None = None) -> SimResult:
    opt = opt or SimOptions()
    t0 = time.perf_counter()
    st = _Static(prob, opt)
    dt = np.dtype(opt.dtype)
    st.U = prob.comp_usage.astype(dt)
    st.cx_col = prob.fam_cx[:, None].astype(dt)
    st.fam_use_col = [u[:, None].astype(dt) for u in st.fam_use]
    st.onehot_t_perm = {k: st.onehot[v].T.astype(dt).copy() for k, v in st.tier_perm.items()}
    st.site_fams = [np.where(prob.site_qual[s] > 0)[0] for s in range(prob.site_qual.shape[0])]
    st.site_seq = prob.site_order if opt.waterfill_least_contested_first else \
        np.arange(prob.site_qual.shape[0])
    rng = np.random.default_rng(seed)       # same seed -> common random numbers across policies
    n, M, L, F = n_sims, prob.firm_base.shape[0], st.L, st.F
    C, S = prob.comp_usage.shape[0], prob.site_cap.shape[0]
    unc = config.uncertainty
    fam = prob.line_fam
    cust = prob.line_cust
    line_mkt = prob.fam_market[fam]

    # --- path-level draws (small: markets x M x n, customers x n) ---
    if opt.shocks:
        eng = FactorEngine(config.factors)
        factors = eng.draw_factor_paths(rng, n, M)
        mkt_mult = np.stack([
            _lognormal(eng.shock(mk, factors, rng), unc.market_demand_sigma[mk]).T
            for mk in prob.market_names], axis=1)                       # (M, markets, n)
        fam_mult = _lognormal(rng.standard_normal((F, n)), opt.fam_sigma)
        n_groups = int(prob.cust_group.max()) + 1
        gz = rng.standard_normal((n_groups, n))
        cz = (np.sqrt(opt.cust_corr) * gz[prob.cust_group]
              + np.sqrt(1 - opt.cust_corr) * rng.standard_normal((prob.n_cust, n)))
        c_sigma = unc.customer_idiosyncratic_sigma if opt.cust_sigma is None else opt.cust_sigma
        line_level = fam_mult[fam] * _lognormal(cz, c_sigma)[cust]      # (L, n) persistent
        asp_mult = _lognormal(rng.standard_normal((prob.n_cust, n)), unc.asp_sigma)[cust]
        tight = eng.shock("Component tightness", factors, rng).T        # (M, n)
        labor = np.clip(_lognormal(eng.shock("EMS execution", factors, rng),
                                   unc.ems_labor_sigma), 0.7, 1.1).T
        tight_mult = np.clip(1 - 0.08 * np.clip(tight, 0, None), 0.5, 1.0).astype(dt)
        mkt_mult, line_level, labor = mkt_mult.astype(dt), line_level.astype(dt), labor.astype(dt)
        asp_mult = asp_mult.astype(dt)
    else:
        asp_mult = np.ones((L, n), dt)
    firm_keep = (1 - prob.line_cancel_p * opt.firm_cancel_mult).astype(dt)[:, None]
    fcst_keep = (1 - prob.line_cancel_p).astype(dt)[:, None]
    push_p = prob.line_push_p.astype(dt)[:, None]

    comp_res = np.repeat(prob.comp_start[:, None], n, axis=1).astype(dt)   # (C, n)
    carry = np.zeros((L, n), dt)
    push_buf = np.zeros((3, L, n), dt)       # pushed orders arriving in m, m+1, m+2
    ship_q = np.zeros((6, L, n), dt)
    dem_q = np.zeros((6, L, n), dt)
    rev_fy = np.zeros((L, n), dt)
    gm_fy = np.zeros((L, n), dt)
    late_um = np.zeros((L, n), dt)
    unit_rev = prob.line_asp[:, None].astype(dt) * asp_mult
    unit_gm = unit_rev - prob.fam_cogs[fam][:, None].astype(dt)
    lag = prob.fam_rec_lag[fam]
    t_alloc = 0.0

    for m in range(M):
        # --- supply this month ---
        po = prob.comp_po.copy()
        for c, (m0, mult) in opt.comp_stress.items():
            if m >= m0:
                po[c] *= mult
        po = po.astype(dt)[:, None]
        site_res = np.repeat(prob.site_cap[:, m:m + 1].astype(dt), n, axis=1)
        integ_res = np.full(n, prob.integ_cap[m], dt) if opt.use_integration else None
        if opt.shocks:
            receipts = po * tight_mult[m]
            receipts *= np.where(rng.random((C, n), dt) < 0.02, dt.type(0.35), dt.type(1.0))
            site_res *= labor[m] * np.where(rng.random((S, n), dt) < 0.02,
                                            dt.type(0.65), dt.type(1.0))
            if integ_res is not None:
                integ_res *= np.clip(labor[m], 0.8, 1.05)
            comp_res += receipts
        else:
            comp_res += po

        # --- demand this month (firm: timing risk only; forecast: volume + timing) ---
        firm_m = prob.firm_base[m].astype(dt)[:, None]
        fcst_m = prob.fcst_base[m].astype(dt)[:, None]
        if opt.shocks:
            z = rng.standard_normal((L, n), dt)
            fcst = fcst_m * fcst_keep * mkt_mult[m, line_mkt] * line_level * \
                np.exp(opt.line_noise_sigma * z - 0.5 * opt.line_noise_sigma ** 2)
            firm = firm_m * firm_keep                                   # broadcasts at use
            # one uniform per line-month: u < p pushes the whole order out;
            # u < 0.3p (30% of pushes) slips two months instead of one
            u = rng.random((L, n), dt)
            stay = u >= push_p
            two = u < 0.3 * push_p
            tot = firm + fcst
            pushed2 = tot * two
            push_buf[1] += tot * ~stay - pushed2
            push_buf[2] += pushed2
            firm = firm * stay
            fcst *= stay
        else:
            firm = np.broadcast_to(firm_m, (L, n))
            fcst = np.broadcast_to(fcst_m, (L, n))
        new_firm = firm + push_buf[0]
        push_buf[0] = push_buf[1]
        push_buf[1] = push_buf[2]
        push_buf[2] = 0.0
        q = m // 3
        dem_q[q] += new_firm + fcst

        want = np.concatenate([new_firm + carry, fcst], axis=0)           # (K, n)

        # --- allocation ---
        ta = time.perf_counter()
        if policy in st.order:
            alloc = _greedy(want, st.order[policy], st, prob, comp_res, site_res, integ_res)
        else:
            alloc = _tiered(want, policy, st, prob, comp_res, site_res,
                            integ_res, opt.waterfill_rounds, opt.prop_passes)
        t_alloc += time.perf_counter() - ta

        ship = alloc[:L] + alloc[L:]
        carry = want[:L] + want[L:] - ship
        late_um += carry
        ship_q[q] += ship
        in_fy = ((m + lag) < FY)[:, None]                                # (L, 1)
        rev_fy += ship * unit_rev * in_fy
        gm_fy += ship * unit_gm * in_fy

    # public outputs are sims-first for metric code
    return SimResult(policy=policy, n_sims=n,
                     ship_q=ship_q.transpose(2, 0, 1), dem_q=dem_q.transpose(2, 0, 1),
                     rev_fy=rev_fy.T, gm_fy=gm_fy.T, late_um=late_um.T, end_backlog=carry.T,
                     t_total=time.perf_counter() - t0, t_alloc=t_alloc)
