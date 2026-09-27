"""Spike: build a demand-line problem from the synthetic APEX inputs.

A demand line is one customer x product family. The problem object holds the
static, per-line arrays the Monte Carlo kernel needs, plus the supply side
(components, EMS sites, integration) taken from `build_planning_arrays` so the
spike runs against the same facts as the production engine.

`scale_lines` splits every customer into k synthetic sub-customers (Dirichlet
shares, so concentration stays skewed) to test how the kernel scales toward a
real customer count while keeping total demand, and therefore constraint
tightness, unchanged.

Exploratory code only (see CLAUDE.md): nothing in src/ imports this.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data_generator import generate_all  # noqa: E402
from src.operations import build_planning_arrays, effective_site_capacity  # noqa: E402
from src.utils import FAMILY_MARKET, N_MONTHS  # noqa: E402


@dataclass
class Problem:
    # customers
    cust_names: list[str]
    cust_priority: np.ndarray      # (N_cust,) 1 = highest
    cust_group: np.ndarray         # (N_cust,) int index of customer group (correlation cluster)
    # demand lines
    line_cust: np.ndarray          # (L,) int
    line_fam: np.ndarray           # (L,) int
    firm_base: np.ndarray          # (M, L) firm backlog units by due month
    fcst_base: np.ndarray          # (M, L) forecast (unbooked) units by month
    line_asp: np.ndarray           # (L,) customer-specific ASP
    line_push_p: np.ndarray        # (L,) monthly push-out probability
    line_cancel_p: np.ndarray      # (L,) monthly cancellation probability
    # families
    fam_names: list[str]
    fam_market: np.ndarray         # (F,) int market index
    fam_cogs: np.ndarray           # (F,) standard unit COGS incl. warranty
    fam_cx: np.ndarray             # (F,) EMS std-equivalent weight per unit
    fam_rec_lag: np.ndarray        # (F,) 0/1 month revenue-recognition lag
    # supply
    comp_usage: np.ndarray         # (C, F)
    comp_start: np.ndarray         # (C,) usable on-hand above hard safety floor
    comp_po: np.ndarray            # (C,) nominal monthly receipts
    site_cap: np.ndarray           # (S, M) effective std-unit capacity
    site_qual: np.ndarray          # (S, F)
    site_order: np.ndarray         # (S,) greedy fill order (least contested, cheapest)
    integ_cap: np.ndarray          # (M,) company-level integration capacity (units)
    market_names: list[str]

    @property
    def n_lines(self) -> int:
        return len(self.line_cust)

    @property
    def n_cust(self) -> int:
        return len(self.cust_names)

    def line_contribution(self) -> np.ndarray:
        """Contribution per unit by line (ASP less standard COGS)."""
        return self.line_asp - self.fam_cogs[self.line_fam]

    def line_contrib_per_std(self) -> np.ndarray:
        """Contribution per EMS std-equivalent unit (the baseline's tiebreak)."""
        return self.line_contribution() / self.fam_cx[self.line_fam]

    def plan_revenue_by_cust(self, months: slice = slice(0, 12)) -> np.ndarray:
        units = (self.firm_base[months] + self.fcst_base[months]).sum(axis=0)
        return np.bincount(self.line_cust, weights=units * self.line_asp,
                           minlength=self.n_cust)


def load_problem(seed: int = 42) -> tuple[Problem, object]:
    """Build the 15-line problem from generate_all (written to a temp dir)."""
    data = generate_all(seed=seed, out_dir=tempfile.mkdtemp())
    config = load_config()
    pa = build_planning_arrays(data)
    fams = pa.families
    fam_ix = {f: i for i, f in enumerate(fams)}
    m_ix = {m: i for i, m in enumerate(pa.months)}

    dem = data.demand
    keys = dem[["customer", "product_family"]].drop_duplicates().reset_index(drop=True)
    cust_names = list(dict.fromkeys(keys["customer"]))
    c_ix = {c: i for i, c in enumerate(cust_names)}
    groups = list(dict.fromkeys(dem["customer_group"]))
    cmeta = dem.drop_duplicates("customer").set_index("customer")

    L = len(keys)
    firm = np.zeros((N_MONTHS, L))
    fcst = np.zeros((N_MONTHS, L))
    asp = np.zeros(L)
    push = np.zeros(L)
    cancel = np.zeros(L)
    for li, (c, f) in enumerate(keys.itertuples(index=False)):
        g = dem[(dem["customer"] == c) & (dem["product_family"] == f)]
        for r in g.itertuples(index=False):
            firm[m_ix[r.month], li] = r.backlog_units
            fcst[m_ix[r.month], li] = r.base_forecast_units
        w = (g["backlog_units"] + g["base_forecast_units"]).clip(lower=0.01)
        asp[li] = np.average(g["asp_usd"], weights=w)
        push[li] = np.average(g["push_out_prob"], weights=w)
        cancel[li] = np.average(g["cancel_prob"], weights=w)

    prod = data.products.set_index("product_family").loc[fams]
    cogs = (prod["material_cost_usd"] + prod["ems_conversion_cost_usd"]
            + prod["integration_test_cost_usd"] + prod["freight_cost_usd"]
            + prod["warranty_reserve_usd"]).to_numpy(float)
    markets = list(config.uncertainty.market_demand_sigma.keys())
    safety_floor = pa.comp_safety * 0.3            # same hard floor as the engine
    contention = pa.site_qual.sum(axis=1)

    prob = Problem(
        cust_names=cust_names,
        cust_priority=np.array([int(cmeta.loc[c, "customer_priority"]) for c in cust_names]),
        cust_group=np.array([groups.index(cmeta.loc[c, "customer_group"]) for c in cust_names]),
        line_cust=np.array([c_ix[c] for c in keys["customer"]]),
        line_fam=np.array([fam_ix[f] for f in keys["product_family"]]),
        firm_base=firm, fcst_base=fcst, line_asp=asp,
        line_push_p=push, line_cancel_p=cancel,
        fam_names=list(fams),
        fam_market=np.array([markets.index(FAMILY_MARKET[f]) for f in fams]),
        fam_cogs=cogs, fam_cx=pa.family_complexity.copy(),
        fam_rec_lag=pa.family_rec_lag.astype(int),
        comp_usage=pa.comp_usage.copy(),
        comp_start=np.clip(pa.comp_on_hand - safety_floor, 0, None),
        comp_po=pa.comp_po_monthly.copy(),
        site_cap=effective_site_capacity(pa),
        site_qual=pa.site_qual.copy(),
        site_order=np.lexsort((pa.site_cost, contention)),
        integ_cap=pa.integration_capacity.copy(),
        market_names=markets,
    )
    return prob, config


def scale_lines(prob: Problem, target_lines: int, seed: int = 7,
                alpha: float = 0.8) -> Problem:
    """Split customers into sub-customers until the line count reaches the target.

    Each original customer c with n_c families is split into k_c sub-customers;
    each sub-customer inherits all of c's families (so lines = sum k_c * n_c).
    Shares are Dirichlet(alpha) per family, keeping demand totals unchanged.
    """
    rng = np.random.default_rng(seed)
    n_fam_per_cust = np.bincount(prob.line_cust, minlength=prob.n_cust)
    base_k = max(1, target_lines // prob.n_lines)
    k = np.full(prob.n_cust, base_k)
    # top up with single-family customers first to hit the target exactly
    gap = target_lines - int((k * n_fam_per_cust).sum())
    order = np.argsort(n_fam_per_cust, kind="stable")
    while gap > 0:
        progressed = False
        for c in order:
            if 0 < n_fam_per_cust[c] <= gap:
                k[c] += 1
                gap -= n_fam_per_cust[c]
                progressed = True
            if gap <= 0:
                break
        if not progressed:
            break

    names, prio, group = [], [], []
    lc, lf, firm, fcst, asp, push, cancel = [], [], [], [], [], [], []
    for c in range(prob.n_cust):
        lines_c = np.where(prob.line_cust == c)[0]
        shares = rng.dirichlet(np.full(k[c], alpha), size=len(lines_c))  # (n_lines_c, k)
        for j in range(k[c]):
            new_c = len(names)
            names.append(f"{prob.cust_names[c]} #{j + 1}")
            prio.append(prob.cust_priority[c])
            group.append(prob.cust_group[c])
            for i, li in enumerate(lines_c):
                lc.append(new_c)
                lf.append(prob.line_fam[li])
                firm.append(prob.firm_base[:, li] * shares[i, j])
                fcst.append(prob.fcst_base[:, li] * shares[i, j])
                asp.append(prob.line_asp[li] * rng.uniform(0.96, 1.04))
                push.append(prob.line_push_p[li])
                cancel.append(prob.line_cancel_p[li])
    return replace(
        prob, cust_names=names, cust_priority=np.array(prio), cust_group=np.array(group),
        line_cust=np.array(lc), line_fam=np.array(lf),
        firm_base=np.array(firm).T, fcst_base=np.array(fcst).T,
        line_asp=np.array(asp), line_push_p=np.array(push), line_cancel_p=np.array(cancel),
    )
