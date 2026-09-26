"""Shared setup for the engine-reconciliation spike.

Zero-shock world: every sigma in config.uncertainty set to 0, EMS regional
disruption probability 0, and timing/cancel/disruption/acceptance events
switched off through sim params. utilization_adherence_penalty and
site_disruption_impact are not sigmas and are left at config values.
Data is generated into a temp dir (never data/generated).
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p_ in (str(ROOT), str(HERE)):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data_generator import generate_all  # noqa: E402

ZERO_PARAMS = dict(cancel_prob_mult=0.0, pushout_prob_add=-1.0, pullin_prob_add=-1.0,
                   comp_disrupt_mult=0.0, acceptance_delay_add=-1.0)
SIGMAS = ["customer_idiosyncratic_sigma", "asp_sigma", "material_cost_sigma",
          "fx_cost_sigma", "conversion_cost_sigma", "freight_sigma", "ems_labor_sigma"]


def load_data(seed: int = 42):
    return generate_all(seed=seed, out_dir=tempfile.mkdtemp())


def zero_shock(data, config):
    """Return (data, config) copies with every shock neutralized."""
    d = copy.deepcopy(data)
    c = config.model_copy(deep=True)
    for k in SIGMAS:
        setattr(c.uncertainty, k, 0.0)
    c.uncertainty.market_demand_sigma = {k: 0.0 for k in c.uncertainty.market_demand_sigma}
    d.ems_sites["regional_disruption_prob_monthly"] = 0.0
    return d, c


def fy_metrics_sim(res: dict) -> dict:
    """FY (months 0-11) metrics, averaged over sims, from a run_variant result."""
    rev = res["revenue"][:, :12].sum(1)
    gp = res["gross_profit"][:, :12].sum(1)
    return {
        "rev": float(rev.mean()) / 1e6,
        "shipped": float(res["ship"][:, :12].sum(axis=(1, 2)).mean()),
        "gm": float((gp / rev).mean()),
        "gp": float(gp.mean()) / 1e6,
        "inv": float(res["inventory"][:, 11].mean()) / 1e6,
        "ems_util": float(res["ems_util"][:, :12].mean()),
        "integ_util": float(res["integ_util"][:, :12].mean()),
    }


def fy_metrics_baseline(b) -> dict:
    m = b.monthly
    rev = m["revenue_usd"][:12].sum()
    gp = m["gross_profit_usd"][:12].sum()
    return {"rev": rev / 1e6, "shipped": float(m["units_built"][:12].sum()),
            "gm": gp / rev, "gp": gp / 1e6, "inv": float(m["inventory_usd"].iloc[11]) / 1e6,
            "ems_util": float(m["ems_utilization"][:12].mean()),
            "integ_util": float(m["integration_utilization"][:12].mean())}


def binding_probe(run, base_kwargs: dict, bump: float = 1.3) -> dict:
    """Which constraint binds: FY revenue response to +30% integration or +30% EMS."""
    base = fy_metrics_sim(run(**base_kwargs))["rev"]
    p_int = dict(base_kwargs.get("params") or {}, integration_capacity_mult=bump)
    p_ems = dict(base_kwargs.get("params") or {},
                 ems_capacity_mult={s: bump for s in ("EMS Americas", "EMS Malaysia",
                                                      "EMS Taiwan", "EMS Eastern Europe")})
    r_int = fy_metrics_sim(run(**{**base_kwargs, "params": p_int}))["rev"]
    r_ems = fy_metrics_sim(run(**{**base_kwargs, "params": p_ems}))["rev"]
    return {"d_rev_integ_x1.3": r_int - base, "d_rev_ems_x1.3": r_ems - base}
