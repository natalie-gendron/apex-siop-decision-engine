"""Sensitivity and risk-driver analysis.

Method: Spearman rank correlation between sampled input drivers and simulated
outcomes. Rank correlation is robust to the nonlinear, bounded transformations
in the engine, but it measures ASSOCIATION, not causation — drivers share
common factors by design, so correlated drivers partially proxy for each other.
The dashboard labels the method accordingly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .models import BaselineResult, SimulationResult
from .simulation import fiscal_year, quarterly, service_level
from .utils import PRODUCT_FAMILIES


def outcome_arrays(result: SimulationResult) -> dict[str, np.ndarray]:
    """The five outcomes that drivers are ranked against."""
    fy_rev = fiscal_year(result.revenue)
    with np.errstate(divide="ignore", invalid="ignore"):
        fy_gm = np.where(fy_rev > 0, fiscal_year(result.gross_profit) / fy_rev, 0.0)
    return {
        "Q1 revenue": quarterly(result.revenue)[:, 0],
        "FY revenue": fy_rev,
        "FY gross margin": fy_gm,
        "Ending inventory": result.inventory[:, 11],
        "Working capital": result.working_capital[:, 11],
        "Service level": service_level(result),
    }


def driver_ranking(result: SimulationResult, outcome: str = "FY revenue") -> pd.DataFrame:
    """Spearman rank correlation of every sampled driver against an outcome."""
    y = outcome_arrays(result)[outcome]
    rows = []
    for name, x in result.drivers.items():
        if np.std(x) < 1e-12:
            continue
        rho, pval = stats.spearmanr(x, y)
        rows.append({"driver": name, "spearman_rho": rho, "abs_rho": abs(rho),
                     "p_value": pval})
    df = pd.DataFrame(rows).sort_values("abs_rho", ascending=False).reset_index(drop=True)
    return df


def all_driver_rankings(result: SimulationResult) -> dict[str, pd.DataFrame]:
    return {name: driver_ranking(result, name) for name in outcome_arrays(result)}


def binding_components(result: SimulationResult, top_n: int = 8) -> pd.DataFrame:
    """Components most frequently binding on shipments across simulations."""
    rows = [{"component": name, "binding_frequency": float(mask.mean())}
            for name, mask in result.component_binding.items()]
    return (pd.DataFrame(rows).sort_values("binding_frequency", ascending=False)
            .head(top_n).reset_index(drop=True))


def site_disruption_frequency(result: SimulationResult) -> pd.DataFrame:
    rows = [{"ems_site": s, "disruption_frequency": float(mask.mean())}
            for s, mask in result.site_disrupted.items()]
    return pd.DataFrame(rows).sort_values("disruption_frequency", ascending=False)


def family_revenue_at_risk(result: SimulationResult,
                           baseline: BaselineResult) -> pd.DataFrame:
    """Revenue at risk by product family: baseline FY plan revenue minus the
    5th-percentile simulated FY revenue (floored at zero)."""
    fam_fy = result.family_revenue[:, :12, :].sum(axis=1)      # (n, F)
    base_fy = baseline.family_revenue.iloc[:12].sum().to_numpy(float)
    rows = []
    for i, fam in enumerate(PRODUCT_FAMILIES):
        p5 = float(np.percentile(fam_fy[:, i], 5))
        rows.append({
            "product_family": fam,
            "baseline_fy_revenue": base_fy[i],
            "expected_fy_revenue": float(fam_fy[:, i].mean()),
            "p5_fy_revenue": p5,
            "revenue_at_risk": max(0.0, base_fy[i] - p5),
        })
    return (pd.DataFrame(rows).sort_values("revenue_at_risk", ascending=False)
            .reset_index(drop=True))


def monthly_capacity_risk(result: SimulationResult) -> pd.DataFrame:
    """Months ranked by probability of a meaningful capacity shortfall."""
    short = result.capacity_shortfall_units + result.component_short_units
    p_short = (short > 2.0).mean(axis=0)                       # (18,)
    from .utils import month_labels
    return pd.DataFrame({
        "month": month_labels(),
        "p_capacity_shortfall": p_short,
        "expected_units_short": short.mean(axis=0),
    }).sort_values("p_capacity_shortfall", ascending=False).reset_index(drop=True)


def quarter_shift_drivers(result: SimulationResult) -> pd.DataFrame:
    """Association between drivers and revenue shifting from Q1 into Q2
    (positive rho = driver moves revenue out of Q1)."""
    rev_q = quarterly(result.revenue)
    shift = rev_q[:, 1] - rev_q[:, 0]
    rows = []
    for name, x in result.drivers.items():
        if np.std(x) < 1e-12:
            continue
        rho, _ = stats.spearmanr(x, shift)
        rows.append({"driver": name, "spearman_rho": rho})
    return (pd.DataFrame(rows).assign(abs_rho=lambda d: d["spearman_rho"].abs())
            .sort_values("abs_rho", ascending=False).drop(columns="abs_rho")
            .reset_index(drop=True))
