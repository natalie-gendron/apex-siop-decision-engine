"""Customer-level outcomes: who gets shorted, and what concentration costs.

Definitions (FY = months 1-12), per customer:
  Revenue at risk   baseline FY revenue (the plan-of-record allocation) minus
                    the P10 simulated FY revenue
  Supply gap        E[(FY demand units - FY shipped units)+ x ASP]: the part
                    of the exposure that supply, not demand, causes
  P(shortfall)      P(FY fill rate < 95%)
  Late unit-months  FY sum of past-due units, per unit demanded
  Lost revenue      orders lost after waiting (customers with a limit only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import BaselineResult, SimulationResult

SHORTFALL_FILL = 0.95


def customer_table(result: SimulationResult, baseline: BaselineResult,
                   priority: dict[str, int] | None = None) -> pd.DataFrame:
    """One row per customer, ordered by FY plan revenue."""
    rev = result.customer_revenue[:, :12].sum(axis=1)                  # (n, Cu)
    dem = result.customer_demand[:, :12].sum(axis=1).astype(float)
    shp = result.customer_shipped[:, :12].sum(axis=1).astype(float)
    fill = np.divide(shp, dem, out=np.ones_like(shp), where=dem > 1e-9).clip(0, 1)
    asp = np.divide(rev.mean(axis=0), shp.mean(axis=0),
                    out=np.zeros(rev.shape[1]), where=shp.mean(axis=0) > 1e-9)
    gap = (np.clip(dem - shp, 0, None) * asp[None, :]).mean(axis=0)
    base = baseline.customer_revenue.iloc[:12].sum().reindex(result.customers).to_numpy(float)
    p10 = np.percentile(rev, 10, axis=0)
    total = rev.mean(axis=0).sum()
    df = pd.DataFrame({
        "Customer": result.customers,
        "Priority": [priority.get(c) if priority else None for c in result.customers],
        "Share of FY revenue": rev.mean(axis=0) / total,
        "Expected FY revenue": rev.mean(axis=0),
        "P10 FY revenue": p10,
        "Revenue at risk": np.clip(base - p10, 0, None),
        "Supply gap": gap,
        "FY fill rate": fill.mean(axis=0),
        "P(shortfall)": (fill < SHORTFALL_FILL).mean(axis=0),
        "Late unit-months per unit": result.customer_late_unit_months.mean(axis=0)
                                     / np.clip(dem.mean(axis=0), 1e-9, None),
        "Lost revenue": result.customer_lost_revenue[:, :12].sum(axis=1).mean(axis=0),
        "Expected FY contribution": result.customer_gross_profit[:, :12].sum(axis=1).mean(axis=0),
    })
    if priority is None:
        df = df.drop(columns="Priority")
    return df


def concentration(result: SimulationResult, baseline: BaselineResult,
                  top: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    """Share, P10 and revenue at risk of the top-N customers (by plan revenue)."""
    rev = result.customer_revenue[:, :12].sum(axis=1)
    base = baseline.customer_revenue.iloc[:12].sum().reindex(result.customers).to_numpy(float)
    total = rev.sum(axis=1)
    rows = []
    for n in top:
        r = rev[:, :n].sum(axis=1)
        rows.append({"Group": f"Top {n}", "Share of FY revenue": float((r / total).mean()),
                     "Expected FY revenue": float(r.mean()),
                     "P10 FY revenue": float(np.percentile(r, 10)),
                     "Revenue at risk": max(0.0, float(base[:n].sum() - np.percentile(r, 10)))})
    return pd.DataFrame(rows)


def policy_comparison(results: dict[str, SimulationResult], reference: str = "proportional",
                      top_n: int = 3) -> pd.DataFrame:
    """The cost of who gets shorted: one row per allocation policy, paired on
    common random numbers against `reference`."""
    ref = results[reference]
    ref_gp = ref.gross_profit[:, :12].sum(axis=1)
    rows = []
    for name, r in results.items():
        dem = r.customer_demand[:, :12].sum(axis=1).astype(float)
        shp = r.customer_shipped[:, :12].sum(axis=1).astype(float)
        fill = np.divide(shp, dem, out=np.ones_like(shp), where=dem > 1e-9)
        short = (fill < SHORTFALL_FILL).mean(axis=0)
        top_short = (fill[:, :top_n].min(axis=1) < SHORTFALL_FILL).mean()
        d_gp = r.gross_profit[:, :12].sum(axis=1) - ref_gp
        rows.append({
            "Policy": name,
            "Expected FY revenue": float(r.revenue[:, :12].sum(axis=1).mean()),
            "Expected FY gross profit": float(r.gross_profit[:, :12].sum(axis=1).mean()),
            "Δ gross profit vs reference": float(d_gp.mean()),
            "Δ GP P10": float(np.percentile(d_gp, 10)),
            "Δ GP P90": float(np.percentile(d_gp, 90)),
            f"P(a top-{top_n} customer short)": float(top_short),
            "Customers with P(shortfall) > 25%": int((short > 0.25).sum()),
            "Worst customer fill": float(fill.mean(axis=0).min()),
        })
    return pd.DataFrame(rows)
