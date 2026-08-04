"""Shared utilities: planning calendar, formatting, and statistics helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_MONTHS = 18
PLAN_START = "2026-07"  # first month of the 18-month rolling horizon
FISCAL_YEAR_MONTHS = 12  # first 12 months constitute the fiscal year (FY2027)

PRODUCT_FAMILIES = [
    "Zenith Compute Test",
    "Vector Memory Test",
    "Horizon Mobility Test",
    "Atlas Automotive & Industrial Test",
    "Nexus System-Level Test",
]

MARKET_SEGMENTS = [
    "AI / HPC",
    "Memory",
    "Mobile / Consumer",
    "Automotive",
    "Industrial / Mixed-Signal",
]

# Primary end-market driver for each product family
FAMILY_MARKET = {
    "Zenith Compute Test": "AI / HPC",
    "Vector Memory Test": "Memory",
    "Horizon Mobility Test": "Mobile / Consumer",
    "Atlas Automotive & Industrial Test": "Automotive",
    "Nexus System-Level Test": "AI / HPC",
}

REGIONS = [
    "North America", "Taiwan", "South Korea", "Japan",
    "Europe", "Southeast Asia", "China",
]

EMS_SITES = ["EMS Americas", "EMS Malaysia", "EMS Taiwan", "EMS Eastern Europe"]
INTEGRATION_SITES = ["Final Integration North America", "Final Integration Asia"]


def month_index() -> pd.PeriodIndex:
    """The 18 monthly planning periods."""
    return pd.period_range(PLAN_START, periods=N_MONTHS, freq="M")


def month_labels() -> list[str]:
    return [str(p) for p in month_index()]


def quarter_of_month(m: int) -> int:
    """Planning quarter (1-6) of a 0-based month index."""
    return m // 3 + 1


def quarter_labels() -> list[str]:
    """Fiscal-quarter labels for the six planning quarters."""
    return [f"FY27 Q{q}" for q in range(1, 5)] + ["FY28 Q1", "FY28 Q2"]


def quarter_slice(q: int) -> slice:
    """0-based month slice for planning quarter q (1-6)."""
    return slice((q - 1) * 3, q * 3)


def fmt_money(x: float, decimals: int = 1) -> str:
    """Format dollars in $M (or $B above 1,000M)."""
    m = x / 1e6
    if abs(m) >= 1000:
        return f"${m / 1000:,.2f}B"
    return f"${m:,.{decimals}f}M"


def fmt_pct(x: float, decimals: int = 1) -> str:
    """Format a fraction (0-1) as a percentage."""
    return f"{100 * x:.{decimals}f}%"


def fmt_pts(x: float, decimals: int = 1) -> str:
    """Format a fraction delta as percentage points."""
    sign = "+" if x >= 0 else ""
    return f"{sign}{100 * x:.{decimals}f} pts"


def fmt_delta_money(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{fmt_money(abs(x))[1:]}" if x < 0 else f"+{fmt_money(x)[1:]}"


def percentile_stats(samples: np.ndarray) -> dict[str, float]:
    """Standard distribution summary used throughout the app."""
    s = np.asarray(samples, dtype=float)
    return {
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "std": float(np.std(s)),
        "p5": float(np.percentile(s, 5)),
        "p25": float(np.percentile(s, 25)),
        "p75": float(np.percentile(s, 75)),
        "p95": float(np.percentile(s, 95)),
    }


def prob_at_least(samples: np.ndarray, threshold: float) -> float:
    """P(sample >= threshold)."""
    s = np.asarray(samples, dtype=float)
    return float(np.mean(s >= threshold))


def downside_at_risk(samples: np.ndarray, reference: float, pct: int = 5) -> float:
    """Reference value minus the pct-th percentile (0 if percentile above reference)."""
    return max(0.0, reference - float(np.percentile(samples, pct)))
