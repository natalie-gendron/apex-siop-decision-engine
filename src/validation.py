"""Input validation with user-friendly messages.

Errors are conditions the engine cannot run with; warnings are suspicious but
tolerable. Technical detail is logged via the standard logging module while the
Streamlit layer shows only the friendly message.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .models import InputData
from .utils import EMS_SITES, PRODUCT_FAMILIES, month_labels

logger = logging.getLogger("siop.validation")


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    area: str
    message: str


PROB_COLUMNS = {
    "demand": ["forecast_confidence", "pull_in_prob", "push_out_prob",
               "cancel_prob", "site_readiness_prob"],
    "components": ["allocation_risk", "disruption_prob_monthly", "obsolescence_risk"],
    "ems_capacity": ["schedule_adherence", "first_pass_yield", "labor_availability"],
}

NONNEG_COLUMNS = {
    "demand": ["base_forecast_units", "backlog_units", "asp_usd"],
    "components": ["unit_cost_usd", "lead_time_weeks", "on_hand_units",
                   "open_po_units_per_month", "safety_stock_units"],
    "ems_capacity": ["available_capacity_units", "max_overtime_units",
                     "capacity_cost_per_unit_usd"],
    "integration_capacity": ["integration_capacity_units", "fat_capacity_units",
                             "installation_capacity_units"],
}


def validate_inputs(data: InputData) -> list[ValidationIssue]:
    """Run all input checks and return a list of issues (empty = clean)."""
    issues: list[ValidationIssue] = []
    frames = {
        "demand": data.demand, "components": data.components,
        "ems_capacity": data.ems_capacity,
        "integration_capacity": data.integration_capacity,
    }

    for table, cols in PROB_COLUMNS.items():
        df = frames[table]
        for col in cols:
            bad = df[(df[col] < 0) | (df[col] > 1)]
            if len(bad):
                issues.append(ValidationIssue(
                    "error", table,
                    f"{col} must be between 0 and 1 — {len(bad)} row(s) out of range."))
                logger.error("Out-of-range %s in %s: rows %s", col, table, bad.index.tolist()[:10])

    for table, cols in NONNEG_COLUMNS.items():
        df = frames[table]
        for col in cols:
            bad = df[df[col] < 0]
            if len(bad):
                issues.append(ValidationIssue(
                    "error", table,
                    f"{col} cannot be negative — {len(bad)} row(s) below zero."))

    # yields within logical bounds
    y = data.ems_capacity["first_pass_yield"]
    if ((y < 0.5) | (y > 1.0)).any():
        issues.append(ValidationIssue(
            "warning", "ems_capacity",
            "First-pass yield outside the plausible 50%-100% range for some site-months."))

    # months align to the planning horizon
    horizon = set(month_labels())
    for table in ("demand", "ems_capacity", "integration_capacity"):
        months = set(frames[table]["month"].unique())
        stray = months - horizon
        if stray:
            issues.append(ValidationIssue(
                "error", table,
                f"Months outside the 18-month planning horizon: {sorted(stray)[:4]}."))

    # product-to-site qualification must be valid
    known_sites = set(EMS_SITES)
    for _, row in data.products.iterrows():
        sites = set(str(row["qualified_ems_sites"]).split(";"))
        unknown = sites - known_sites
        if unknown:
            issues.append(ValidationIssue(
                "error", "products",
                f"{row['product_family']} references unknown EMS site(s): {sorted(unknown)}."))
        if not sites & known_sites:
            issues.append(ValidationIssue(
                "error", "products",
                f"{row['product_family']} has no qualified EMS site — it could never be built."))

    # component usage must reference valid products
    known_fams = set(PRODUCT_FAMILIES)
    for _, row in data.components.iterrows():
        fams = {f for f in str(row["products_using"]).split(";") if f}
        unknown = fams - known_fams
        if unknown:
            issues.append(ValidationIssue(
                "error", "components",
                f"Component '{row['component']}' references unknown product(s): {sorted(unknown)}."))

    # demand families must exist
    unknown_fams = set(data.demand["product_family"].unique()) - known_fams
    if unknown_fams:
        issues.append(ValidationIssue(
            "error", "demand", f"Unknown product families in demand plan: {sorted(unknown_fams)}."))

    return issues


def check_psd(matrix: np.ndarray, repair: bool = True) -> tuple[np.ndarray, bool]:
    """Verify a correlation matrix is positive semidefinite; repair by eigenvalue
    clipping if requested. Returns (matrix, was_valid)."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    if eigvals.min() >= -1e-10:
        return matrix, True
    if not repair:
        return matrix, False
    logger.warning("Correlation matrix not PSD (min eigenvalue %.4g); repairing.", eigvals.min())
    clipped = np.clip(eigvals, 1e-10, None)
    repaired = eigvecs @ np.diag(clipped) @ eigvecs.T
    d = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(d, d)
    np.fill_diagonal(repaired, 1.0)
    return repaired, False


def clamp_probability(value: float, name: str = "probability") -> float:
    """Clamp a probability into [0, 1], logging if it was out of range."""
    if not 0.0 <= value <= 1.0:
        logger.warning("%s value %.4f clamped to [0, 1]", name, value)
    return float(np.clip(value, 0.0, 1.0))
