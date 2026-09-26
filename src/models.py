"""Typed containers for the major inputs and outputs of the engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class InputData:
    """All generated planning inputs, loaded as DataFrames."""

    demand: pd.DataFrame          # customer x family x month demand plan
    products: pd.DataFrame        # product-family cost/cycle assumptions
    components: pd.DataFrame      # critical component master
    ems_capacity: pd.DataFrame    # EMS site x month capacity assumptions
    ems_sites: pd.DataFrame       # EMS site master
    integration_capacity: pd.DataFrame  # integration site x month (not read: integration is at the EMS)
    financial_plan: pd.DataFrame  # month-level revenue plan and targets
    seed: int = 42


@dataclass
class BaselineResult:
    """Deterministic monthly baseline plan."""

    monthly: pd.DataFrame            # company P&L / inventory by month
    family_units: pd.DataFrame       # planned recognized units, family x month
    family_revenue: pd.DataFrame     # recognized revenue, family x month
    site_load: pd.DataFrame          # EMS site x month builds (std-equivalent)
    site_capacity: pd.DataFrame      # EMS site x month available capacity
    component_usage: pd.DataFrame    # component x month consumption vs supply
    constraints: pd.DataFrame        # constraint log (month, type, detail, units lost)
    unmet: pd.DataFrame              # unmet demand by family x month
    backlog_aging: pd.DataFrame      # backlog aging buckets by month
    demand_units: np.ndarray         # (n_months, n_families) demand before constraints
    supply_units: np.ndarray         # (n_months, n_families) constrained builds
    revenue_plan_q: np.ndarray       # quarterly revenue plan (6,)
    revenue_plan_m: np.ndarray       # monthly revenue plan (18,)


@dataclass
class SimulationResult:
    """Vectorized Monte Carlo outputs. Arrays are (n_sims, n_months) unless noted."""

    n_sims: int
    seed: int
    scenario_name: str
    revenue: np.ndarray
    cogs: np.ndarray
    gross_profit: np.ndarray
    operating_income: np.ndarray
    ebitda: np.ndarray
    cash_flow: np.ndarray
    inventory: np.ndarray            # ending inventory by month
    raw_inventory: np.ndarray
    wip_inventory: np.ndarray
    fg_inventory: np.ndarray
    working_capital: np.ndarray
    expedite_cost: np.ndarray
    rework_cost: np.ndarray
    eo_reserve: np.ndarray           # (n_sims,) fiscal-year E&O reserve estimate
    family_revenue: np.ndarray       # (n_sims, n_months, n_families)
    family_units: np.ndarray         # (n_sims, n_months, n_families) recognized
    family_shipped: np.ndarray       # (n_sims, n_months, n_families)
    family_demand: np.ndarray        # (n_sims, n_months, n_families)
    units_shipped: np.ndarray        # (n_sims, n_months)
    units_demanded: np.ndarray       # (n_sims, n_months)
    ems_utilization: np.ndarray      # (n_sims, n_months)
    capacity_shortfall_units: np.ndarray   # (n_sims, n_months)
    component_short_units: np.ndarray      # (n_sims, n_months)
    component_binding: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (n_sims,) bool
    site_disrupted: dict[str, np.ndarray] = field(default_factory=dict)
    drivers: dict[str, np.ndarray] = field(default_factory=dict)  # sampled inputs for sensitivity
    params: dict[str, Any] = field(default_factory=dict)
    # allocation detail (per path); the baseline supply plan reads path 0
    family_backlog: np.ndarray | None = None   # (n, M, F) end-of-month unmet (past-due)
    site_load: np.ndarray | None = None        # (n, S, M) std-units built per EMS site
    site_capacity: np.ndarray | None = None    # (n, S, M) effective std-unit capacity
    limit_units: np.ndarray | None = None      # (2, n, M, F) cut by component / EMS capacity
    binding_component: np.ndarray | None = None  # (n, M, F) index of ceiling component, -1 none
    component_consumed: np.ndarray | None = None  # (n, M, C) units consumed per month
    component_usable_supply: np.ndarray | None = None  # (n, M, C) cumulative usable supply

    @property
    def gross_margin(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.revenue > 0, self.gross_profit / self.revenue, 0.0)


@dataclass
class ScenarioSpec:
    """A named set of assumption overrides applied before simulation."""

    name: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)
    action_cost_usd: float = 0.0     # direct cost of the management decision (fees etc.)
    horizon: str = "Tactical"        # Execution (0-3 mo) / Tactical (1-3 qtrs) / Long-lead (6-18 mo)


@dataclass
class Recommendation:
    """A traceable, evidence-based management recommendation."""

    title: str
    risk: str                        # the detected risk or opportunity that triggered it
    threshold: str                   # measurable trigger that fired
    action: str                      # the modeled management action
    expected_value_usd: float
    revenue_protected_usd: float
    gross_profit_protected_usd: float
    prob_plan_improvement: float     # change in P(hit revenue plan), fraction
    incremental_cost_usd: float
    inventory_change_usd: float
    working_capital_change_usd: float
    service_level_change: float
    score: float
    confidence: str                  # High / Medium / Low
    caveat: str
