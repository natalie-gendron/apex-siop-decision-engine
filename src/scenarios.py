"""Scenario manager: prebuilt scenarios, management actions, KPI summaries and
scenario-versus-base comparison."""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import AppConfig
from .models import BaselineResult, ScenarioSpec, SimulationResult
from .simulation import fiscal_year, quarterly, service_level
from .utils import percentile_stats


def prebuilt_scenarios() -> dict[str, ScenarioSpec]:
    """The prebuilt SIOP scenarios — exogenous world-states only.

    A scenario is something that happens TO the business (demand and/or
    supply environment); it carries no decision cost. Things the business
    chooses to DO live in config/management_actions.yaml as costed,
    latency-ramped management actions and combine into response packages.
    Both share the ScenarioSpec override-bundle primitive."""
    memory_recovery = np.concatenate([np.linspace(1.0, 1.3, 12), np.full(6, 1.3)])
    memory_delay = np.concatenate([np.full(9, 0.80), np.linspace(0.82, 1.0, 9)])
    scenarios = [
        ScenarioSpec("Base Case", "Current SIOP assumptions with no overrides.", {}),
        ScenarioSpec(
            "AI Surge with Supply Tightening",
            "AI / HPC demand runs 25% above plan; select customers pull in orders; "
            "FPGA and high-speed interconnect supply tightens.",
            {"demand_market_mult": {"AI / HPC": 1.25},
             "pullin_prob_add": 0.03,
             "comp_supply_mult": {"High-End FPGA": 0.95, "High-Speed Interconnect Set": 0.95},
             "comp_disrupt_mult": 1.2}),
        ScenarioSpec(
            "Memory Recovery",
            "Memory-test demand improves steadily to +30% over twelve months.",
            {"demand_market_mult": {"Memory": memory_recovery}}),
        ScenarioSpec(
            "Memory Recovery Delay",
            "Memory demand stays 20% below plan for nine months before recovering.",
            {"demand_market_mult": {"Memory": memory_delay}}),
        ScenarioSpec(
            "Major Customer Push-Out",
            "A large compute customer shifts ~15 Zenith systems from mid-Q1 into "
            "Q2 (acceptance-based recognition moves the revenue with them).",
            {"forced_pushout": {"family": "Zenith Compute Test",
                                "from_month": 1, "to_month": 4, "units": 15}}),
        ScenarioSpec(
            "Critical FPGA Shortage",
            "High-end FPGA receipts fall 30%, lead times stretch 35%, and expedite "
            "premiums rise 60%.",
            {"comp_supply_mult": {"High-End FPGA": 0.70},
             "lead_time_mult": 1.35, "expedite_premium_mult": 1.6,
             "comp_disrupt_mult": 1.5}),
        ScenarioSpec(
            "EMS Malaysia Disruption",
            "EMS Malaysia loses 45% of capacity for three months (months 2-4) with "
            "degraded schedule adherence.",
            {"ems_window_mult": {"EMS Malaysia": (1, 4, 0.55)},
             "adherence_delta": -0.02}),
        ScenarioSpec(
            "Customer Site-Readiness Delay",
            "Completed systems wait longer for installation and acceptance; slip "
            "probability rises 15 points.",
            {"acceptance_delay_add": 0.15}),
    ]
    return {s.name: s for s in scenarios}


VALID_HORIZONS = ("Execution", "Tactical", "Long-lead")


def management_actions(catalog_path: "str | None" = None,
                       catalog_text: "str | None" = None) -> dict[str, ScenarioSpec]:
    """Load the management-action catalog from config/management_actions.yaml.

    The catalog is the analyst-owned "claim sheet" layer: capability, cost and
    timing are authored there; the simulation computes the consequences. The
    file is validated on load (horizons, non-negative costs, and override keys
    checked against the simulator's known parameters).

    `catalog_text` parses a catalog from a YAML string instead of a file —
    used by the session-only what-if upload on the Assumptions & Data page."""
    import yaml

    from .config import PROJECT_ROOT
    from .simulation import default_params

    if catalog_text is not None:
        raw = yaml.safe_load(catalog_text)
    else:
        path = catalog_path or (PROJECT_ROOT / "config" / "management_actions.yaml")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "actions" not in raw:
        raise ValueError("Catalog must be a YAML mapping with a top-level "
                         "'actions' list.")
    known_keys = set(default_params())
    out: dict[str, ScenarioSpec] = {}
    for entry in raw["actions"]:
        name = entry["name"]
        overrides = entry.get("overrides", {}) or {}
        unknown = set(overrides) - known_keys
        if unknown:
            raise ValueError(
                f"Action '{name}' uses unknown simulation parameter(s) "
                f"{sorted(unknown)} — valid keys are defined in "
                f"simulation.default_params().")
        horizon = entry.get("horizon", "Tactical")
        if horizon not in VALID_HORIZONS:
            raise ValueError(f"Action '{name}' has invalid horizon '{horizon}' "
                             f"(must be one of {VALID_HORIZONS}).")
        cost = float(entry.get("action_cost_usd", 0.0))
        if cost < 0:
            raise ValueError(f"Action '{name}' has a negative cost.")
        # YAML lists arrive where the simulator expects tuples; normalize
        overrides = {k: _normalize_override(v) for k, v in overrides.items()}
        out[name] = ScenarioSpec(name, str(entry["description"]).strip(),
                                 overrides, action_cost_usd=cost,
                                 horizon=horizon)
    return out


def _normalize_override(value):
    if isinstance(value, list):
        return tuple(_normalize_override(v) for v in value)
    if isinstance(value, dict):
        return {k: _normalize_override(v) for k, v in value.items()}
    return value


# business-language rendering of override keys for the claim-sheet view
def describe_overrides(overrides: dict[str, Any]) -> str:
    """Translate an action's simulation overrides into business language."""
    bits: list[str] = []
    for key, v in overrides.items():
        if key == "overtime_fraction":
            bits.append(f"authorize {v:.0%} of each site's maximum overtime")
        elif key == "overtime_start_month":
            bits.append(f"overtime effective from month {int(v) + 1}")
        elif key == "ems_capacity_add":
            bits += [f"{s}: {u:+g} std-units/mo" for s, u in v.items()]
        elif key == "ems_capacity_add_ramp":
            bits += [f"{s}: {u:+g} std-units/mo from month {int(m) + 1}"
                     for s, (m, u) in v.items()]
        elif key == "comp_supply_mult":
            bits += [f"{'all components' if c == '__all__' else c}: receipts ×{x:g}"
                     for c, x in v.items()]
        elif key == "comp_supply_ramp":
            bits += [f"{'all components' if c == '__all__' else c}: receipts "
                     f"×{x:g} from month {int(m) + 1}" for c, (m, x) in v.items()]
        elif key == "safety_stock_mult":
            bits.append(f"safety-stock policy ×{v:g}")
        elif key == "expedite_recovery":
            bits.append(f"expedite recovery of delayed receipts set to {v:.0%}")
        elif key == "expedite_recovery_by_comp":
            bits += [f"{c}: expedite recovery to {x:.0%} (targeted)"
                     for c, x in v.items()]
        elif key == "expedite_premium_mult":
            bits.append(f"expedite premium ×{v:g}")
        elif key == "integration_capacity_mult":
            bits.append(f"integration capacity ×{v:g}")
        elif key == "integration_capacity_ramp":
            m, x = v
            bits.append(f"integration capacity ×{x:g} from month {int(m) + 1}")
        elif key == "add_qualification":
            bits += [f"{s} qualified for {f} from month {int(m) + 1}"
                     for s, f, m in v]
        else:
            bits.append(f"{key} = {v}")
    return "; ".join(bits)


def action_assumptions_table(
        actions: "dict[str, ScenarioSpec] | None" = None) -> "pd.DataFrame":
    """The full claim-sheet table for the Assumptions & Data page."""
    import pandas as pd
    rows = []
    for name, spec in (actions or management_actions()).items():
        rows.append({
            "Action": name, "Horizon": spec.horizon,
            "Decision cost": spec.action_cost_usd,
            "Authored levers (capability · timing)": describe_overrides(spec.overrides),
        })
    return pd.DataFrame(rows)


def custom_scenario(name: str = "Custom Scenario", **knobs: Any) -> ScenarioSpec:
    """Build a custom scenario from keyword overrides (validated upstream)."""
    return ScenarioSpec(name, "User-defined custom scenario.", knobs)


# ---------------------------------------------------------------------------
# KPI summary and comparison
# ---------------------------------------------------------------------------

def kpi_summary(result: SimulationResult, baseline: BaselineResult,
                config: AppConfig) -> dict[str, Any]:
    """All decision-relevant KPIs for a simulation result."""
    fin = config.financial
    rev_q = quarterly(result.revenue)
    gp_q = quarterly(result.gross_profit)
    plan_q = baseline.revenue_plan_q
    fy_rev = fiscal_year(result.revenue)
    fy_gp = fiscal_year(result.gross_profit)
    fy_plan = plan_q[:4].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        fy_gm = np.where(fy_rev > 0, fy_gp / fy_rev, 0.0)
        q1_gm = np.where(rev_q[:, 0] > 0, gp_q[:, 0] / rev_q[:, 0], 0.0)
    inv_end = result.inventory[:, 11]
    sl = service_level(result)
    avg_inv = result.inventory[:, :12].mean(axis=1)
    fy_cogs = fiscal_year(result.cogs)
    with np.errstate(divide="ignore", invalid="ignore"):
        turns = np.where(avg_inv > 0, fy_cogs / avg_inv, 0.0)

    # past-due, not backlog: simulated demand not yet served (forecast
    # included) — an outcome measure, not the order book
    total_past_due = np.clip(fiscal_year(result.units_demanded)
                             - fiscal_year(result.units_shipped), 0, None)

    return {
        "scenario": result.scenario_name,
        "n_sims": result.n_sims,
        # quarter 1
        "q1_revenue": percentile_stats(rev_q[:, 0]),
        "q1_plan": float(plan_q[0]),
        "p_q1_plan": float((rev_q[:, 0] >= plan_q[0]).mean()),
        "q1_revenue_at_risk": max(0.0, float(plan_q[0] - np.percentile(rev_q[:, 0], 5))),
        "q1_gm": percentile_stats(q1_gm),
        # quarter 2
        "q2_revenue": percentile_stats(rev_q[:, 1]),
        "q2_plan": float(plan_q[1]),
        "p_q2_plan": float((rev_q[:, 1] >= plan_q[1]).mean()),
        # fiscal year
        "fy_revenue": percentile_stats(fy_rev),
        "fy_plan": float(fy_plan),
        "p_fy_plan": float((fy_rev >= fy_plan).mean()),
        "fy_revenue_at_risk": max(0.0, float(fy_plan - np.percentile(fy_rev, 5))),
        "fy_gm": percentile_stats(fy_gm),
        "p_gm_target": float((fy_gm >= fin.gross_margin_target).mean()),
        "gm_target": fin.gross_margin_target,
        "margin_at_risk": max(0.0, float(fin.gross_margin_target - np.percentile(fy_gm, 5))),
        "fy_gross_profit": percentile_stats(fy_gp),
        # full 18-month horizon — the second EV frame for long-lead actions,
        # whose benefits partly fall beyond the fiscal-year window
        "horizon_gross_profit": percentile_stats(result.gross_profit.sum(axis=1)),
        "fy_operating_income": percentile_stats(fiscal_year(result.operating_income)),
        "fy_ebitda": percentile_stats(fiscal_year(result.ebitda)),
        "fy_cash_flow": percentile_stats(fiscal_year(result.cash_flow)),
        # inventory & working capital
        "ending_inventory": percentile_stats(inv_end),
        "avg_inventory": percentile_stats(avg_inv),
        "inventory_target": fin.inventory_target_usd,
        "p_inventory_over_target": float((inv_end > fin.inventory_target_usd).mean()),
        "inventory_turns": percentile_stats(turns),
        "working_capital": percentile_stats(result.working_capital[:, 11]),
        "eo_reserve": percentile_stats(result.eo_reserve),
        # service & operations
        "service_level": percentile_stats(sl),
        "p_missed_commitment": float((sl < 0.95).mean()),
        "p_stockout": float((result.component_short_units.sum(axis=1) > 1.0).mean()),
        "expected_past_due_units": float(total_past_due.mean()),
        "expedite_cost": percentile_stats(fiscal_year(result.expedite_cost)),
        "rework_cost": percentile_stats(fiscal_year(result.rework_cost)),
        "ems_utilization": float(result.ems_utilization[:, :12].mean()),
        "integration_utilization": float(result.integration_utilization[:, :12].mean()),
        "capacity_shortfall_units": float(fiscal_year(result.capacity_shortfall_units).mean()),
        "component_short_units": float(fiscal_year(result.component_short_units).mean()),
    }


def compare_scenarios(base_kpi: dict[str, Any], scen_kpi: dict[str, Any],
                      action_cost_usd: float = 0.0) -> dict[str, Any]:
    """Scenario-versus-base deltas for the comparison views."""
    d_gp = scen_kpi["fy_gross_profit"]["mean"] - base_kpi["fy_gross_profit"]["mean"]
    d_exp = scen_kpi["expedite_cost"]["mean"] - base_kpi["expedite_cost"]["mean"]
    d_p = scen_kpi["p_fy_plan"] - base_kpi["p_fy_plan"]
    d_risk = base_kpi["fy_revenue_at_risk"] - scen_kpi["fy_revenue_at_risk"]
    incr_ev = d_gp - action_cost_usd
    return {
        "scenario": scen_kpi["scenario"],
        "d_q1_revenue": scen_kpi["q1_revenue"]["mean"] - base_kpi["q1_revenue"]["mean"],
        "d_fy_revenue": scen_kpi["fy_revenue"]["mean"] - base_kpi["fy_revenue"]["mean"],
        "d_p_q1_plan": scen_kpi["p_q1_plan"] - base_kpi["p_q1_plan"],
        "d_p_fy_plan": d_p,
        "d_fy_gm": scen_kpi["fy_gm"]["mean"] - base_kpi["fy_gm"]["mean"],
        "d_gross_profit": d_gp,
        "d_inventory": (scen_kpi["ending_inventory"]["mean"]
                        - base_kpi["ending_inventory"]["mean"]),
        "d_working_capital": (scen_kpi["working_capital"]["mean"]
                              - base_kpi["working_capital"]["mean"]),
        "d_expedite": d_exp,
        "d_service": (scen_kpi["service_level"]["mean"]
                      - base_kpi["service_level"]["mean"]),
        "d_revenue_at_risk": -1.0 * (scen_kpi["fy_revenue_at_risk"]
                                     - base_kpi["fy_revenue_at_risk"]),
        "action_cost": action_cost_usd,
        "incremental_ev": incr_ev,
        "risk_reduced_per_dollar": (d_risk / action_cost_usd) if action_cost_usd > 0 else np.nan,
    }
