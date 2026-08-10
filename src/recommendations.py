"""Transparent, evidence-based recommendation engine.

Every recommendation is tied to (a) a detected risk with a measurable
threshold, (b) a management action actually simulated against the base case
with common random numbers, and (c) quantified benefits, costs and trade-offs.
Actions with unsupported or immaterial modeled benefit are not recommended.

Scoring (visible, documented):
    score = EV_norm * 0.35 + prob_norm * 0.30 + revenue_norm * 0.20
            - cash_use_norm * 0.15
where each term is normalized against the best action in the set. Actions must
clear BOTH gates to appear: expected value > -$1M AND (probability improvement
>= 1 point OR expected value >= $2M).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .models import Recommendation, ScenarioSpec
from .utils import fmt_money, fmt_pts

# risk-detection thresholds (measurable triggers, shown in the UI)
THRESHOLDS = {
    "p_q1_plan_min": 0.75,       # Q1 plan attainment probability below this = risk
    "p_fy_plan_min": 0.65,
    "p_gm_target_min": 0.60,
    "service_level_min": 0.95,
    "p_inventory_over_max": 0.35,
    "binding_freq_min": 0.10,    # a component binding in >=10% of sims = risk
}

SCORE_WEIGHTS = {"ev": 0.35, "prob": 0.30, "revenue": 0.20, "cash": -0.15}


def detect_risks(base_kpi: dict[str, Any], binding: "pd.DataFrame",
                 demand_confidence=None) -> list[dict[str, str]]:
    """Detected risks/opportunities with the threshold that fired."""
    risks = []
    if demand_confidence is not None and demand_confidence.level in ("Low", "Very Low"):
        risks.append({
            "risk": (f"Demand Confidence is {demand_confidence.level} "
                     f"({demand_confidence.score:.0f}/100), widening the demand "
                     f"distribution and raising push-out and cancellation rates "
                     f"in the simulation"),
            "threshold": "Demand Confidence at or below Low",
            "tag": "revenue"})
    if base_kpi["p_q1_plan"] < THRESHOLDS["p_q1_plan_min"]:
        risks.append({
            "risk": (f"Q1 revenue-plan attainment probability is "
                     f"{base_kpi['p_q1_plan']:.0%} with "
                     f"{fmt_money(base_kpi['q1_revenue_at_risk'])} at risk (P5 vs plan)"),
            "threshold": f"P(Q1 plan) < {THRESHOLDS['p_q1_plan_min']:.0%}",
            "tag": "revenue"})
    if base_kpi["p_fy_plan"] < THRESHOLDS["p_fy_plan_min"]:
        risks.append({
            "risk": (f"Full-year plan attainment probability is "
                     f"{base_kpi['p_fy_plan']:.0%} with "
                     f"{fmt_money(base_kpi['fy_revenue_at_risk'])} at risk"),
            "threshold": f"P(FY plan) < {THRESHOLDS['p_fy_plan_min']:.0%}",
            "tag": "revenue"})
    if base_kpi["p_gm_target"] < THRESHOLDS["p_gm_target_min"]:
        risks.append({
            "risk": (f"Gross-margin target attainment probability is "
                     f"{base_kpi['p_gm_target']:.0%}"),
            "threshold": f"P(GM target) < {THRESHOLDS['p_gm_target_min']:.0%}",
            "tag": "margin"})
    if base_kpi["service_level"]["mean"] < THRESHOLDS["service_level_min"]:
        risks.append({
            "risk": (f"Expected FY fill rate is "
                     f"{base_kpi['service_level']['mean']:.1%}, below the 95% "
                     f"service objective"),
            "threshold": f"service level < {THRESHOLDS['service_level_min']:.0%}",
            "tag": "service"})
    if base_kpi["p_inventory_over_target"] > THRESHOLDS["p_inventory_over_max"]:
        risks.append({
            "risk": (f"Probability of exceeding the inventory target is "
                     f"{base_kpi['p_inventory_over_target']:.0%}"),
            "threshold": f"P(inventory > target) > {THRESHOLDS['p_inventory_over_max']:.0%}",
            "tag": "inventory"})
    for _, row in binding.iterrows():
        if row["binding_frequency"] >= THRESHOLDS["binding_freq_min"]:
            risks.append({
                "risk": (f"{row['component']} constrains shipments in "
                         f"{row['binding_frequency']:.0%} of simulations"),
                "threshold": f"binding frequency >= {THRESHOLDS['binding_freq_min']:.0%}",
                "tag": "component"})
    return risks


# maps action names to the risk tags they address and standing caveats
ACTION_RISK_MAP: dict[str, tuple[list[str], str]] = {
    "Reserve additional EMS capacity": (
        ["revenue", "service"],
        "Benefit is limited when component supply, not capacity, is the binding "
        "constraint; reservation fees are committed even if demand softens."),
    "Authorize overtime at EMS sites": (
        ["revenue", "service"],
        "Sustained overtime erodes yield and adherence in practice; modeled as "
        "cost-only here."),
    "Expedite critical component receipts": (
        ["revenue", "component", "service"],
        "Premiums land in COGS and compress gross margin; benefit shrinks if a "
        "major customer push-out also occurs."),
    "Expedite high-end FPGA receipts only": (
        ["component", "revenue"],
        "Expediting recovers late receipts, not cut allocations — if supply is "
        "reduced at the source, this buys timing, not volume."),
    "Increase FPGA safety stock": (
        ["component", "revenue"],
        "Adds raw-material inventory and E&O exposure if AI demand softens."),
    "Dual-source the high-end FPGA": (
        ["component", "revenue"],
        "No relief during the six-month qualification window; benefits accrue in "
        "the second half of the year."),
    "Shift eligible builds to EMS Taiwan": (
        ["revenue", "service"],
        "Transition friction temporarily reduces EMS Malaysia output; regional "
        "concentration risk increases."),
    "Add temporary integration capacity": (
        ["revenue", "service"],
        "Contract labor first-pass completion is typically lower; ramp time is "
        "not modeled."),
    "Pre-build standard subassemblies": (
        ["component", "revenue", "inventory"],
        "Increases WIP and raw inventory; exposure if configurations change."),
    "Accept shipment risk (no extraordinary cost)": (
        ["margin", "inventory"],
        "Protects margin and cash at the cost of service level and revenue "
        "timing; customer-relationship impact is not modeled."),
    "Inventory reduction initiative": (
        ["inventory"],
        "Releases working capital by accepting stockout exposure; revenue and "
        "service downside grows if demand firms or supply tightens after "
        "stocks are cut."),
    "Commit long-lead component orders": (
        ["component", "revenue"],
        "Non-cancellable commitments become E&O and cash exposure if demand "
        "softens before receipts arrive; no benefit inside the current quarter."),
    "Qualify EMS Eastern Europe for Zenith Compute": (
        ["revenue", "service"],
        "No relief for roughly three quarters; qualification timelines slip in "
        "practice, and early builds at a new site typically run lower yield."),
    "Expand final-integration headcount": (
        ["revenue", "service"],
        "Permanent fixed-cost addition justified only by a sustained demand "
        "level; benefit starts in the second half of the horizon."),
}


def build_recommendations(
    base_kpi: dict[str, Any],
    action_results: dict[str, tuple[dict[str, Any], ScenarioSpec]],
    binding, min_ev_usd: float = -1.0e6, demand_confidence=None,
) -> list[Recommendation]:
    """Score simulated actions against detected risks; emit ranked recommendations."""
    import pandas as pd  # local import to keep module load light

    risks = detect_risks(base_kpi, binding, demand_confidence)
    if not risks:
        return []
    risk_by_tag: dict[str, dict[str, str]] = {}
    for r in risks:
        risk_by_tag.setdefault(r["tag"], r)

    candidates = []
    for name, (kpi, spec) in action_results.items():
        tags, caveat = ACTION_RISK_MAP.get(name, ([], ""))
        matched = [risk_by_tag[t] for t in tags if t in risk_by_tag]
        if not matched:
            continue
        d_gp = kpi["fy_gross_profit"]["mean"] - base_kpi["fy_gross_profit"]["mean"]
        d_rev = kpi["fy_revenue"]["mean"] - base_kpi["fy_revenue"]["mean"]
        d_p = kpi["p_fy_plan"] - base_kpi["p_fy_plan"]
        d_p_q1 = kpi["p_q1_plan"] - base_kpi["p_q1_plan"]
        d_inv = kpi["ending_inventory"]["mean"] - base_kpi["ending_inventory"]["mean"]
        d_wc = kpi["working_capital"]["mean"] - base_kpi["working_capital"]["mean"]
        d_sl = kpi["service_level"]["mean"] - base_kpi["service_level"]["mean"]
        ev = d_gp - spec.action_cost_usd
        # materiality gates
        if ev < min_ev_usd:
            continue
        if max(d_p, d_p_q1) < 0.01 and ev < 2.0e6:
            continue
        candidates.append({
            "name": name, "spec": spec, "risk": matched[0],
            "ev": ev, "d_rev": d_rev, "d_gp": d_gp, "d_p": d_p,
            "d_p_q1": d_p_q1, "d_inv": d_inv, "d_wc": d_wc, "d_sl": d_sl,
            "caveat": caveat,
        })

    if not candidates:
        return []

    df = pd.DataFrame(candidates)
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 1e-9 else s * 0.0
    df["score"] = (SCORE_WEIGHTS["ev"] * norm(df["ev"])
                   + SCORE_WEIGHTS["prob"] * norm(df[["d_p", "d_p_q1"]].max(axis=1))
                   + SCORE_WEIGHTS["revenue"] * norm(df["d_rev"])
                   + SCORE_WEIGHTS["cash"] * norm(df["d_wc"].clip(lower=0)))

    recs = []
    for _, row in df.sort_values("score", ascending=False).iterrows():
        prob_gain = max(row["d_p"], row["d_p_q1"])
        confidence = ("High" if row["ev"] > 5e6 and prob_gain >= 0.05
                      else "Medium" if row["ev"] > 1e6 or prob_gain >= 0.03
                      else "Low")
        caveat = row["caveat"]
        # executive explainability: reference Demand Confidence where it
        # materially changes how the action should be read
        if demand_confidence is not None:
            level = demand_confidence.level
            if level in ("Low", "Very Low") and row["d_inv"] > 2e6:
                caveat += (f" Note: Demand Confidence is {level}, so inventory "
                           f"committed by this action carries elevated E&O risk "
                           f"if the softness materializes.")
            elif level in ("Low", "Very Low"):
                caveat += (f" This action's value is amplified by {level} Demand "
                           f"Confidence, which widens the downside it protects "
                           f"against.")
            elif level in ("High", "Very High") and row["d_inv"] > 2e6:
                caveat += (f" With {level} Demand Confidence, the downside this "
                           f"insures against is less likely; weigh the "
                           f"working-capital cost accordingly.")
        recs.append(Recommendation(
            title=row["name"],
            risk=row["risk"]["risk"],
            threshold=row["risk"]["threshold"],
            action=row["spec"].description,
            expected_value_usd=row["ev"],
            revenue_protected_usd=row["d_rev"],
            gross_profit_protected_usd=row["d_gp"],
            prob_plan_improvement=prob_gain,
            incremental_cost_usd=row["spec"].action_cost_usd,
            inventory_change_usd=row["d_inv"],
            working_capital_change_usd=row["d_wc"],
            service_level_change=row["d_sl"],
            score=float(row["score"]),
            confidence=confidence,
            caveat=caveat,
        ))
    return recs
