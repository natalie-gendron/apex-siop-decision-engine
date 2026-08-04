"""Dynamic executive summary generation.

The narrative is produced by a deterministic, rules-based provider — every
number comes from actual simulation output, nothing is hardcoded, and wording
rules guard against false precision (probabilities in whole points, dollars in
$M at one decimal, changes below materiality thresholds are not narrated).

`NarrativeProvider` is the abstraction seam: an LLM-backed provider could be
registered later without touching callers. The prototype intentionally ships
only the rules-based implementation so it runs with no external API.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .models import Recommendation
from .utils import fmt_money, fmt_pts

MATERIAL_REVENUE_USD = 5.0e6     # deltas below this are "broadly unchanged"
MATERIAL_GM_PTS = 0.005
MATERIAL_PROB_PTS = 0.02


@dataclass
class ReportContext:
    """Everything a narrative provider may draw on."""

    base_kpi: dict[str, Any]
    risks: list[dict[str, str]]
    binding: pd.DataFrame
    family_risk: pd.DataFrame
    capacity_risk: pd.DataFrame
    recommendations: list[Recommendation]
    scenario_kpi: dict[str, Any] | None = None
    scenario_compare: dict[str, Any] | None = None
    driver_ranking: pd.DataFrame | None = None
    demand_confidence: Any | None = None     # market_intelligence.DemandConfidence


class NarrativeProvider(ABC):
    """Interface for narrative generation. Version 2 may add an LLM provider."""

    @abstractmethod
    def executive_summary(self, ctx: ReportContext) -> str: ...

    @abstractmethod
    def appendix(self, ctx: ReportContext) -> str: ...


class RulesBasedNarrative(NarrativeProvider):
    """Deterministic template-and-rules narrative generator."""

    # ------------------------------------------------------------------
    def executive_summary(self, ctx: ReportContext) -> str:
        parts = [
            self._outlook(ctx),
            self._revenue_margin_risk(ctx),
            self._operational_constraints(ctx),
            self._inventory_cash(ctx),
            self._decisions(ctx),
        ]
        if ctx.scenario_compare:
            parts.insert(1, self._scenario_delta(ctx))
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    def _outlook(self, ctx: ReportContext) -> str:
        k = ctx.base_kpi
        p_q1, p_fy, p_gm = k["p_q1_plan"], k["p_fy_plan"], k["p_gm_target"]
        gap_q1 = k["q1_revenue"]["median"] - k["q1_plan"]
        tone = ("broadly on track" if p_q1 >= 0.7 and p_gm >= 0.55
                else "achievable but at risk" if p_q1 >= 0.5
                else "unlikely to be achieved without intervention")
        gap_txt = (f"{fmt_money(abs(gap_q1))} {'above' if gap_q1 >= 0 else 'below'} plan "
                   f"at the median")
        text = (
            f"**Overall outlook.** The current plan is {tone}. Simulation places the "
            f"probability of achieving the quarterly revenue plan at {p_q1:.0%}, the "
            f"full-year plan at {p_fy:.0%}, and the {k['gm_target']:.1%} gross-margin "
            f"target at {p_gm:.0%}. Expected Q1 revenue of "
            f"{fmt_money(k['q1_revenue']['median'])} is {gap_txt} "
            f"({fmt_money(k['q1_plan'])}); the credible range (5th-95th percentile) is "
            f"{fmt_money(k['q1_revenue']['p5'])} to {fmt_money(k['q1_revenue']['p95'])}."
        )
        dc = ctx.demand_confidence
        if dc is not None:
            text += (f" Demand Confidence is assessed at {dc.level} "
                     f"({dc.score:.0f}/100); the simulation's demand variance, "
                     f"push-out and cancellation assumptions reflect that "
                     f"assessment.")
        return text

    # ------------------------------------------------------------------
    def _revenue_margin_risk(self, ctx: ReportContext) -> str:
        k = ctx.base_kpi
        fam = ctx.family_risk
        top_two = fam.head(2)
        fam_txt = " and ".join(top_two["product_family"].tolist())
        sentences = [
            f"**Revenue and margin risk.** Downside revenue-at-risk against the "
            f"full-year plan is {fmt_money(k['fy_revenue_at_risk'])} at the 5th "
            f"percentile, concentrated in {fam_txt}."
        ]
        if k["p_gm_target"] < 0.6:
            sentences.append(
                f"Gross margin is expected at {k['fy_gm']['mean']:.1%} against the "
                f"{k['gm_target']:.1%} target; expedite premiums "
                f"({fmt_money(k['expedite_cost']['mean'])} expected for the year) and "
                f"rework are the main cost-side pressures.")
        else:
            sentences.append(
                f"Gross margin is expected at {k['fy_gm']['mean']:.1%}, comfortably "
                f"clearing the target in {k['p_gm_target']:.0%} of simulations.")
        sentences.append(
            f"Timing remains a first-order uncertainty: expected past-due backlog is "
            f"{k['expected_backlog_units']:.0f} systems for the year, and "
            f"{k['p_missed_commitment']:.0%} of simulations show the fill rate "
            f"falling below 95%.")
        return " ".join(sentences)

    # ------------------------------------------------------------------
    def _operational_constraints(self, ctx: ReportContext) -> str:
        k = ctx.base_kpi
        parts = ["**Operational constraints.**"]
        if len(ctx.binding) and ctx.binding.iloc[0]["binding_frequency"] >= 0.05:
            top = ctx.binding.iloc[0]
            parts.append(
                f"{top['component']} is the most frequently binding supply item, "
                f"constraining shipments in {top['binding_frequency']:.0%} of "
                f"simulations.")
        if len(ctx.capacity_risk):
            risky = ctx.capacity_risk.iloc[0]
            if risky["p_capacity_shortfall"] >= 0.2:
                parts.append(
                    f"Capacity risk peaks in {risky['month']}, where "
                    f"{risky['p_capacity_shortfall']:.0%} of simulations show a "
                    f"meaningful shortfall.")
        parts.append(
            f"EMS utilization averages {k['ems_utilization']:.0%} and final "
            f"integration {k['integration_utilization']:.0%}; upside demand beyond "
            f"this level cannot ship without additional capacity or overtime.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    def _inventory_cash(self, ctx: ReportContext) -> str:
        k = ctx.base_kpi
        inv = k["ending_inventory"]
        direction = "above" if inv["mean"] > k["inventory_target"] else "below"
        return (
            f"**Inventory and cash.** Ending fiscal-year inventory is expected at "
            f"{fmt_money(inv['mean'])} ({direction} the {fmt_money(k['inventory_target'])} "
            f"target; {k['p_inventory_over_target']:.0%} probability of exceeding it), "
            f"with working capital of {fmt_money(k['working_capital']['mean'])} and an "
            f"expected excess-and-obsolescence reserve of "
            f"{fmt_money(k['eo_reserve']['mean'])}. Expected full-year operating cash "
            f"flow is {fmt_money(k['fy_cash_flow']['mean'])}. Customer push-outs and "
            f"acceptance delays convert directly into finished-goods inventory, so "
            f"revenue protection and working capital trade off against each other."
        )

    # ------------------------------------------------------------------
    def _decisions(self, ctx: ReportContext) -> str:
        recs = ctx.recommendations
        positive = [r for r in recs if r.expected_value_usd > 0]
        if not positive:
            return ("**Recommended decisions.** No modeled management action clears "
                    "the expected-value bar this cycle; the recommendation is to "
                    "hold current commitments and revisit at the next SIOP meeting.")
        lines = ["**Recommended decisions.**"]
        for i, r in enumerate(positive[:3], 1):
            lines.append(
                f"({i}) {r.title}: {fmt_pts(r.prob_plan_improvement)} on plan "
                f"attainment, {fmt_money(r.gross_profit_protected_usd)} expected gross "
                f"profit protected, at {fmt_money(r.incremental_cost_usd)} incremental "
                f"cost.")
        if len(positive) >= 2:
            lines.append(
                "Benefits of these actions overlap where they relieve the same "
                "constraint; sequencing matters more than adopting all of them at "
                "once.")
        return " ".join(lines)

    # ------------------------------------------------------------------
    def _scenario_delta(self, ctx: ReportContext) -> str:
        c = ctx.scenario_compare
        k = ctx.scenario_kpi
        assert c is not None and k is not None
        bits = [f"**Scenario: {k['scenario']}.**"]
        if abs(c["d_fy_revenue"]) >= MATERIAL_REVENUE_USD:
            bits.append(
                f"Versus the base case, expected full-year revenue moves "
                f"{fmt_money(c['d_fy_revenue'])} and plan-attainment probability "
                f"{fmt_pts(c['d_p_fy_plan'])}.")
        else:
            bits.append("Versus the base case, expected full-year revenue is broadly "
                        "unchanged.")
        if abs(c["d_fy_gm"]) >= MATERIAL_GM_PTS:
            bits.append(f"Gross margin shifts {fmt_pts(c['d_fy_gm'])}.")
        if abs(c["d_inventory"]) >= MATERIAL_REVENUE_USD:
            bits.append(f"Ending inventory moves {fmt_money(c['d_inventory'])}.")
        if c["action_cost"] > 0:
            ev = c["incremental_ev"]
            verdict = "positive" if ev > 0 else "negative"
            bits.append(
                f"Net of the {fmt_money(c['action_cost'])} decision cost, incremental "
                f"expected value is {fmt_money(ev)} ({verdict}).")
        return " ".join(bits)

    # ------------------------------------------------------------------
    def appendix(self, ctx: ReportContext) -> str:
        k = ctx.base_kpi
        lines = [
            "### Appendix — detailed readout",
            "",
            f"- Simulations: {k['n_sims']:,} correlated Monte Carlo paths.",
            f"- Q1 revenue: mean {fmt_money(k['q1_revenue']['mean'])}, median "
            f"{fmt_money(k['q1_revenue']['median'])}, P5 {fmt_money(k['q1_revenue']['p5'])}, "
            f"P95 {fmt_money(k['q1_revenue']['p95'])}, plan {fmt_money(k['q1_plan'])}.",
            f"- FY revenue: mean {fmt_money(k['fy_revenue']['mean'])}, P5 "
            f"{fmt_money(k['fy_revenue']['p5'])}, P95 {fmt_money(k['fy_revenue']['p95'])}, "
            f"plan {fmt_money(k['fy_plan'])} (P(plan) {k['p_fy_plan']:.0%}).",
            f"- FY gross margin: mean {k['fy_gm']['mean']:.1%}, P5 {k['fy_gm']['p5']:.1%}, "
            f"P95 {k['fy_gm']['p95']:.1%}, target {k['gm_target']:.1%} "
            f"(P(target) {k['p_gm_target']:.0%}).",
            f"- FY operating income: mean {fmt_money(k['fy_operating_income']['mean'])}; "
            f"EBITDA proxy {fmt_money(k['fy_ebitda']['mean'])}; operating cash-flow "
            f"proxy {fmt_money(k['fy_cash_flow']['mean'])}.",
            f"- Ending inventory: mean {fmt_money(k['ending_inventory']['mean'])} "
            f"(target {fmt_money(k['inventory_target'])}); inventory turns "
            f"{k['inventory_turns']['mean']:.1f}; working capital "
            f"{fmt_money(k['working_capital']['mean'])}.",
            f"- Expected expedite spend {fmt_money(k['expedite_cost']['mean'])}; rework "
            f"{fmt_money(k['rework_cost']['mean'])}; E&O reserve "
            f"{fmt_money(k['eo_reserve']['mean'])}.",
            f"- Service: expected fill rate {k['service_level']['mean']:.1%}; "
            f"P(stockout event) {k['p_stockout']:.0%}; P(missed commitment) "
            f"{k['p_missed_commitment']:.0%}.",
            "",
            "Association-based driver rankings, binding-constraint frequencies and "
            "scenario deltas are available on the Risk Drivers and Scenario "
            "Comparison pages. Rankings use Spearman rank correlation and indicate "
            "association, not causation.",
        ]
        if ctx.driver_ranking is not None and len(ctx.driver_ranking):
            top = ctx.driver_ranking.head(3)
            drivers = ", ".join(
                f"{r.driver} (ρ={r.spearman_rho:+.2f})" for r in top.itertuples())
            lines.insert(-1, f"- Top FY-revenue drivers by rank correlation: {drivers}.")
        return "\n".join(lines)


def get_provider(name: str = "rules") -> NarrativeProvider:
    """Provider registry. Only 'rules' ships in Version 1; an 'llm' provider can
    be registered here later without changing any caller."""
    if name == "rules":
        return RulesBasedNarrative()
    raise ValueError(f"Unknown narrative provider '{name}'. Available: rules")
