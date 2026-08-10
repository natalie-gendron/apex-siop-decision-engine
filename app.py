"""Apex Test Systems — AI-enabled SIOP Risk & Scenario Engine (Streamlit app).

All data is synthetic. Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.baseline_plan import run_baseline
from src.config import load_config
from src.correlations import FactorEngine
from src.data_generator import generate_all, load_or_generate
from src.executive_report import ReportContext, get_provider
from src.exports import build_excel_export
from src.market_intelligence import (
    CONFIDENCE_LEVELS,
    assumptions_in_business_language,
    build_demand_confidence,
    confidence_mapping_table,
    merge_confidence_params,
    parse_curated_external,
    CONFIDENCE_SIM_PARAMS,
)
from src.models import InputData
from src.recommendations import THRESHOLDS, build_recommendations, detect_risks
from src.scenarios import (
    action_assumptions_table,
    compare_scenarios,
    custom_scenario,
    describe_overrides,
    kpi_summary,
    management_actions,
    prebuilt_scenarios,
)
from src.sensitivity import (
    all_driver_rankings,
    binding_components,
    expected_past_due_by_family,
    expected_past_due_curve,
    family_revenue_at_risk,
    monthly_capacity_risk,
    quarter_shift_drivers,
    site_disruption_frequency,
)
from src.simulation import run_simulation
from src.utils import MARKET_SEGMENTS, fmt_money, fmt_pct, fmt_pts, month_labels
from src.validation import validate_inputs
from src import visualizations as viz

st.set_page_config(page_title="Apex SIOP Decision Engine", page_icon="📊",
                   layout="wide")

CONFIG = load_config()
SIM_MODES = {"Quick (1,000)": 1000, "Standard (5,000)": 5000,
             "Detailed (10,000)": 10000}


# ---------------------------------------------------------------------------
# Cached pipeline
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_data(seed: int) -> InputData:
    return load_or_generate(seed=seed)


@st.cache_data(show_spinner=False)
def cached_regenerate(seed: int, nonce: int) -> InputData:
    return generate_all(seed=seed)


@st.cache_data(show_spinner=False)
def cached_baseline(seed: int, _data: InputData):
    return run_baseline(_data, CONFIG)


def _params_key(params: dict) -> str:
    """JSON-serializable cache key for a params dict (arrays -> lists)."""
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    return json.dumps(conv(params), sort_keys=True)


@st.cache_data(show_spinner=False)
def cached_simulation(data_seed: int, sim_seed: int, n_sims: int,
                      scenario_name: str, params_key: str,
                      _data: InputData, _baseline, _params: dict):
    # data_seed is part of the cache key so a new synthetic company invalidates
    # cached simulations (the _data/_baseline args themselves are not hashed)
    return run_simulation(_data, CONFIG, _baseline, params=_params, n_sims=n_sims,
                          seed=sim_seed, scenario_name=scenario_name)


EXTERNAL_INTEL_PATH = Path(__file__).parent / "data" / "external_intel.csv"


@st.cache_data(show_spinner=False)
def cached_confidence(data_seed: int, intel_text: str | None, _data: InputData):
    external = parse_curated_external(intel_text) if intel_text else None
    return build_demand_confidence(_data, CONFIG, seed=data_seed,
                                   external=external)


@st.cache_data(show_spinner=False)
def cached_actions(data_seed: int, sim_seed: int, n_sims: int,
                   confidence_level: str, context_key: str, catalog_key: str,
                   _data: InputData, _baseline, _context: dict,
                   _catalog: dict):
    """Evaluate all actions on a given backdrop: confidence + optional scenario
    context (empty context = the standing base-case evaluation). catalog_key
    is derived from the claim-sheet contents, so edited or what-if catalogs
    reprice instead of returning stale cached results."""
    conf = CONFIDENCE_SIM_PARAMS[confidence_level]
    out = {}
    for name, spec in _catalog.items():
        stacked = merge_confidence_params(_context, spec.overrides)
        r = run_simulation(_data, CONFIG, _baseline,
                           params=merge_confidence_params(conf, stacked),
                           n_sims=n_sims, seed=sim_seed, scenario_name=name)
        out[name] = (kpi_summary(r, _baseline, CONFIG), spec)
    return out


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Apex SIOP Engine")
st.sidebar.caption("Synthetic executive prototype — Apex Test Systems (fictional)")

data_seed = int(st.sidebar.number_input(
    "Input data seed", 1, 10_000, CONFIG.random_seed,
    help="Seeds the generated input data: demand book, backlog, prices, plan "
         "and capacity tables. Only applies while inputs are synthetic — once "
         "connected to real ERP/CRM data, this control is retired."))
sim_seed = int(st.sidebar.number_input(
    "Simulation seed", 1, 10_000, CONFIG.random_seed,
    help="Drives only the Monte Carlo draws (the what-ifs). With enough "
         "simulations, results should barely move when this changes — a "
         "convergence sanity check."))
mode = st.sidebar.selectbox("Simulation mode", list(SIM_MODES), index=1)
n_sims = SIM_MODES[mode]

# Evaluation context = (world, response): a scenario describes the world
# (exogenous, no cost); a response package is what Apex chooses to do
# (costed, latency-ramped actions from the claim-sheet catalog).
scen_names = list(prebuilt_scenarios()) + ["Custom Scenario"]
scenario_name = st.sidebar.selectbox(
    "Scenario (the world)", scen_names, index=0,
    help="Exogenous demand/supply assumptions — what could happen TO the "
         "business. Base Case = current SIOP assumptions.")

# Demand Confidence override widget lives here (world axis: scenario, then
# calibration), but its options depend on the assessed level, which needs the
# input data — reserve the slot now, fill it after the assessment is built.
confidence_slot = st.sidebar.container()

# action catalog: the repo claim-sheet file, or a session-only what-if
# upload from the Assumptions & Data page (demo of the analyst edit loop)
_catalog_text = st.session_state.get("catalog_override_text")
try:
    ACTIONS_CATALOG = management_actions(catalog_text=_catalog_text)
except (ValueError, KeyError, TypeError) as exc:
    st.sidebar.warning(f"Session action catalog could not be used ({exc}); "
                       "reverted to the repo catalog.")
    st.session_state.pop("catalog_override_text", None)
    _catalog_text = None
    ACTIONS_CATALOG = management_actions()
# content-derived cache key: action pricing re-runs when the claims change
catalog_key = _params_key({n: [s.horizon, s.action_cost_usd, s.overrides]
                           for n, s in ACTIONS_CATALOG.items()})
if _catalog_text:
    st.sidebar.info(f"Action catalog: session what-if "
                    f"({len(ACTIONS_CATALOG)} actions). Not saved — the repo "
                    f"file is the record.")
    if st.sidebar.button("Revert to repo action catalog"):
        st.session_state.pop("catalog_override_text", None)
        st.session_state["catalog_upload_nonce"] = (
            st.session_state.get("catalog_upload_nonce", 0) + 1)
        st.rerun()

st.session_state["package_names"] = [
    n for n in st.session_state.get("package_names", [])
    if n in ACTIONS_CATALOG]
package_names = st.sidebar.multiselect(
    "Response package (what we will do)", list(ACTIONS_CATALOG),
    key="package_names",
    help="Management actions evaluated together on top of the selected "
         "scenario, with their combined cost. Empty = no action taken. "
         "Also editable via the checkboxes on Management Recommendations.")

custom_params: dict = {}
if scenario_name == "Custom Scenario":
    with st.sidebar.expander("Custom scenario controls", expanded=True):
        dm = {}
        for mkt in MARKET_SEGMENTS:
            v = st.slider(f"{mkt} demand", 0.6, 1.5, 1.0, 0.05)
            if abs(v - 1.0) > 1e-9:
                dm[mkt] = v
        if dm:
            custom_params["demand_market_mult"] = dm
        push = st.slider("Push-out probability (+pts)", 0.0, 0.25, 0.0, 0.01)
        pull = st.slider("Pull-in probability (+pts)", 0.0, 0.15, 0.0, 0.01)
        cancel = st.slider("Cancellation multiplier", 0.5, 3.0, 1.0, 0.1)
        aspm = st.slider("ASP multiplier", 0.9, 1.1, 1.0, 0.01)
        lt = st.slider("Component lead-time multiplier", 0.8, 1.8, 1.0, 0.05)
        dis = st.slider("Supplier disruption multiplier", 0.5, 3.0, 1.0, 0.1)
        emsc = st.slider("EMS capacity multiplier", 0.7, 1.3, 1.0, 0.02)
        fpy = st.slider("EMS yield delta (pts)", -0.08, 0.05, 0.0, 0.01)
        adh = st.slider("Schedule adherence delta (pts)", -0.08, 0.04, 0.0, 0.01)
        integ = st.slider("Integration capacity multiplier", 0.8, 1.3, 1.0, 0.02)
        freight = st.slider("Freight cost multiplier", 0.8, 2.0, 1.0, 0.05)
        exp_p = st.slider("Expedite premium multiplier", 0.5, 2.5, 1.0, 0.1)
        acc = st.slider("Acceptance delay probability (+pts)", 0.0, 0.3, 0.0, 0.02)
        st.caption("These knobs describe the world (demand and supply "
                   "environment). Things Apex chooses to do — safety stock, "
                   "reserved capacity, overtime — are management actions; add "
                   "them to the response package instead.")
        for key, val, default in [
            ("pushout_prob_add", push, 0.0), ("pullin_prob_add", pull, 0.0),
            ("cancel_prob_mult", cancel, 1.0), ("asp_mult", aspm, 1.0),
            ("lead_time_mult", lt, 1.0), ("comp_disrupt_mult", dis, 1.0),
            ("fpy_delta", fpy, 0.0), ("adherence_delta", adh, 0.0),
            ("integration_capacity_mult", integ, 1.0),
            ("freight_mult", freight, 1.0), ("expedite_premium_mult", exp_p, 1.0),
            ("acceptance_delay_add", acc, 0.0),
        ]:
            if abs(val - default) > 1e-9:
                custom_params[key] = val
        if abs(emsc - 1.0) > 1e-9:
            custom_params["ems_capacity_mult"] = {
                s: emsc for s in ["EMS Americas", "EMS Malaysia", "EMS Taiwan",
                                  "EMS Eastern Europe"]}

if st.sidebar.button("Regenerate synthetic data",
                     help="Rebuild all synthetic tables with the selected seed."):
    st.session_state["regen_nonce"] = st.session_state.get("regen_nonce", 0) + 1
    cached_data.clear()
    cached_baseline.clear()
    cached_simulation.clear()
    cached_actions.clear()

st.sidebar.divider()
st.sidebar.caption(
    "Data, plans and results are entirely synthetic and generated locally. "
    "Monte Carlo results are reproducible for a given seed.")

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

data = (cached_regenerate(data_seed, st.session_state["regen_nonce"])
        if st.session_state.get("regen_nonce") else cached_data(data_seed))

issues = validate_inputs(data)
errors = [i for i in issues if i.severity == "error"]
warnings = [i for i in issues if i.severity == "warning"]
if errors:
    st.error("Input validation failed — fix the data before running the model:")
    for i in errors:
        st.error(f"[{i.area}] {i.message}")
    st.stop()

with st.spinner("Building deterministic baseline plan..."):
    baseline = cached_baseline(data_seed, data)

# Demand Confidence assessment — feeds every simulation as an uncertainty driver.
# A curated external-intelligence file (analyst- or Claude-produced) replaces
# the synthetic external templates when present.
intel_text = (EXTERNAL_INTEL_PATH.read_text(encoding="utf-8")
              if EXTERNAL_INTEL_PATH.exists() else None)
try:
    dc = cached_confidence(data_seed, intel_text, data)
except (ValueError, KeyError) as exc:
    st.warning(f"Curated external-intelligence file could not be used "
               f"({exc}); falling back to synthetic external intelligence.")
    dc = cached_confidence(data_seed, None, data)

# Confidence is calibration, not hypothesis (see ARCHITECTURE.md): the
# assessed level applies to every simulation by default; the sidebar override
# allows what-ifs at other levels but is always labeled as an override.
with confidence_slot:
    conf_choice = st.selectbox(
        "Demand confidence (trust in the plan)",
        ["Assessed"] + CONFIDENCE_LEVELS,
        format_func=lambda o: (f"Assessed — {dc.level}" if o == "Assessed"
                               else o),
        help="How much to trust the demand plan, applied to every simulation. "
             "'Assessed' uses the level derived from market signals and "
             "external intelligence (Market Intelligence tab); picking a "
             "level instead is a what-if override. Confidence never rewrites "
             "the demand plan — it calibrates demand volatility, push-out "
             "and cancellation assumptions around it.")
    with st.popover("What each level does", width="stretch"):
        st.dataframe(confidence_mapping_table(
            dc.level if conf_choice == "Assessed" else conf_choice,
            plain_language=False), width='stretch', hide_index=True)
        st.caption("→ marks the level being applied. Moderate = the base "
                   "rates estimated from history, unadjusted.")
effective_level = dc.level if conf_choice == "Assessed" else conf_choice
confidence_overridden = effective_level != dc.level
if confidence_overridden:
    confidence_slot.caption(f"Override active — signals assess "
                            f"**{dc.level}**; simulations run at "
                            f"**{effective_level}**.")
conf_params = dict(CONFIDENCE_SIM_PARAMS[effective_level])

scenarios = prebuilt_scenarios()
if scenario_name == "Custom Scenario":
    spec = custom_scenario(**custom_params)
else:
    spec = scenarios[scenario_name]

progress = st.progress(0.0, text="Running Monte Carlo simulation...")


def _cb(frac: float, msg: str) -> None:
    progress.progress(frac, text=msg)


def sim_with_confidence(n: int, name: str, overrides: dict):
    """All simulations run on the Demand Confidence backdrop."""
    merged = merge_confidence_params(conf_params, overrides)
    return cached_simulation(data_seed, sim_seed, n, name, _params_key(merged),
                             data, baseline, merged)


# ---- evaluation context: (world, response) --------------------------------
package_specs = [ACTIONS_CATALOG[n] for n in package_names]
package_cost = float(sum(a.action_cost_usd for a in package_specs))
package_overrides: dict = {}
for a in package_specs:   # later actions win collisions; adds sum, mults multiply
    package_overrides = merge_confidence_params(package_overrides, a.overrides)
package_label = (f"response package ({len(package_specs)} "
                 f"action{'s' if len(package_specs) != 1 else ''})")

base_result = sim_with_confidence(n_sims, "Base Case", {})
world_result = (base_result if spec.name == "Base Case" else
                sim_with_confidence(n_sims, spec.name, spec.overrides))
if package_specs:
    context_label = ((f"{spec.name} + " if spec.name != "Base Case" else "")
                     + package_label)
    final_overrides = merge_confidence_params(spec.overrides, package_overrides)
    # the sim spreads the one-time decision cost over Q1 operating income
    final_overrides["action_cost_usd"] = package_cost
    ctx_result = sim_with_confidence(n_sims, context_label, final_overrides)
else:
    context_label = spec.name
    ctx_result = world_result

progress.progress(0.7, text="Scoring management actions...")
action_results = cached_actions(data_seed, sim_seed, min(n_sims, 2000),
                                effective_level, "base", catalog_key,
                                data, baseline, {}, ACTIONS_CATALOG)
progress.progress(1.0, text="Done")
progress.empty()

base_kpi = kpi_summary(base_result, baseline, CONFIG)
world_kpi = (base_kpi if world_result is base_result else
             kpi_summary(world_result, baseline, CONFIG))
ctx_kpi = (world_kpi if ctx_result is world_result else
           kpi_summary(ctx_result, baseline, CONFIG))
ctx_cmp = (compare_scenarios(base_kpi, ctx_kpi, package_cost)
           if ctx_result is not base_result else None)

binding = binding_components(base_result)
fam_risk = family_revenue_at_risk(base_result, baseline)
cap_risk = monthly_capacity_risk(base_result)
rankings = all_driver_rankings(base_result)

# context-conditioned views for the outcome pages (identical to the base
# views when the context is (Base Case, no actions))
if ctx_result is base_result:
    ctx_binding, ctx_fam_risk, ctx_cap_risk = binding, fam_risk, cap_risk
    ctx_rankings = rankings
else:
    ctx_binding = binding_components(ctx_result)
    ctx_fam_risk = family_revenue_at_risk(ctx_result, baseline)
    ctx_cap_risk = monthly_capacity_risk(ctx_result)
    ctx_rankings = all_driver_rankings(ctx_result)
ctx_suffix = "" if ctx_result is base_result else f" — {context_label}"
risks = detect_risks(base_kpi, binding, dc)
recs = build_recommendations(base_kpi, action_results, binding,
                             demand_confidence=dc)

# plan-of-record narrative — anchors the Excel export regardless of context
provider = get_provider("rules")
report_ctx = ReportContext(kpi=base_kpi, risks=risks, binding=binding,
                           family_risk=fam_risk, capacity_risk=cap_risk,
                           recommendations=recs,
                           scenario_kpi=ctx_kpi if ctx_cmp else None,
                           scenario_compare=ctx_cmp,
                           driver_ranking=rankings["FY revenue"],
                           demand_confidence=dc)
summary_text = provider.executive_summary(report_ctx)

# context-conditioned counterparts for the Executive Overview (outcome page)
if ctx_result is base_result:
    ctx_risks, overview_summary = risks, summary_text
else:
    ctx_risks = detect_risks(ctx_kpi, ctx_binding, dc)
    overview_summary = provider.executive_summary(ReportContext(
        kpi=ctx_kpi, risks=ctx_risks, binding=ctx_binding,
        family_risk=ctx_fam_risk, capacity_risk=ctx_cap_risk,
        recommendations=recs, scenario_kpi=ctx_kpi, scenario_compare=ctx_cmp,
        driver_ranking=ctx_rankings["FY revenue"], demand_confidence=dc,
        conditioned_on=context_label))

# ---------------------------------------------------------------------------
# Header + tabs
# ---------------------------------------------------------------------------

def md(text: str) -> str:
    """Escape $ so Streamlit's markdown doesn't render dollar amounts as LaTeX."""
    return text.replace("$", "\\$")


st.title("Apex Test Systems — SIOP Risk & Scenario Engine")
st.caption(f"18-month horizon from 2026-07 · scenario: **{spec.name}** · "
           f"response package: "
           f"**{f'{len(package_specs)} actions' if package_specs else 'none'}** · "
           f"demand confidence: **{effective_level}"
           f"{' — overridden' if confidence_overridden else ''}** · "
           f"{n_sims:,} simulations · "
           f"data seed {data_seed} · sim seed {sim_seed} · all data synthetic")

# Tab order deliberately follows the SIOP process: the executive answer
# first, then demand review, supply review, integrated reconciliation,
# decision support, and governance. The last tab carries the process guide.
tabs = st.tabs([
    "Executive Overview", "Demand & Backlog",
    "Market Intelligence & Demand Confidence", "Supply & Component Risk",
    "Manufacturing Capacity", "Financial Outcomes", "Risk Drivers",
    "Scenario Comparison", "Management Recommendations", "Assumptions & Data",
    "Guide, Methodology & Export",
])

# ---------------------------- 1. Executive Overview ------------------------
# Outcome page: everything follows the evaluation context (world, response) —
# design rule: a view is either plan-of-record or outcome. Plan/targets stay
# the frozen anchors in the tile deltas; the strip below the tiles prices the
# context vs the standing base outlook; the bridge decomposes the move.
with tabs[0]:
    k = ctx_kpi
    with st.expander("How to read this page"):
        st.markdown(md(
            "Every view in this app is one of two kinds: **plan-of-record** "
            "(frozen anchors — the plan, targets, the deterministic baseline) "
            "or **outcome** (follows the evaluation context you set in the "
            "sidebar). The context has two parts: a **scenario** — the world, "
            "what could happen *to* Apex (exogenous, no cost) — and a "
            "**response package** — what Apex chooses to *do* (costed, "
            "latency-ramped management actions).\n\n"
            "This page is an outcome page. The tiles answer *\"in this "
            "context, do we still make plan?\"* — deltas are always against "
            "the frozen plan of record. The strip below them answers *\"what "
            "does this context change vs the standing base outlook (no "
            "scenario, no actions)?\"*, and the bridge splits that move into "
            "what the world does to us and what our response recovers.\n\n"
            "Numbers run on two clocks, grouped and labeled: **this quarter** "
            "(the shipment commitment) and the **fiscal year** (plan, "
            "targets, and the window action EVs are measured over). The tile "
            "cards share one column grid — a blank slot means that measure "
            "isn't meaningful on that clock (the margin target is annual, so "
            "the quarter card carries no P(margin) tile). Full guide on the "
            "last tab."))
    if ctx_result is not base_result:
        cost_note = (f" The package's {fmt_money(package_cost)} decision cost "
                     f"is charged to Q1 operating income." if package_cost
                     else "")
        st.info(md(f"Conditioned on **{context_label}** — every figure on "
                   f"this page answers *\"if this happens and we act, where "
                   f"do we land?\"* Tiles compare against the frozen plan of "
                   f"record; the strip below prices this context against the "
                   f"standing base-case outlook.{cost_note}"))
    # tiles grouped by clock in bordered cards on one shared 5-column grid.
    # Columns are concepts — revenue/level, its probability, margin, margin
    # probability, downside — and a slot stays BLANK when the measure isn't
    # meaningful on that clock (e.g. the margin target is annual, so the
    # quarter card has no P(margin) tile by design).
    with st.container(border=True):
        st.markdown("##### This quarter — do we ship it?")
        q = st.columns(5)
        q[0].metric("Expected Q1 revenue", fmt_money(k["q1_revenue"]["mean"]),
                    f"{fmt_money(k['q1_revenue']['mean'] - k['q1_plan'])} vs plan")
        q[1].metric("P(Q1 revenue plan)", fmt_pct(k["p_q1_plan"], 0))
        q[2].metric("Expected Q1 gross margin", fmt_pct(k["q1_gm"]["mean"]))
        # q[3] blank: the margin target is annual — no quarterly commitment
        q[4].metric("Q1 revenue at risk (P5 vs plan)",
                    fmt_money(k["q1_revenue_at_risk"]))
    with st.container(border=True):
        st.markdown("##### Full year — do we make the plan?")
        y = st.columns(5)
        y[0].metric("Expected FY revenue", fmt_money(k["fy_revenue"]["mean"]),
                    f"{fmt_money(k['fy_revenue']['mean'] - k['fy_plan'])} vs plan")
        y[1].metric("P(FY revenue plan)", fmt_pct(k["p_fy_plan"], 0))
        y[2].metric("Expected FY gross margin", fmt_pct(k["fy_gm"]["mean"]),
                    f"{100 * (k['fy_gm']['mean'] - k['gm_target']):+.1f} pts "
                    f"vs target")
        y[3].metric("P(FY margin target)", fmt_pct(k["p_gm_target"], 0))
        y[4].metric("FY revenue at risk (P5 vs plan)",
                    fmt_money(k["fy_revenue_at_risk"]))
    with st.container(border=True):
        st.markdown("##### Year-end position — stock & service")
        p = st.columns(5)
        p[0].metric("Year-end inventory", fmt_money(k["ending_inventory"]["mean"]),
                    f"{fmt_money(k['ending_inventory']['mean'] - k['inventory_target'])} "
                    f"vs target", delta_color="inverse")
        p[1].metric("P(inventory over target)",
                    fmt_pct(k["p_inventory_over_target"], 0))
        p[2].metric("FY service level (fill rate)",
                    fmt_pct(k["service_level"]["mean"]))
        # p[3], p[4] blank: working capital is deliberately not a headline —
        # its level is dominated by fixed DSO/DPO assumptions, not simulation
        # insight; its deltas remain in the action and scenario tables

    if ctx_cmp:
        with st.container(border=True):
            st.markdown(f"##### What {context_label} changes vs the base outlook")
            r1 = st.columns(5)
            r1[0].metric("Δ Q1 revenue", fmt_money(ctx_cmp["d_q1_revenue"]))
            r1[1].metric("Δ P(Q1 plan)", fmt_pts(ctx_cmp["d_p_q1_plan"]))
            r1[2].metric("Δ FY revenue", fmt_money(ctx_cmp["d_fy_revenue"]))
            r1[3].metric("Δ P(FY plan)", fmt_pts(ctx_cmp["d_p_fy_plan"]))
            r1[4].metric("Δ FY gross margin", fmt_pts(ctx_cmp["d_fy_gm"]))
            r2 = st.columns(5)
            r2[0].metric("Δ year-end inventory", fmt_money(ctx_cmp["d_inventory"]),
                         delta_color="off")
            r2[1].metric("FY revenue-at-risk reduced",
                         fmt_money(ctx_cmp["d_revenue_at_risk"]))
            r2[2].metric("Δ FY service level", fmt_pts(ctx_cmp["d_service"]))
            if package_specs:
                r2[3].metric("FY incremental EV (net of cost)",
                             fmt_money(ctx_cmp["incremental_ev"]),
                             f"cost {fmt_money(package_cost)}",
                             delta_color="off")
            # r2[3] blank without a package; r2[4] always blank
        pkg_stage = package_label if package_specs else None
        b1, b2 = st.columns(2)
        b1.plotly_chart(viz.context_bridge(base_kpi, world_kpi, ctx_kpi,
                                           spec.name, pkg_stage, "revenue"),
                        width='stretch', key="bridge_revenue")
        b2.plotly_chart(viz.context_bridge(base_kpi, world_kpi, ctx_kpi,
                                           spec.name, pkg_stage, "margin"),
                        width='stretch', key="bridge_margin")
        if package_specs:
            st.caption("Read the bridges together: a response can recover "
                       "revenue while paying margin for it (expedite premiums "
                       "land in COGS) — recovering both is a different "
                       "decision than trading one for the other.")
        if len(package_specs) >= 2 and spec.name == "Base Case" \
                and all(n in action_results for n in package_names):
            additive = sum(
                action_results[n][0]["fy_gross_profit"]["mean"]
                - base_kpi["fy_gross_profit"]["mean"]
                - ACTIONS_CATALOG[n].action_cost_usd for n in package_names)
            st.caption(md(
                f"Interaction check: the package's incremental EV is "
                f"{fmt_money(ctx_cmp['incremental_ev'])}, vs "
                f"{fmt_money(additive)} if the actions were simply additive. "
                f"The gap is overlap — actions relieving the same constraint "
                f"don't stack. (Individual EVs approximated at up to 2,000 "
                f"paths.)"))

    st.markdown("### Executive summary")
    st.markdown(md(overview_summary))

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"#### Top risks{ctx_suffix}")
        for r in ctx_risks[:3]:
            st.warning(md(f"{r['risk']}  \n*Trigger: {r['threshold']}*"))
    with col_r:
        st.markdown("#### Top recommended actions")
        if ctx_result is not base_result:
            st.caption("Valued against the standing base-case outlook, "
                       "excluding actions already in your package — open "
                       "Management Recommendations to price actions under "
                       "the selected scenario.")
        top_recs = [r for r in recs if r.title not in package_names]
        if top_recs:
            for r in top_recs[:3]:
                st.info(md(
                    f"**{r.title}** — EV {fmt_money(r.expected_value_usd)}, "
                    f"P(plan) {fmt_pts(r.prob_plan_improvement)}, cost "
                    f"{fmt_money(r.incremental_cost_usd)} ({r.confidence} confidence)"))
        else:
            st.info("No modeled action clears the expected-value bar this cycle.")

    from src.simulation import quarterly
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(viz.distribution_with_target(
            quarterly(ctx_result.revenue)[:, 0], k["q1_plan"],
            f"Q1 revenue distribution vs plan{ctx_suffix}", "Q1 revenue ($)"),
            width='stretch')
    with g2:
        fy_rev = ctx_result.revenue[:, :12].sum(axis=1)
        fy_gm = np.where(fy_rev > 0,
                         ctx_result.gross_profit[:, :12].sum(axis=1) / fy_rev, 0)
        st.plotly_chart(viz.distribution_with_target(
            fy_gm, CONFIG.financial.gross_margin_target,
            f"FY gross-margin distribution vs target{ctx_suffix}", "Gross margin",
            value_fmt="pct", target_label="Target"), width='stretch')
    st.plotly_chart(viz.quarterly_fan_chart(ctx_result, baseline.revenue_plan_q,
                                            ctx_suffix),
                    width='stretch')

# ---------------------------- 2. Demand & Backlog --------------------------
with tabs[1]:
    st.markdown(f"#### Projected delinquency — outcome view{ctx_suffix}")
    pd_base_curve = expected_past_due_curve(base_result)
    pd_ctx_curve = (pd_base_curve if ctx_result is base_result else
                    expected_past_due_curve(ctx_result))
    months = month_labels()
    dcols = st.columns(4)
    for col, label, m_idx in [
            (dcols[0], "Expected past-due at Q1 end", 2),
            (dcols[1], "Expected past-due at FY end", 11),
            (dcols[2], "Expected past-due at horizon end", 17)]:
        col.metric(label, f"{pd_ctx_curve[m_idx]:.0f} systems",
                   None if ctx_result is base_result else
                   f"{pd_ctx_curve[m_idx] - pd_base_curve[m_idx]:+.0f} vs base",
                   delta_color="inverse")
    peak_m = int(pd_ctx_curve.argmax())
    dcols[3].metric("Peak expected past-due", f"{pd_ctx_curve.max():.0f} systems",
                    f"in {months[peak_m]}", delta_color="off")
    t1, t2 = st.columns(2)
    t1.plotly_chart(viz.backlog_trajectory_chart(base_result, ctx_result,
                                                 context_label),
                    width='stretch', key="delinquency_trajectory")
    t2.plotly_chart(viz.past_due_family_chart(
        expected_past_due_by_family(ctx_result), ctx_suffix),
        width='stretch', key="delinquency_family")
    st.caption("Past-due backlog = cumulative demand minus cumulative "
               "shipments across simulated futures, floored at zero "
               "(expected value; band = P25-P75 of the conditioned outlook). "
               "Left: how big and when. Right: what the queue is made of — "
               "computed per family, so the stack can sit slightly above the "
               "total curve (one family's early shipments cannot serve "
               "another family's demand). Set a scenario and response package "
               "in the sidebar to see projected delinquency for the world "
               "plus what we will do.")

    st.divider()
    st.caption("Everything below is plan-of-record over the full 18-month "
               "planning horizon (monthly buckets from 2026-07) — it never "
               "moves with the sidebar.")
    st.markdown("#### Demand plan (customer × family × month)")
    dem = data.demand.copy()
    dem["units"] = dem["base_forecast_units"] + dem["backlog_units"]
    pivot = dem.pivot_table(index=["customer", "product_family"], columns="month",
                            values="units", aggfunc="sum").fillna(0).astype(int)
    st.dataframe(pivot, width='stretch', height=420)
    st.plotly_chart(viz.backlog_aging_chart(baseline), width='stretch')

# --------------- 3. Market Intelligence & Demand Confidence ----------------
with tabs[2]:
    # -- Executive summary -----------------------------------------------
    level_icon = {"Very High": "🟢", "High": "🟢", "Moderate": "🟡",
                  "Low": "🟠", "Very Low": "🔴"}[dc.level]
    h1, h2, h3 = st.columns([2, 1, 1])
    h1.metric("Demand Confidence", f"{level_icon} {dc.level}",
              f"{dc.score:.0f}/100 composite score", delta_color="off")
    h2.metric("Forecast Confidence",
              f"{dc.forecast_confidence_pct:.0f}%",
              dc.forecast_confidence_label, delta_color="off")
    h3.metric("Simulation variance setting",
              f"×{conf_params['demand_sigma_mult']:.2f}",
              "applied to demand volatility", delta_color="off")
    st.markdown(md(dc.narrative))
    if confidence_overridden:
        st.warning(f"Sidebar override active: simulations are running at "
                   f"**{effective_level}**, not the assessed "
                   f"**{dc.level}** shown above. Set the sidebar back to "
                   f"'Assessed' to apply this page's assessment.")
    st.caption("This page answers: how much confidence should management have "
               "in the current demand plan? The assessment feeds the Monte "
               "Carlo simulation directly — see the mapping below.")

    # -- Market signals ----------------------------------------------------
    st.markdown("#### Market signals")
    sig_cols = st.columns(4)
    lower_is_better = {"Customer Push-Out Rate", "Cancellation Rate",
                       "Backlog Aging"}
    for i, s in enumerate(dc.signals):
        unit_txt = {"%": "%", "x": "×", "count": "", "pts": " pts"}[s.unit]
        sig_cols[i % 4].metric(
            s.name, f"{s.current:g}{unit_txt}",
            f"{s.trend:+g}{unit_txt} vs 3 mo ago",
            delta_color="inverse" if s.name in lower_is_better else "normal",
            help=s.comment)
    with st.expander("Signal detail — trailing 12 months"):
        pick_sig = st.selectbox("Signal", [s.name for s in dc.signals])
        s = next(x for x in dc.signals if x.name == pick_sig)
        st.plotly_chart(viz.signal_history_chart(s.name, s.history, s.unit),
                        width='stretch')
        st.dataframe(pd.DataFrame([{
            "Signal": x.name, "Current": x.current, "Unit": x.unit,
            "Trend vs 3 mo": x.trend, "Score (0-100)": round(x.score),
            "Weight": x.weight} for x in dc.signals]),
            width='stretch', hide_index=True)

    # -- External market intelligence -------------------------------------
    st.markdown("#### External market intelligence")
    if dc.external_source == "curated":
        as_of = next((e.as_of for e in dc.external if e.as_of), "")
        st.success(f"Curated intelligence feed loaded from "
                   f"`data/external_intel.csv`{f' (as of {as_of})' if as_of else ''}. "
                   f"Net proposed impact applied to the confidence score: "
                   f"{dc.external_adjustment_pts:+.1f} pts (each topic capped at "
                   f"±3, total at ±12).")
    else:
        st.caption("Showing synthetic external intelligence (no curated feed "
                   "found at `data/external_intel.csv`). A monthly Claude "
                   "routine or analyst can supply one — topic, stance, "
                   "two-sentence summary, sources, proposed score impact.")
    stance_icon = {"Favorable": "🟢 Favorable", "Neutral": "⚪ Neutral",
                   "Watch": "🟠 Watch", "Unfavorable": "🔴 Unfavorable"}
    ext_cols = st.columns(2)
    for i, e in enumerate(dc.external):
        with ext_cols[i % 2]:
            impact_txt = (f" · proposed impact {e.proposed_impact_pts:+.1f} pts"
                          if e.proposed_impact_pts is not None else "")
            st.markdown(f"**{e.topic}** — {stance_icon[e.stance]}{impact_txt}")
            st.caption(e.summary)
            if e.sources:
                st.caption("Sources: " + " · ".join(e.sources))
    st.caption("Intelligence impacts adjust the confidence *score* (0–100), "
               "not the simulation directly. They reach the Monte Carlo only "
               "by tipping the score across a level threshold — at which "
               "point all three simulation knobs change together (see the "
               "mapping below). No single article moves an individual knob.")
    with st.expander("Update the intelligence feed (upload a new CSV)"):
        up = st.file_uploader("external_intel.csv", type="csv",
                              label_visibility="collapsed")
        if up is not None:
            new_text = up.getvalue().decode("utf-8")
            try:
                parse_curated_external(new_text)   # validate before accepting
                EXTERNAL_INTEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                EXTERNAL_INTEL_PATH.write_text(new_text, encoding="utf-8")
                cached_confidence.clear()
                st.success("Feed updated — rerun to apply (press R or refresh).")
            except (ValueError, KeyError) as exc:
                st.error(f"File rejected: {exc}")

    # -- Customer demand confidence ----------------------------------------
    st.markdown("#### Customer demand confidence")
    rating_icon = {"High": "🟢 High", "Moderate": "🟡 Moderate", "Low": "🔴 Low"}
    cust_cols = st.columns(3)
    for i, c in enumerate(dc.customers[:6]):
        with cust_cols[i % 3]:
            st.markdown(f"**{c.customer}**  \n{c.group} · "
                        f"{md(fmt_money(c.horizon_revenue))} horizon demand")
            st.markdown(f"Demand Confidence: **{rating_icon[c.rating]}**")
            st.caption(" · ".join(c.reasons))
            st.divider()

    # -- Forecast confidence ------------------------------------------------
    st.markdown("#### Forecast confidence")
    fc1, fc2 = st.columns([1, 3])
    fc1.metric("Forecast Confidence", f"{dc.forecast_confidence_pct:.0f}%",
               dc.forecast_confidence_label, delta_color="off")
    fc2.markdown(dc.forecast_confidence_reason)

    # -- Monte Carlo assumptions --------------------------------------------
    st.markdown("#### Assumptions feeding the Monte Carlo simulation")
    st.dataframe(assumptions_in_business_language(data, CONFIG, dc),
                 width='stretch', hide_index=True)
    st.markdown("**How Demand Confidence adjusts the simulation**")
    st.markdown(
        "Confidence never rewrites the demand plan. It changes how tightly "
        "the simulated world hugs the plan (demand volatility) and how "
        "orders behave on the way (push-out and cancellation probabilities) "
        "— three knobs, read straight from the table the simulation uses:")
    st.dataframe(confidence_mapping_table(effective_level),
                 width='stretch', hide_index=True)
    if confidence_overridden:
        st.caption(f"→ marks the sidebar override ({effective_level}), "
                   f"currently applied to the base case, every scenario, and "
                   f"every management action. The assessed level is "
                   f"{dc.level}.")
    else:
        st.caption(f"→ marks the current assessment ({effective_level}), "
                   "applied to the base case, every scenario, and every "
                   "management action — lower confidence widens every "
                   "probability distribution in this app. Volatility is pure "
                   "spread (it widens or narrows outcomes around the same "
                   "plan); push-outs move revenue later; cancellations "
                   "remove it.")

    # -- Demand sensitivity ---------------------------------------------------
    st.markdown("#### Demand sensitivity — what confidence is worth")
    sens_rows = []
    for lvl in CONFIDENCE_LEVELS:
        lp = dict(CONFIDENCE_SIM_PARAMS[lvl])
        r = cached_simulation(data_seed, sim_seed, n_sims,
                              f"Confidence: {lvl}",
                              _params_key(lp), data, baseline, lp)
        fy_rev = r.revenue[:, :12].sum(axis=1)
        fy_gp = r.gross_profit[:, :12].sum(axis=1)
        sens_rows.append({
            "level": lvl,
            "mean": fy_rev.mean(), "p5": np.percentile(fy_rev, 5),
            "p95": np.percentile(fy_rev, 95),
            "P(FY plan)": (fy_rev >= baseline.revenue_plan_q[:4].sum()).mean(),
            "GM": (fy_gp / fy_rev).mean(),
            "EBITDA": r.ebitda[:, :12].sum(axis=1).mean(),
            "Inventory": r.inventory[:, 11].mean(),
            "Working capital": r.working_capital[:, 11].mean(),
            "Cash flow": r.cash_flow[:, :12].sum(axis=1).mean(),
        })
    sens_df = pd.DataFrame(sens_rows)
    st.plotly_chart(viz.confidence_sensitivity_chart(sens_df, effective_level),
                    width='stretch')
    disp = sens_df.assign(**{
        "FY revenue (mean)": sens_df["mean"], "FY revenue (P5)": sens_df["p5"],
    })[["level", "FY revenue (mean)", "FY revenue (P5)", "P(FY plan)", "GM",
        "EBITDA", "Inventory", "Working capital", "Cash flow"]]
    disp.columns = ["Confidence level", "FY revenue (mean)", "FY revenue (P5)",
                    "P(FY plan)", "Gross margin", "EBITDA", "Ending inventory",
                    "Working capital", "Cash flow"]
    st.dataframe(disp.style.format({
        "FY revenue (mean)": lambda v: fmt_money(v),
        "FY revenue (P5)": lambda v: fmt_money(v),
        "P(FY plan)": "{:.0%}", "Gross margin": "{:.1%}",
        "EBITDA": lambda v: fmt_money(v),
        "Ending inventory": lambda v: fmt_money(v),
        "Working capital": lambda v: fmt_money(v),
        "Cash flow": lambda v: fmt_money(v)}),
        width='stretch', hide_index=True)
    st.caption(f"Each row re-runs the simulation with only the Demand "
               f"Confidence adjustment changed, at the same path count and "
               f"seeds as the headline results ({n_sims:,} paths, common "
               f"random numbers) — so the applied level's row reconciles "
               f"exactly with the standing base outlook on the Executive "
               f"Overview. The Moderate row is the unadjusted world (base "
               f"rates from history, no intel adjustment). Expected values "
               f"move modestly, but the downside tail (P5) and "
               f"plan-attainment probability move sharply — demand "
               f"confidence is primarily a tail-risk driver, which is why "
               f"it belongs in front of executives.")

# ---------------------------- 3. Supply & Components -----------------------
with tabs[3]:
    if ctx_result is not base_result:
        st.caption(f"Showing supply risk under **{context_label}**. Select "
                   "Base Case (and clear the package) for the standing risk "
                   "picture.")
    st.plotly_chart(viz.component_risk_heatmap(data.components, ctx_binding),
                    width='stretch')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### Most frequently binding components{ctx_suffix}")
        merged_bind = ctx_binding.rename(
            columns={"binding_frequency": "binding (conditioned)"})
        if ctx_result is not base_result:
            merged_bind = merged_bind.merge(
                binding.rename(columns={"binding_frequency": "binding (base)"}),
                on="component", how="left")
        st.dataframe(merged_bind.style.format(
            {c: "{:.1%}" for c in merged_bind.columns if c != "component"}),
            width='stretch')
        st.caption("Binding frequency = share of simulations in which the "
                   "component constrains shipments at least once over the "
                   "18-month horizon.")
    with c2:
        st.markdown("#### Site disruption frequency (simulated)")
        st.dataframe(site_disruption_frequency(ctx_result)
                     .style.format({"disruption_frequency": "{:.1%}"}),
                     width='stretch')
    st.divider()
    st.caption("Below: input reference — the component master never moves "
               "with the sidebar.")
    st.markdown("#### Component master")
    st.dataframe(data.components, width='stretch', height=380)

# ---------------------------- 4. EMS & Integration -------------------------
with tabs[4]:
    if ctx_result is not base_result:
        st.caption(f"Showing capacity utilization under **{context_label}**. "
                   "Select Base Case (and clear the package) for the standing "
                   "picture.")
    st.plotly_chart(viz.utilization_heatmap(ctx_result, ctx_suffix),
                    width='stretch')
    st.divider()
    st.caption("Everything below is the plan-of-record baseline — "
               "deterministic site loading and capacity vs demand; it never "
               "moves with the sidebar.")
    st.plotly_chart(viz.site_utilization_heatmap(baseline), width='stretch')
    st.plotly_chart(viz.ems_capacity_vs_demand(baseline), width='stretch')
    st.markdown("#### Baseline constraint log")
    st.dataframe(baseline.constraints, width='stretch', height=260)

# ---------------------------- 5. Financial Outcomes ------------------------
with tabs[5]:
    from src.simulation import fiscal_year
    if ctx_result is not base_result:
        st.caption(f"Showing financial outcomes under **{context_label}**. "
                   "Select Base Case (and clear the package) for the standing "
                   "picture.")
    fy_map = {
        "FY revenue": ctx_result.revenue[:, :12].sum(axis=1),
        "FY gross profit": ctx_result.gross_profit[:, :12].sum(axis=1),
        "FY operating income": ctx_result.operating_income[:, :12].sum(axis=1),
        "FY EBITDA proxy": ctx_result.ebitda[:, :12].sum(axis=1),
        "FY cash-flow proxy": ctx_result.cash_flow[:, :12].sum(axis=1),
    }
    pick = st.selectbox("Distribution", list(fy_map))
    ref = baseline.revenue_plan_q[:4].sum() if pick == "FY revenue" else \
        float(np.median(fy_map[pick]))
    st.plotly_chart(viz.distribution_with_target(
        fy_map[pick], ref, f"{pick} distribution{ctx_suffix}", f"{pick} ($)",
        target_label="Plan" if pick == "FY revenue" else "Median"),
        width='stretch')
    st.plotly_chart(viz.inventory_trajectory(
        ctx_result, CONFIG.financial.inventory_target_usd, ctx_suffix),
        width='stretch')
    st.markdown(f"#### FY distribution statistics{ctx_suffix}")
    stats_rows = []
    for name, arr in fy_map.items():
        stats_rows.append({
            "Metric": name, "Mean": arr.mean(), "Median": np.median(arr),
            "P5": np.percentile(arr, 5), "P95": np.percentile(arr, 95)})
    st.dataframe(pd.DataFrame(stats_rows).set_index("Metric")
                 .style.format(lambda v: fmt_money(v)), width='stretch')
    st.divider()
    st.caption("Below: the plan-of-record monthly baseline P&L — never moves "
               "with the sidebar.")
    st.markdown("#### Monthly baseline P&L")
    st.dataframe(baseline.monthly.round(3), width='stretch', height=320)

# ---------------------------- 6. Risk Drivers ------------------------------
with tabs[6]:
    if ctx_result is not base_result:
        st.caption(f"Showing risk drivers under **{context_label}**. Select "
                   "Base Case (and clear the package) for the standing risk "
                   "picture.")
    outcome = st.selectbox("Outcome to explain", list(ctx_rankings))
    st.plotly_chart(viz.tornado_chart(ctx_rankings[outcome],
                                      outcome + ctx_suffix),
                    width='stretch')
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.family_risk_chart(ctx_fam_risk), width='stretch')
    with c2:
        st.plotly_chart(viz.risk_matrix(ctx_fam_risk, ctx_kpi), width='stretch')
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"#### Months with greatest capacity risk{ctx_suffix}")
        st.dataframe(ctx_cap_risk.head(6).style.format(
            {"p_capacity_shortfall": "{:.0%}", "expected_units_short": "{:.1f}"}),
            width='stretch')
    with c4:
        st.markdown("#### Drivers of revenue shifting Q1 → Q2")
        st.dataframe(quarter_shift_drivers(ctx_result).head(6)
                     .style.format({"spearman_rho": "{:+.2f}"}),
                     width='stretch')
    st.caption("Method: Spearman rank correlation between sampled inputs and "
               "outcomes. Rankings show association, not causation — drivers share "
               "common factors by design.")

# ---------------------------- 7. Scenario Comparison -----------------------
with tabs[7]:
    chosen = st.multiselect(
        "Scenarios to compare against the base case",
        [s for s in scenarios if s != "Base Case"],
        default=["AI Demand Surge", "Critical FPGA Shortage",
                 "Major Customer Push-Out", "EMS Malaysia Disruption"])
    rows = []
    scen_curves: dict = {}
    for name in chosen:
        s = scenarios[name]
        r = sim_with_confidence(n_sims, name, s.overrides)
        rows.append(compare_scenarios(base_kpi, kpi_summary(r, baseline, CONFIG),
                                      s.action_cost_usd))
        scen_curves[name] = expected_past_due_curve(r)
    if rows:
        st.plotly_chart(viz.scenario_comparison_chart(rows), width='stretch')
        col_names = {
            "d_q1_revenue": "Δ Q1 revenue", "d_fy_revenue": "Δ FY revenue",
            "d_p_q1_plan": "Δ P(Q1 plan)", "d_p_fy_plan": "Δ P(FY plan)",
            "d_fy_gm": "Δ FY gross margin",
            "d_gross_profit": "Δ FY gross profit",
            "d_inventory": "Δ year-end inventory",
            "d_working_capital": "Δ year-end working capital",
            "d_expedite": "Δ FY expedite cost", "d_service": "Δ FY service",
            "d_revenue_at_risk": "FY revenue-at-risk reduced",
            "action_cost": "Decision cost",
            "incremental_ev": "FY incremental EV",
            "risk_reduced_per_dollar": "Risk reduced per dollar"}
        money = {"Δ Q1 revenue", "Δ FY revenue", "Δ FY gross profit",
                 "Δ year-end inventory", "Δ year-end working capital",
                 "Δ FY expedite cost", "FY revenue-at-risk reduced",
                 "Decision cost", "FY incremental EV"}
        pts = {"Δ P(Q1 plan)", "Δ P(FY plan)", "Δ FY gross margin",
               "Δ FY service"}
        disp = (pd.DataFrame(rows).set_index("scenario")
                .rename(columns=col_names))
        st.dataframe(disp.style.format(
            {**{c: (lambda v: fmt_money(v)) for c in money},
             **{c: (lambda v: fmt_pts(v)) for c in pts},
             "Risk reduced per dollar": "{:.1f}"}), width='stretch')
    if world_result is not base_result:
        st.plotly_chart(viz.revenue_bridge(base_kpi, world_kpi),
                        width='stretch')
    if scen_curves:
        st.plotly_chart(viz.backlog_comparison_chart(pd_base_curve,
                                                     scen_curves),
                        width='stretch', key="scen_cmp_trajectory")
        st.caption("Expected past-due backlog = cumulative demand minus "
                   "cumulative shipments, floored at zero, averaged across "
                   "simulated futures — one line per selected scenario, "
                   "unmitigated (no response package). The conditioned view "
                   "for your selected world + package lives on Demand & "
                   "Backlog.")

# ---------------------------- 8. Management Actions ------------------------
with tabs[8]:
    st.markdown("#### Modeled management actions")
    # Actions are priced in the WORLD of the sidebar context, automatically —
    # no page-local world selector (that would be a second source of truth).
    # Under a scenario, the base-world EV stays visible as a comparison
    # column: both frames at once, never toggled.
    conditioned = spec.name != "Base Case"
    if conditioned:
        with st.spinner(f"Pricing every action in the {spec.name} world..."):
            page_actions = cached_actions(data_seed, sim_seed, min(n_sims, 2000),
                                          dc.level, _params_key(spec.overrides),
                                          catalog_key, data, baseline,
                                          spec.overrides, ACTIONS_CATALOG)
        ref_kpi = world_kpi
        st.info(f"Every number below answers *\"if {spec.name} occurs, what "
                f"does this action buy us?\"* — each action is simulated on "
                f"top of the scenario and compared against the scenario "
                f"without it, on the same futures. The base-world EV column "
                f"shows how the price shifts: an action only pays under "
                f"stress if it relieves the constraint the scenario actually "
                f"creates (e.g., expediting recovers late shipments but "
                f"cannot recover a cut supply allocation, and its premiums "
                f"rise in a shortage).")
    else:
        page_actions = action_results
        ref_kpi = base_kpi
        st.caption("Actions are priced in the base world — the standing "
                   "outlook. Select a scenario in the sidebar and this table "
                   "reprices automatically in that world (with the base-world "
                   "EV kept alongside). Use the checkboxes below to build the "
                   "response package the rest of the app evaluates.")

    act_rows = []
    for name, (kpi, aspec) in page_actions.items():
        cmpv = compare_scenarios(ref_kpi, kpi, aspec.action_cost_usd)
        row = {
            "Action": name,
            "In package": name in package_names,
            "Horizon": aspec.horizon,
            "Δ FY revenue ($M)": cmpv["d_fy_revenue"] / 1e6,
            "Δ gross profit ($M)": cmpv["d_gross_profit"] / 1e6,
            "Δ P(Q1 plan) (pts)": cmpv["d_p_q1_plan"] * 100,
            "Δ P(FY plan) (pts)": cmpv["d_p_fy_plan"] * 100,
            "Δ FY GM (pts)": cmpv["d_fy_gm"] * 100,
            "Δ inventory ($M)": cmpv["d_inventory"] / 1e6,
            "Δ working capital ($M)": cmpv["d_working_capital"] / 1e6,
            "Δ service (pts)": cmpv["d_service"] * 100,
            "Cost ($M)": aspec.action_cost_usd / 1e6,
            "Incremental EV ($M)": cmpv["incremental_ev"] / 1e6}
        if conditioned and name in action_results:
            base_cmp = compare_scenarios(base_kpi, action_results[name][0],
                                         aspec.action_cost_usd)
            row["EV in base world ($M)"] = base_cmp["incremental_ev"] / 1e6
        act_rows.append(row)
    adf = (pd.DataFrame(act_rows)
           .sort_values("Incremental EV ($M)", ascending=False)
           .set_index("Action"))

    def _sync_package(editor_key: str) -> None:
        """Apply checkbox edits to the shared package (runs before rerun,
        so the sidebar multiselect picks the change up on the same pass)."""
        edits = st.session_state[editor_key].get("edited_rows", {})
        rows = st.session_state["package_editor_rows"]
        chosen = set(st.session_state.get("package_names", []))
        for pos, change in edits.items():
            if "In package" in change:
                name = rows[int(pos)]
                (chosen.add if change["In package"] else chosen.discard)(name)
        st.session_state["package_names"] = [n for n in ACTIONS_CATALOG
                                             if n in chosen]

    st.session_state["package_editor_rows"] = list(adf.index)
    # a package-derived key gives the editor a clean slate whenever the
    # package changes (from either the checkboxes or the sidebar), so its
    # checkbox states always match the shared package exactly
    editor_key = "package_editor|" + "|".join(sorted(package_names))
    st.data_editor(
        adf, key=editor_key, on_change=_sync_package, args=(editor_key,),
        disabled=[c for c in adf.columns if c != "In package"],
        column_config={
            "In package": st.column_config.CheckboxColumn(
                "In package",
                help="Check to add this action to the response package "
                     "(synced with the sidebar selector)."),
            **{c: st.column_config.NumberColumn(format="%+.1f")
               for c in adf.columns if "($M)" in c or "(pts)" in c},
            "Cost ($M)": st.column_config.NumberColumn(format="%.1f"),
        },
        width='stretch')
    ref_label = f"the {spec.name} scenario" if conditioned else "the base case"
    st.caption(f"Every delta is action-versus-{ref_label} on identical simulated "
               "futures (common random numbers), so differences are the action's "
               "effect alone. All deltas and EV are measured over the "
               "fiscal-year window (first 12 months); each action carries a "
               "SIOP horizon (Execution 0-3 mo / Tactical 1-3 qtrs / Long-lead "
               "6-18 mo) and a realistic effective-month latency, so long-lead "
               "actions show little or no Q1 effect — and part of their benefit "
               "falls beyond the EV window — by construction. Scoring weights: "
               "EV 35%, probability 30%, revenue 20%, cash use −15%. "
               "Materiality gates: EV > −\\$1M and (ΔP ≥ 1pt or EV ≥ \\$2M).")
    st.markdown("#### Ranked recommendations"
                + (f" — if {spec.name} occurs" if conditioned else ""))
    if conditioned:
        world_binding = (binding if world_result is base_result else
                         binding_components(world_result))
        page_recs = build_recommendations(ref_kpi, page_actions, world_binding,
                                          demand_confidence=dc)
        st.caption("Ranked for the conditioned world. The Executive Overview's "
                   "top actions remain anchored to the base case.")
    else:
        page_recs = recs
    for i, r in enumerate(page_recs, 1):
        with st.expander(f"{i}. {r.title} — EV {fmt_money(r.expected_value_usd)} "
                         f"({r.confidence} confidence)"):
            st.markdown(md(
                f"**Why:** {r.risk}.  \n"
                f"**Trigger:** {r.threshold}.  \n"
                f"**Action:** {r.action}  \n\n"
                f"**Modeled impact (fiscal-year window):** "
                f"revenue {fmt_money(r.revenue_protected_usd)}, "
                f"gross profit {fmt_money(r.gross_profit_protected_usd)}, "
                f"P(plan) {fmt_pts(r.prob_plan_improvement)}, inventory "
                f"{fmt_money(r.inventory_change_usd)}, working capital "
                f"{fmt_money(r.working_capital_change_usd)}, service "
                f"{fmt_pts(r.service_level_change)}, incremental cost "
                f"{fmt_money(r.incremental_cost_usd)}.  \n"
                f"**Authored assumptions (the claim being made):** "
                f"{describe_overrides(page_actions[r.title][1].overrides)} — "
                f"decision cost {fmt_money(page_actions[r.title][1].action_cost_usd)}. "
                f"Edit in config/management_actions.yaml.  \n"
                f"**Caveat:** {r.caveat}"))

# ---------------------------- 9. Assumptions & Data ------------------------
with tabs[9]:
    st.markdown("#### Financial plan & targets")
    fin = CONFIG.financial
    st.dataframe(pd.DataFrame({
        "Assumption": ["Gross-margin target", "Inventory target", "Inventory-turns target",
                       "Monthly opex", "Monthly depreciation", "Monthly capex",
                       "Tax rate", "DSO days", "DPO days", "E&O reserve rate"],
        "Value": [fmt_pct(fin.gross_margin_target), fmt_money(fin.inventory_target_usd),
                  f"{fin.inventory_turns_target:.1f}", fmt_money(fin.opex_monthly_usd),
                  fmt_money(fin.depreciation_monthly_usd), fmt_money(fin.capex_monthly_usd),
                  fmt_pct(fin.tax_rate), f"{fin.dso_days:.0f}", f"{fin.dpo_days:.0f}",
                  fmt_pct(fin.eo_reserve_rate)]}), width='stretch',
        hide_index=True)
    st.caption("Edit config/default_config.yaml to change targets; use the sidebar "
               "custom scenario for what-if overrides without editing files.")

    st.markdown("#### Management action claim sheets")
    st.caption("The authored assumptions behind every management action: the "
               "capability claimed, its cost, and when it takes effect. These "
               "are analyst-owned inputs (edit config/management_actions.yaml — "
               "validated on load); everything on the Management "
               "Recommendations page is computed from them. The overtime "
               "conversion premium comes from overtime_premium_pct in the EMS "
               "site table below (40-48% by site).")
    claim_df = action_assumptions_table(ACTIONS_CATALOG)
    st.dataframe(claim_df.style.format({"Decision cost": lambda v: fmt_money(v)}),
                 width='stretch', hide_index=True)

    with st.expander("What-if: try a modified action catalog (session only)"):
        st.caption("Upload an edited claim-sheet YAML to reprice every "
                   "action, package and recommendation in this session — a "
                   "demo of the analyst edit loop. Nothing is saved: the repo "
                   "file (edited via pull request) remains the record, and a "
                   "redeploy or the revert button in the sidebar restores it. "
                   "New actions are priced and can join the response package; "
                   "appearing in *ranked recommendations* additionally needs "
                   "a risk-tag mapping in src/recommendations.py.")
        catalog_path = Path(__file__).parent / "config" / "management_actions.yaml"
        st.download_button("Download the current catalog as a starting point",
                           catalog_path.read_text(encoding="utf-8"),
                           file_name="management_actions.yaml",
                           mime="text/yaml")
        up_nonce = st.session_state.get("catalog_upload_nonce", 0)
        up_yaml = st.file_uploader(
            "management_actions.yaml", type=["yaml", "yml"],
            label_visibility="collapsed", key=f"catalog_upload_{up_nonce}")
        if up_yaml is not None:
            new_catalog_text = up_yaml.getvalue().decode("utf-8")
            try:
                parsed = management_actions(catalog_text=new_catalog_text)
            except (ValueError, KeyError, TypeError) as exc:
                st.error(md(f"Catalog rejected: {exc}"))
            else:
                if st.session_state.get("catalog_override_text") != new_catalog_text:
                    st.session_state["catalog_override_text"] = new_catalog_text
                    st.rerun()
                st.success(f"Session catalog active — {len(parsed)} actions "
                           "validated and priced. Revert from the sidebar.")

    st.markdown("#### Factor model (correlations)")
    engine = FactorEngine(CONFIG.factors)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Factor loadings")
        st.dataframe(engine.loadings_table().style.format("{:.2f}"),
                     width='stretch')
    with c2:
        st.markdown("Implied correlation matrix (PSD by construction)")
        st.dataframe(engine.implied_correlation().style.format("{:.2f}"),
                     width='stretch')

    if warnings:
        st.markdown("#### Validation warnings")
        for i in warnings:
            st.warning(f"[{i.area}] {i.message}")
    else:
        st.success("All input validation checks passed.")

    st.markdown("#### Products");  st.dataframe(data.products, width='stretch')
    st.markdown("#### EMS sites"); st.dataframe(data.ems_sites, width='stretch')
    st.markdown("#### Demand plan (raw)")
    st.dataframe(data.demand, width='stretch', height=300)

# ------------------------ 10. Guide, Methodology & Export ------------------
with tabs[10]:
    st.markdown("#### How this engine works — the SIOP process guide")
    st.markdown(
        """
**What SIOP is.** Sales, Inventory & Operations Planning is a monthly decision
cadence. The executive meeting exists to answer four questions, in order:
*How much should we trust the plan? What could happen to it? What should we
do about it?* — and then sign the result and record it. Every page in this
app supports one of those questions.

**The one idea behind every number.** Each simulated outcome distribution is

> **outcome = Simulate(inputs, world, response)** — judged against a frozen
> plan of record.

- The **world** is what happens *to* Apex: the base assumptions (calibrated
  by the Demand Confidence assessment on the Market Intelligence tab, or a
  sidebar override of that level for what-ifs) plus an
  optional **scenario** deviation — demand and/or supply, no cost attached.
- The **response** is what Apex chooses to *do*: a package of management
  actions from the claim-sheet catalog, each with a decision cost and a
  realistic effective-date latency.

The sidebar sets this **evaluation context** — scenario and response package
together. Bullish or bearish postures are scenarios too: they are beliefs
about the world, not decisions, and a response package should look sensible
under the base case, not only under the posture that flatters it.

**Two kinds of view.** Every chart and table is either **plan-of-record**
(a frozen anchor that never moves with the context — the plan, targets, the
deterministic baseline, the Demand & Backlog page) or an **outcome view**
(follows the evaluation context and says so in its title). If a title carries
the context suffix, it is conditioned.

**Three fixed reference frames.** (1) *vs plan* — the commitment: do we still
make the number? Tile deltas on the Executive Overview. (2) *vs the base
outlook* — what this context changes against the standing risk-adjusted
outlook with no scenario and no actions. The strip under the tiles. (3) *the
decomposition* — the bridge chart splits the total move into what the world
does to us and what our response recovers, in that world, on identical
simulated futures.

**The clocks.** The planning horizon is 18 monthly buckets from 2026-07.
Within it, figures are reported on two clocks: **this quarter** (months 1-3
— the shipment commitment, execution's question) and the **fiscal year**
(months 1-12 — plan, margin and inventory targets, and the window every
action EV is measured over; long-lead actions' benefits partly fall beyond
it by construction). Every metric label and chart title names its clock;
"year-end" means fiscal year-end. Trajectory charts run the full 18 months.

**How the tabs follow the process.**

| SIOP stage | Tabs |
| --- | --- |
| Executive answer | Executive Overview |
| Demand review | Demand & Backlog · Market Intelligence & Demand Confidence |
| Supply review | Supply & Component Risk · Manufacturing Capacity |
| Integrated reconciliation | Financial Outcomes · Risk Drivers |
| Decision support | Scenario Comparison · Management Recommendations |
| Governance & reference | Assumptions & Data · this tab (guide, methodology, export) |

**The cycle.** Market intelligence refreshes monthly (a curated feed reviewed
before merge). Within a cycle, analysts maintain the action claim sheets and
scenario set; the meeting evaluates contexts and picks a response package;
the Excel export below is the signed readout. The approved package becomes
next cycle's plan of record — that closing step is manual today and is the
engine's planned next capability.
        """)
    st.divider()
    st.markdown("#### Excel export")
    scenario_rows_all = []
    for name in ["AI Demand Surge", "Critical FPGA Shortage",
                 "Major Customer Push-Out", "EMS Malaysia Disruption",
                 "Memory Recovery Delay"]:
        s = scenarios[name]
        r = sim_with_confidence(n_sims, name, s.overrides)
        scenario_rows_all.append(
            compare_scenarios(base_kpi, kpi_summary(r, baseline, CONFIG),
                              s.action_cost_usd))
    xl_bytes = build_excel_export(
        base_kpi, baseline, base_result,
        summary_text + "\n\n" + provider.appendix(report_ctx),
        scenario_rows_all, fam_risk, binding, rankings, recs,
        data.components, data.demand)
    st.download_button("Download Excel workbook (.xlsx)", xl_bytes,
                       file_name="apex_siop_readout.xlsx",
                       mime="application/vnd.openxmlformats-officedocument"
                            ".spreadsheetml.sheet")
    st.markdown("#### CSV downloads")
    csv_map = {
        "Demand plan": data.demand, "Products": data.products,
        "Components": data.components, "EMS capacity": data.ems_capacity,
        "Monthly baseline": baseline.monthly,
        "Family revenue at risk": fam_risk, "Binding components": binding,
    }
    cols = st.columns(4)
    for i, (name, df) in enumerate(csv_map.items()):
        cols[i % 4].download_button(
            f"{name} (.csv)", df.to_csv(index=False).encode(),
            file_name=f"{name.lower().replace(' ', '_')}.csv", mime="text/csv",
            key=f"csv_{i}")

    st.markdown("#### Methodology")
    st.markdown(
        """
**Baseline plan.** Deterministic greedy allocation by month: firm backlog before
forecast, higher customer priority first, earlier requested date, then higher
contribution margin per standard-equivalent unit. Builds go only to qualified
EMS sites (least-contested site first, then cost) and respect component
availability, EMS capacity (derated by adherence and labor), and final
integration capacity. Unmet demand rolls forward and ages. The heuristic is
transparent but greedy — it does not optimize across months; a MILP is a
documented Version-2 candidate.

**Monte Carlo.** Correlated common-factor model: eight factors (semicap cycle,
AI/HPC, memory, mobile, auto/industrial, component tightness, logistics, EMS
execution) follow AR(1) paths; every stochastic variable loads on factors with
loadings whose squared sum ≤ 1, so the implied correlation matrix is positive
semidefinite by construction. Distributions are bounded: lognormal mean-one
multipliers for demand/cost, Bernoulli disruption events, beta-shaped
acceptance-slip fractions, clipped probabilities. Demand timing shocks
(pull-ins, push-outs, cancellations) shift units between months; supply is
rationed proportionally within each month (a vectorized approximation of the
baseline priority order — documented simplification). Revenue recognizes at
shipment or one month later for acceptance-based families, with stochastic
acceptance/site-readiness slip.

**Financial translation.** COGS = material (with PPV, FX and tightness-driven
variance) + EMS conversion + integration and test + freight + warranty + scrap
+ rework + expedite premiums + overtime premiums. Operating income subtracts
opex and one-time action costs; EBITDA adds back depreciation; the cash-flow
proxy is EBITDA − Δworking capital − capex − cash taxes. Working capital is
inventory + simplified receivables (DSO) − simplified payables (DPO). E&O is a
reserve rate on critical-component stock above 2.5 months of forward usage
plus 5% of aged finished goods.

**Known simplifications.** Monthly buckets; family-level Monte Carlo (customer
detail lives in the deterministic baseline); proportional within-month
rationing; no balance-sheet FX; recognition simplified to a 0/1-month lag with
stochastic slip; overtime/reservation costs approximated. A fast, credible
prototype is preferred to an unusably detailed model — see README for the full
list and Version-2 candidates.
        """)
