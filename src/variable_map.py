"""Variable map: every input that can change a simulation result.

One row per variable: its layer, where it lives, who should own it, what it
does in the engine, and which outputs it moves. `tests/test_variable_map.py`
fails if an engine knob (a lever, shock switch, uncertainty or financial
setting) is missing here, so the map cannot drift from the code.

Layers, from the world to what we do:
  Volatility      how uncertain things are (config sigmas, factor model)
  Event rate      how often things happen and the operating facts (data tables)
  Shock switch    on/off amplitude per shock group (Shocks; zero = baseline)
  Lever           what a scenario or management action changes (sim params)
  Financial       how operations translate to money (config.financial)
  Engine constant hard-coded in the engine; no owner yet (calibration candidate)
"""
from __future__ import annotations

import pandas as pd

LAYERS = ["Volatility", "Event rate", "Shock switch", "Lever", "Financial",
          "Engine constant"]

# (variable, layer, where it lives, owner, what it does, outputs moved)
_ROWS: list[tuple[str, str, str, str, str, str]] = [
    # ---- Volatility -------------------------------------------------------
    ("market_demand_sigma", "Volatility", "config/default_config.yaml: uncertainty",
     "Demand Planning", "Lognormal volatility of demand by end market",
     "Revenue, service, P(plan), inventory"),
    ("customer_idiosyncratic_sigma", "Volatility", "config/default_config.yaml: uncertainty",
     "Demand Planning", "Extra demand volatility per family (per customer once the customer dimension lands)",
     "Revenue, service"),
    ("asp_sigma", "Volatility", "config/default_config.yaml: uncertainty",
     "Pricing", "Realized ASP variability", "Revenue, GM"),
    ("material_cost_sigma", "Volatility", "config/default_config.yaml: uncertainty",
     "Cost Accounting", "Purchase price variance on material", "COGS, GM"),
    ("fx_cost_sigma", "Volatility", "config/default_config.yaml: uncertainty",
     "Treasury", "FX-driven material cost variance", "COGS, GM"),
    ("conversion_cost_sigma", "Volatility", "config/default_config.yaml: uncertainty",
     "Cost Accounting", "EMS conversion cost variance", "COGS, GM"),
    ("freight_sigma", "Volatility", "config/default_config.yaml: uncertainty",
     "Logistics", "Freight cost variance", "COGS, GM"),
    ("ems_labor_sigma", "Volatility", "config/default_config.yaml: uncertainty",
     "EMS Program Mgmt", "EMS labor and execution swings on site capacity", "Shipments, revenue"),
    ("site_disruption_impact", "Volatility", "config/default_config.yaml: uncertainty",
     "Risk / EMS Program Mgmt", "Share of site capacity lost in a disrupted month",
     "Shipments, revenue"),
    ("factors (loadings, persistence)", "Volatility", "config/default_config.yaml: factors",
     "FP&A / Demand Planning", "Which shocks move together, and how long cycles persist",
     "Width of every distribution, P(plan)"),
    # ---- Event rates and operating facts (data tables) ---------------------
    ("base_forecast_units, backlog_units", "Event rate", "demand_plan",
     "Demand Planning / Sales Ops", "The demand plan: forecast plus backlog by customer, family, month",
     "Everything"),
    ("asp_usd", "Event rate", "demand_plan", "Pricing",
     "Selling price by customer and family", "Revenue, GM, customer revenue"),
    ("customer, customer_priority", "Event rate", "demand_plan", "SIOP leadership",
     "Demand lines, and the serving order under strict priority", "Who gets shorted, customer revenue"),
    ("customer_group", "Event rate", "demand_plan", "Sales Ops",
     "Customers in a group share part of their demand shock", "Spread of customer revenue"),
    ("lost_after_months", "Event rate", "config/default_config.yaml: customers", "Sales",
     "Months a shorted order waits before it is lost, per customer (default: waits)",
     "Lost revenue, revenue, backlog"),
    ("push_out_prob, pull_in_prob, cancel_prob", "Event rate", "demand_plan", "Sales Ops",
     "How often orders move or cancel, by customer, family and month; a push-out "
     "moves the whole line-month order", "Revenue timing, P(Q1 plan), service"),
    ("site_readiness_prob", "Event rate", "demand_plan", "Service",
     "Odds a customer site is ready; low readiness slips acceptance", "Revenue timing"),
    ("acceptance_lag_months", "Event rate", "products", "Revenue Accounting",
     "Recognition at shipment or one month later; base acceptance slip odds", "Revenue timing"),
    ("config_complexity", "Event rate", "products", "EMS Program Mgmt",
     "EMS capacity consumed per system (std-equivalent weight)", "Shipments, utilization"),
    ("qualified_ems_sites", "Event rate", "products", "Operations Engineering",
     "Which EMS sites can build each family", "Shipments, revenue"),
    ("material_cost_usd, ems_conversion_cost_usd, integration_test_cost_usd, "
     "freight_cost_usd, warranty_reserve_usd, scrap_prob", "Event rate", "products",
     "Cost Accounting", "Standard cost build-up per system",
     "COGS, GM, inventory"),
    ("build_cycle_months", "Event rate", "products", "Planning", "Months a system sits in WIP",
     "Inventory"),
    ("on_hand_units, open_po_units_per_month", "Event rate", "components", "Materials / Procurement",
     "Critical-part stock; open POs inside each part's lead time, and the "
     "supplier's committed rate", "Shipments, inventory, E&O"),
    ("usage_per_system, products_using", "Event rate", "components", "Engineering",
     "Critical-part BOM by family", "Shipments, inventory"),
    ("safety_stock_units", "Event rate", "components", "Materials",
     "Safety-stock policy: the buffer buyers order toward, usable when parts "
     "run short", "Service, inventory, cash"),
    ("lead_time_weeks", "Event rate", "components", "Procurement",
     "How long before purchases can respond (open POs cover it); also "
     "structural receipt lateness", "Shortage under demand change, inventory, expedite cost"),
    ("allocation_risk", "Event rate", "components", "Procurement",
     "Supplier allocation exposure; more lateness when supply is tight", "Shipments, expedite cost"),
    ("disruption_prob_monthly", "Event rate", "components", "Procurement",
     "Monthly odds a supplier loses 65% of receipts", "Shipments, revenue"),
    ("expedite_available, expedite_premium_pct", "Event rate", "components", "Procurement",
     "Whether late receipts can be pulled in, and at what premium", "Shipments, COGS"),
    ("unit_cost_usd", "Event rate", "components", "Cost Accounting",
     "Critical-part cost", "Inventory, E&O, payables, expedite cost"),
    ("obsolescence_risk", "Event rate", "components", "Engineering / Controller",
     "Probability a part becomes obsolete: raises its E&O reserve rate toward "
     "full write-off", "E&O"),
    ("available_capacity_units, max_overtime_units", "Event rate", "ems_capacity",
     "EMS Program Mgmt", "EMS site capacity by month, including final integration and test",
     "Shipments, revenue"),
    ("schedule_adherence, labor_availability", "Event rate", "ems_capacity", "EMS Program Mgmt",
     "Derate capacity per site and month", "Shipments, revenue"),
    ("first_pass_yield", "Event rate", "ems_capacity", "Quality",
     "Failed units are reworked: extra EMS load and rework cost", "Shipments, COGS, GM"),
    ("cost_per_std_unit_usd, overtime_premium_pct", "Event rate", "ems_sites", "Procurement",
     "Site cost order for filling, and the overtime premium", "COGS, fill order"),
    ("regional_disruption_prob_monthly", "Event rate", "ems_sites", "Risk / EMS Program Mgmt",
     "Monthly odds a site is disrupted", "Shipments, revenue"),
    ("revenue_plan_usd", "Event rate", "financial_plan", "FP&A",
     "The plan of record every P(plan) is measured against", "P(plan), revenue at risk"),
    # ---- Shock switches ------------------------------------------------------
    ("demand", "Shock switch", "src/shocks.py", "Engine", "Scales demand volatility", "Spread of revenue"),
    ("timing_events", "Shock switch", "src/shocks.py", "Engine",
     "Scales cancel, push-out, pull-in rates and their noise", "Revenue timing"),
    ("price_cost", "Shock switch", "src/shocks.py", "Engine",
     "Scales ASP, material, FX, conversion, freight volatility", "Spread of GM"),
    ("component_tightness", "Shock switch", "src/shocks.py", "Engine",
     "Tightness factor: lateness, disruption odds, material cost", "Shipments, COGS"),
    ("logistics", "Shock switch", "src/shocks.py", "Engine",
     "Logistics factor: lateness, freight cost", "Shipments, COGS"),
    ("receipt_delay", "Shock switch", "src/shocks.py", "Engine",
     "Structural lateness by lead time", "Expedite cost, timing"),
    ("component_disruption", "Shock switch", "src/shocks.py", "Engine",
     "Supplier disruption events", "Shipments, revenue"),
    ("ems_execution", "Shock switch", "src/shocks.py", "Engine",
     "EMS labor and execution factor (capacity, yield)", "Shipments, COGS"),
    ("site_disruption", "Shock switch", "src/shocks.py", "Engine",
     "EMS regional disruption events", "Shipments, revenue"),
    ("acceptance_slip", "Shock switch", "src/shocks.py", "Engine",
     "Acceptance and site-readiness slip", "Revenue timing"),
    # ---- Levers (scenarios and management actions) ----------------------------
    ("demand_market_mult", "Lever", "src/simulation.py: default_params", "Scenario",
     "Demand multiplier by end market (can vary by month)", "Revenue, service, inventory"),
    ("demand_family_mult", "Lever", "src/simulation.py: default_params", "Scenario",
     "Demand multiplier by family", "Revenue, service, inventory"),
    ("demand_sigma_mult", "Lever", "src/simulation.py: default_params", "Demand Confidence",
     "Widens or narrows demand volatility", "Spread of revenue, P(plan)"),
    ("pushout_prob_add", "Lever", "src/simulation.py: default_params", "Demand Confidence / Scenario",
     "Adds to push-out odds", "Revenue timing"),
    ("pullin_prob_add", "Lever", "src/simulation.py: default_params", "Scenario",
     "Adds to pull-in odds", "Revenue timing"),
    ("cancel_prob_mult", "Lever", "src/simulation.py: default_params", "Demand Confidence / Scenario",
     "Scales cancellation odds", "Revenue"),
    ("asp_mult", "Lever", "src/simulation.py: default_params", "Scenario",
     "Scales selling price", "Revenue, GM"),
    ("forced_pushout", "Lever", "src/simulation.py: default_params", "Scenario",
     "Moves a named quantity of a family from one month to another", "Revenue timing"),
    ("lead_time_mult", "Lever", "src/simulation.py: default_params", "Scenario",
     "Scales lead time: a longer open-PO window before buying responds, "
     "plus lateness", "Shortage, inventory, expedite cost"),
    ("comp_disrupt_mult", "Lever", "src/simulation.py: default_params", "Scenario",
     "Scales supplier disruption odds", "Shipments, revenue"),
    ("comp_supply_mult", "Lever", "src/simulation.py: default_params", "Scenario / Action",
     "Scales supplier capacity by part: a cut applies to open POs at once, "
     "an increase only after the lead time", "Shipments, inventory"),
    ("comp_supply_ramp", "Lever", "src/simulation.py: default_params", "Action",
     "Scales supplier capacity by part from a start month (no sooner than "
     "the lead time for increases)", "Shipments, inventory"),
    ("safety_stock_mult", "Lever", "src/simulation.py: default_params", "Action",
     "Scales the safety-stock policy, the buffer target buyers order toward",
     "Service, inventory, cash"),
    ("expedite_recovery", "Lever", "src/simulation.py: default_params", "Action",
     "Share of late receipts that can be expedited", "Shipments, COGS"),
    ("expedite_recovery_by_comp", "Lever", "src/simulation.py: default_params", "Action",
     "Targeted expedite share by part", "Shipments, COGS"),
    ("expedite_premium_mult", "Lever", "src/simulation.py: default_params", "Scenario / Action",
     "Scales expedite premiums", "COGS"),
    ("ems_capacity_mult", "Lever", "src/simulation.py: default_params", "Scenario",
     "Scales EMS site capacity", "Shipments, revenue"),
    ("ems_capacity_add", "Lever", "src/simulation.py: default_params", "Action",
     "Adds std-units of site capacity", "Shipments, revenue"),
    ("ems_capacity_add_ramp", "Lever", "src/simulation.py: default_params", "Action",
     "Adds site capacity from a start month", "Shipments, revenue"),
    ("ems_window_mult", "Lever", "src/simulation.py: default_params", "Scenario",
     "Temporary capacity change at a site", "Shipments, revenue"),
    ("adherence_delta", "Lever", "src/simulation.py: default_params", "Scenario",
     "Shifts schedule adherence at every site", "Shipments, revenue"),
    ("fpy_delta", "Lever", "src/simulation.py: default_params", "Scenario",
     "Shifts first-pass yield: EMS load from rework and rework cost", "Shipments, COGS"),
    ("overtime_fraction", "Lever", "src/simulation.py: default_params", "Action",
     "Share of maximum overtime authorized", "Shipments, COGS"),
    ("overtime_start_month", "Lever", "src/simulation.py: default_params", "Action",
     "Month overtime takes effect", "Shipments timing"),
    ("add_qualification", "Lever", "src/simulation.py: default_params", "Action",
     "Qualifies a family at a site from a month", "Shipments, revenue"),
    ("acceptance_delay_add", "Lever", "src/simulation.py: default_params", "Scenario",
     "Adds to acceptance slip odds", "Revenue timing"),
    ("freight_mult", "Lever", "src/simulation.py: default_params", "Scenario",
     "Scales freight cost", "COGS"),
    ("material_cost_mult", "Lever", "src/simulation.py: default_params", "Scenario",
     "Scales material cost", "COGS"),
    ("allocation_policy", "Lever", "src/simulation.py: default_params", "SIOP leadership",
     "Who is served first when supply is short: strict priority backlog first "
     "(default), strict priority priority first, proportional, or protect top customers", "Customer revenue and fill; GM only through mix"),
    ("protect_top_n", "Lever", "src/simulation.py: default_params", "SIOP leadership",
     "How many top customers the protect policy serves first", "Customer revenue and fill"),
    ("action_cost_usd", "Lever", "src/simulation.py: default_params", "Action",
     "One-time decision cost, spread over Q1", "Operating income"),
    ("recurring_cost", "Lever", "config/management_actions.yaml: "
     "recurring_cost_usd_per_month", "Action",
     "Monthly cost of a permanent action while it is held (take-or-pay, headcount)",
     "Operating income, action EV"),
    # ---- Financial translation ------------------------------------------------
    ("gross_margin_target", "Financial", "config/default_config.yaml: financial", "FP&A",
     "Target for P(GM target)", "P(GM target), risks"),
    ("inventory_target_usd", "Financial", "config/default_config.yaml: financial", "FP&A",
     "Target for P(inventory over target)", "Inventory risk"),
    ("inventory_turns_target", "Financial", "config/default_config.yaml: financial", "FP&A",
     "Displayed only", "None"),
    ("opex_monthly_usd", "Financial", "config/default_config.yaml: financial", "FP&A",
     "Operating expense", "Operating income, cash"),
    ("depreciation_monthly_usd", "Financial", "config/default_config.yaml: financial", "FP&A",
     "Added back for EBITDA", "EBITDA, cash"),
    ("capex_monthly_usd", "Financial", "config/default_config.yaml: financial", "FP&A",
     "Capital spend", "Cash"),
    ("tax_rate", "Financial", "config/default_config.yaml: financial", "Tax", "Cash taxes", "Cash"),
    ("dso_days", "Financial", "config/default_config.yaml: financial", "Treasury",
     "Receivables proxy", "Working capital, cash"),
    ("dpo_days", "Financial", "config/default_config.yaml: financial", "Treasury",
     "Payables: days of purchases (parts received, other material, EMS "
     "conversion and freight billed)", "Working capital, cash"),
    ("eo_reserve_rate", "Financial", "config/default_config.yaml: financial", "Controller",
     "Reserve rate on excess critical stock", "E&O"),
    ("revenue_plan_buffer", "Financial", "config/default_config.yaml: financial", "FP&A",
     "Declared but not read (the plan comes from financial_plan)", "None"),
    # ---- Engine constants (no owner yet) ---------------------------------------
    ("backlog cancels at half the forecast rate", "Engine constant",
     "src/simulation.py FIRM_CANCEL_SHARE", "None yet",
     "Booked orders cancel less often than forecast", "Revenue"),
    ("customer group share 0.5", "Engine constant", "src/simulation.py CUSTOMER_GROUP_SHARE",
     "None yet", "Share of a customer's demand shock shared with its group",
     "Spread of customer and top-N revenue"),
    ("timing noise sigma 0.35", "Engine constant", "src/simulation.py section 1", "None yet",
     "Path-level variation in push-out and pull-in intensity", "Revenue timing spread"),
    ("push-out split 70% / 30%", "Engine constant", "src/simulation.py section 1", "None yet",
     "Share of push-outs landing one vs two months later", "Revenue timing"),
    ("receipt lateness (LT - 8) / 100", "Engine constant", "src/simulation.py section 2", "None yet",
     "Structural share of receipts one month late, 1% to 25%", "Expedite cost, timing"),
    ("tightness and logistics lateness 0.06 / 0.03", "Engine constant", "src/simulation.py section 2",
     "None yet", "Extra lateness when supply or logistics is tight", "Shipments, expedite cost"),
    ("disrupted receipts kept 35%", "Engine constant", "src/simulation.py section 2", "None yet",
     "Share of receipts still arriving in a disrupted month", "Shipments"),
    ("labor multiplier clip 0.7 to 1.1", "Engine constant", "src/simulation.py section 3", "None yet",
     "Bounds on EMS labor swings", "Shipments"),
    ("FPY execution drag 0.02", "Engine constant", "src/simulation.py section 3", "None yet",
     "Yield loss when EMS execution is poor", "COGS"),
    ("rework takes 50% of a build slot and of conversion cost", "Engine constant",
     "src/operations.py REWORK_SHARE", "None yet",
     "EMS load and cost of reworking a unit that fails first pass", "Shipments, COGS"),
    ("acceptance slip noise beta(4, 6)", "Engine constant", "src/simulation.py section 5", "None yet",
     "Path-level variation in acceptance slip", "Revenue timing spread"),
    ("expedite on component shortage 2%", "Engine constant", "src/simulation.py section 6", "None yet",
     "Premium per component-short unit", "COGS"),
    ("non-critical material 0.9 months", "Engine constant", "src/simulation.py section 6", "None yet",
     "Proxy for raw material outside the 30 critical parts", "Inventory"),
    ("WIP at 60% of cycle cost", "Engine constant", "src/simulation.py section 6", "None yet",
     "WIP valuation", "Inventory"),
    ("supplier upside flex 25%", "Engine constant", "src/simulation.py SUPPLIER_UPSIDE_FLEX",
     "None yet", "Suppliers deliver up to 125% of the open-PO rate (cumulative; none when cut)",
     "Upside shipments, shortage"),
    ("buyer run rate: trailing 3 months, clipped 0.5 to 2", "Engine constant",
     "src/simulation.py section 4", "None yet",
     "How buyers read demand when scaling the plan for planned orders", "Shortage, inventory"),
    ("E&O window 2.5 months, FG 5%", "Engine constant", "src/simulation.py section 6", "None yet",
     "Excess threshold (months of expected usage at the demand run rate) and "
     "aged finished-goods reserve", "E&O"),
]

COLUMNS = ["Variable", "Layer", "Where it lives", "Owner", "What it does", "Outputs moved"]


def variable_map() -> pd.DataFrame:
    """The full map as a table, in layer order."""
    df = pd.DataFrame(_ROWS, columns=COLUMNS)
    df["Layer"] = pd.Categorical(df["Layer"], categories=LAYERS, ordered=True)
    return df.sort_values("Layer", kind="stable").reset_index(drop=True)


def mapped_names() -> set[str]:
    """Every individual name covered, splitting comma-joined rows."""
    out: set[str] = set()
    for name, *_ in _ROWS:
        out.update(part.strip() for part in name.split(","))
    return out
