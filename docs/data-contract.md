# APEX data contract (draft, 2026-09)

APEX turns operational decisions into revenue, gross margin, inventory, E&O and cash consequences. Today every input is synthetic (`src/data_generator.py`, 7 tables). This document defines the minimum real data that would replace those tables, where it would come from, and which fields the engine actually reads. Governing rule: a field belongs in the contract only if leaving it out would change a financial answer. Everything below is read-only input; APEX never writes back to a source system.

**Status legend.** *Used*: read by the engine (`operations.build_planning_arrays`, `baseline_plan`, `simulation`). *Used (score only)*: read only by the Demand Confidence score (`market_intelligence`), validation or exports, so it does not move a financial number. *Generated but unused*: produced by the generator, never read. *Missing*: needed, no synthetic stand-in exists.

Horizon: 18 monthly buckets. Refresh cadence assumes a monthly SIOP cycle with a weekly supply check.

---

## 1. Minimum data contract

### 1.1 Demand plan (Demand Planning, CRM)

| Field | Grain | Refresh | Likely owner | Replaces (synthetic) | Status |
|---|---|---|---|---|---|
| Consensus demand, forecast (unbooked) units | customer x family x month | Monthly | Demand Planning | `demand_plan.base_forecast_units` | Used |
| Firm backlog units | customer x family x month (ideally order line) | Weekly | Sales Ops / CRM | `demand_plan.backlog_units` | Used (baseline serves backlog first; sim uses total only) |
| Customer requested date | order line or customer x family x month | Weekly | Sales Ops / CRM | `demand_plan.requested_month` | Used (baseline sort tie-break only; sim ignores) |
| Committed (promised) date | order line | Weekly | Planning / Order Mgmt | `demand_plan.committed_month` | Generated but unused (always equals `month`) |
| Bookings units | customer x family x month | Monthly | Sales Ops | `demand_plan.bookings_units` | Generated but unused (copy of backlog) |
| Customer priority / allocation tier | customer | Quarterly | SIOP leadership | `demand_plan.customer_priority` | Used (baseline sort) |
| Customer group, region | customer | On change | Sales Ops | `demand_plan.customer_group`, `.region` | Used (score only / exports) |
| Customer ASP | customer x family (x month if price changes) | Quarterly | Pricing / ERP | `demand_plan.asp_usd` | Used, but collapsed to a demand-weighted family average before revenue is computed |
| Customer site readiness (install sites ready) | customer x family | Monthly | Service / CRM | `demand_plan.site_readiness_prob` | Used (sim acceptance slip) |
| Upside / downside scenario units | customer x family x month | Monthly | Demand Planning | `demand_plan.upside_units`, `.downside_units` | Generated but unused |
| New product flag | family (or SKU) | On change | Product Mgmt | `demand_plan.new_product_flag` | Generated but unused |
| Order line identifier | order line | Weekly | CRM / ERP | none | Missing (needed for pegging and cancel history) |

### 1.2 Planning-system visibility (planning system, e.g. Kinaxis)

| Field | Grain | Refresh | Likely owner | Replaces (synthetic) | Status |
|---|---|---|---|---|---|
| Critical part list | part | Quarterly | Supply Chain | `components.component` (30 parts) | Used |
| On-hand by critical part | part x location (EMS site / hub) | Weekly | Materials | `components.on_hand_units` | Used (one company-wide pool; no location) |
| Open POs, dated | part x PO line x confirmed date x qty | Weekly | Procurement | `components.open_po_units_per_month` (flat monthly rate) | Used (as a flat rate; dated receipts Missing) |
| Supplier lead time | part x supplier | Monthly | Procurement | `components.lead_time_weeks` | Used (sim delay fraction) |
| Lead-time variability | part x supplier | Quarterly | Procurement | `components.lead_time_std_weeks` | Generated but unused |
| Safety stock policy | part (x location) | Quarterly | Materials | `components.safety_stock_units` | Used (sim: 30% hard floor; baseline ignores it) |
| Minimum order quantity | part x supplier | On change | Procurement | `components.min_order_qty` | Generated but unused |
| Usage per system, critical parts | part x family | On BOM change | Engineering / Planning | `components.usage_per_system`, `.products_using` | Used |
| Supplier allocation exposure | part | Monthly | Procurement | `components.allocation_risk` | Used (sim delay fraction) |
| Expedite availability and premium | part | Quarterly | Procurement | `components.expedite_available`, `.expedite_premium_pct` | Used (sim) |
| Alternate source available, qualification lag | part | On change | Procurement / Engineering | `components.alt_source_available`, `.alt_source_qual_months`; `products.alt_source_eligible` | Generated but unused (dual-source action hard-codes month 6 in `config/management_actions.yaml`) |
| EMS capacity | EMS site x month, std-equivalent systems | Monthly | EMS Program Mgmt | `ems_capacity.available_capacity_units` | Used |
| Capacity weight per family (config complexity) | family (x site if it differs) | On change | EMS Program Mgmt / Planning | `products.config_complexity` | Used |
| Qualification matrix | family x EMS site (with effective date) | On change | Operations Engineering | `products.qualified_ems_sites` | Used (`ems_sites.eligible_families` duplicate is unused) |
| Schedule adherence, labor availability, first-pass yield | EMS site x month | Monthly | EMS scorecards | `ems_capacity.schedule_adherence`, `.labor_availability`, `.first_pass_yield` | Used |
| Build cycle time | family (x site) | Quarterly | Planning | `products.build_cycle_months` | Used (WIP value); `ems_capacity.cycle_time_weeks` Generated but unused |
| Final integration and test capacity at EMS | EMS site x month | Monthly | EMS Program Mgmt | `integration_capacity.integration_capacity_units` (modeled as two in-house sites) | Used, but structurally wrong: target business integrates at the EMS, so this should fold into EMS capacity or become a per-site test-cell constraint |
| Installation capacity | region x month | Monthly | Field Service | `integration_capacity.installation_capacity_units` | Read, never binds (baseline decrements it, sim ignores) |
| Allocation / priority rules | rule set | On change | SIOP leadership | none (hard-coded heuristic in `baseline_plan.py`; proportional rationing in `simulation.py`) | Missing |
| Constrained supply plan (if solved) | family x site x month, pegged to customer | Weekly | Planning | none (APEX solves its own rough cut) | Missing, unconfirmed whether it exists |

### 1.3 ERP (Finance systems, Cost Accounting)

| Field | Grain | Refresh | Likely owner | Replaces (synthetic) | Status |
|---|---|---|---|---|---|
| Standard cost build-up (material, EMS conversion, integration/test, freight, warranty) | family | Quarterly (standard roll) | Cost Accounting | `products.material_cost_usd`, `.ems_conversion_cost_usd`, `.integration_test_cost_usd`, `.freight_cost_usd`, `.warranty_reserve_usd` | Used |
| Standard cost, critical part | part | Quarterly | Cost Accounting | `components.unit_cost_usd` | Used |
| EMS conversion cost per std unit | EMS site | Per contract | Cost Accounting / EMS contracts | `ems_sites.cost_per_std_unit_usd` | Used (overtime premium base) |
| ASP by customer x family | customer x family | Quarterly | Pricing | `demand_plan.asp_usd` (`products.list_asp_usd` unused) | Used (see 1.1) |
| Rev-rec policy and acceptance lag | family (x customer where contracts differ) | On policy change | Revenue Accounting | `products.acceptance_lag_months` (lag >= 0.75 means 1-month lag) | Used; `demand_plan.revrec_method` Generated but unused |
| Rework and scrap rates | family | Quarterly | Quality / Cost Accounting | `products.rework_prob`, `.scrap_prob` | Used (COGS); `ems_capacity.rework_rate`, `.scrap_rate` Generated but unused |
| Inventory value by class (RM critical, RM other, WIP, FG / awaiting acceptance) | class x month (actuals) | Monthly close | Controller | none (engine derives; non-critical RM is a 0.9-month proxy) | Missing (needed to anchor opening balance) |
| E&O / obsolescence policy | part or class | Annual | Controller | `config.financial.eo_reserve_rate` (0.25), hard-coded 2.5-month excess and 5% FG factor; `components.obsolescence_risk` | Rate Used; `obsolescence_risk` Used (score only: validation range check) |

### 1.4 FP&A

| Field | Grain | Refresh | Likely owner | Replaces (synthetic) | Status |
|---|---|---|---|---|---|
| Revenue plan | month (family and customer if available) | Quarterly, reforecast monthly | FP&A | `financial_plan.revenue_plan_usd`, today computed inside APEX by `_financial_plan` (3-month smoothed demand revenue x hard-coded 0.86) | Used, but violates "plan is an FP&A input" |
| Gross margin target | FY | Annual | FP&A | `config.financial.gross_margin_target` | Used |
| Inventory target | FY end | Annual | FP&A | `config.financial.inventory_target_usd` | Used |
| Inventory turns target | FY | Annual | FP&A | `config.financial.inventory_turns_target` | Used (UI display only, `app.py`) |
| Opex, depreciation | month | Quarterly | FP&A | `config.financial.opex_monthly_usd`, `.depreciation_monthly_usd` | Used |
| Capex | month | Quarterly | FP&A | `config.financial.capex_monthly_usd` | Used |
| DSO, DPO | company (customer-level DSO preferred) | Quarterly | Treasury / FP&A | `config.financial.dso_days`, `.dpo_days` | Used |
| Tax rate | company | Annual | Tax | `config.financial.tax_rate` | Used |

### 1.5 EMS contracts (Procurement, EMS Program Mgmt)

| Field | Grain | Refresh | Likely owner | Replaces (synthetic) | Status |
|---|---|---|---|---|---|
| Base (contracted) capacity | EMS site x month | Per contract / quarterly | EMS Program Mgmt | `ems_capacity.available_capacity_units` | Used |
| Reserved vs flexible capacity | EMS site x month | Quarterly | EMS Program Mgmt | `ems_capacity.reserved_capacity_units`, `.flexible_capacity_units` | Generated but unused |
| Overtime maximum | EMS site x month | Per contract | EMS Program Mgmt | `ems_capacity.max_overtime_units` | Used |
| Overtime premium | EMS site | Per contract | Procurement | `ems_sites.overtime_premium_pct` | Used |
| Take-or-pay / reservation fee | EMS site | Per contract | Procurement | `ems_sites.capacity_reservation_fee_usd` | Generated but unused (reserve-capacity action uses a hard-coded `action_cost_usd` of 4.5e6) |
| Ramp limit | EMS site | Per contract | EMS Program Mgmt | `ems_sites.max_ramp_pct_per_month` | Generated but unused |
| Minimum production lot | EMS site (x family) | Per contract | EMS Program Mgmt | `ems_sites.min_production_lot` | Generated but unused |
| Logistics lead time to customer | EMS site x region | Quarterly | Logistics | `ems_sites.logistics_lead_time_weeks` | Generated but unused |

### 1.6 Uncertainty calibration (history)

| Field | Grain | Refresh | Likely owner | Replaces (synthetic) | Status |
|---|---|---|---|---|---|
| Forecast error by customer and horizon (lag 1 to 18) | customer x family x lag | Quarterly | Demand Planning | `demand_plan.hist_forecast_error`, `.forecast_confidence`, `.demand_std_units`; `config.uncertainty.market_demand_sigma`, `.customer_idiosyncratic_sigma` | Sigmas Used; `hist_forecast_error`, `forecast_confidence` Used (score only); `demand_std_units` Generated but unused |
| Push-out, pull-in, cancel history | customer x family (x lag) | Quarterly | Sales Ops | `demand_plan.push_out_prob`, `.pull_in_prob`, `.cancel_prob` | Used |
| Supplier delivery performance (on-time, days late) | part x supplier x month | Monthly | Procurement | `components.disruption_prob_monthly`; hard-coded delay formula in `simulation.py` | Used |
| EMS regional disruption history | EMS site | Annual | Risk / EMS Program Mgmt | `ems_sites.regional_disruption_prob_monthly` | Used |
| EMS quality escapes | EMS site | Quarterly | Quality | `ems_sites.quality_escape_prob` | Generated but unused |
| Cost variability (PPV, FX, freight, conversion) | category | Annual | Cost Accounting | `config.uncertainty.*_sigma`; `ems_sites.cost_variability_pct` | Sigmas Used; `cost_variability_pct` Generated but unused |
| Acceptance slip history | family x customer | Quarterly | Revenue Accounting / Service | derived from `products.acceptance_lag_months` (`lag/3`, clipped) | Used (derived, not calibrated) |

---

## 2. Generated but never read by any engine code

Verified by `grep -rnw` over all `*.py` outside `src/data_generator.py` and `tests/` (includes `app.py`, `scripts/`).

**Read nowhere at all:**

| Table | Columns |
|---|---|
| `demand_plan` | `upside_units`, `downside_units`, `demand_std_units`, `committed_month`, `new_product_flag`, `revrec_method`, `bookings_units`, `market_segment` (engine uses `utils.FAMILY_MARKET`) |
| `products` | `integration_weeks`, `fat_weeks`, `install_weeks`, `base_first_pass_yield`, `alt_source_eligible`, `list_asp_usd` |
| `components` | `min_order_qty`, `lead_time_std_weeks`, `alt_source_available`, `alt_source_qual_months`, `category` |
| `ems_sites` | `capacity_reservation_fee_usd`, `min_production_lot`, `max_ramp_pct_per_month`, `quality_escape_prob`, `cost_variability_pct`, `logistics_lead_time_weeks`, `eligible_families` (only displayed as a raw table in `app.py`) |
| `ems_capacity` | `reserved_capacity_units`, `flexible_capacity_units`, `cycle_time_weeks`, `rework_rate`, `scrap_rate` |
| `integration_capacity` | `calibration_capacity_units`, `first_pass_completion`, `rework_days`, `customer_acceptance_weeks`, `shipping_lanes`, `eligible_families` |
| `config.financial` | `revenue_plan_buffer` (declared in `src/config.py:32`, never passed to `generate_all`, which defaults `plan_buffer=0.86`) |

**Read, but only outside the financial engine:**

| Field | Where read | Effect on financial output |
|---|---|---|
| `hist_forecast_error` | `market_intelligence.py:167, 222` (seeds the forecast-accuracy signal) | None; Demand Confidence score only |
| `forecast_confidence` | `market_intelligence.py:384, 533, 618`; `validation.py:28` | None; score and range check |
| `obsolescence_risk` | `validation.py:30` (0 to 1 range check) | None; E&O uses a flat rate |
| `fat_capacity_units` | `validation.py:40` (non-negative check) | None |
| `capacity_cost_per_unit_usd` | `validation.py:39` | None; engine uses `ems_sites.cost_per_std_unit_usd` |
| `inventory_turns_target` | `app.py:1302` (assumptions display) | None |
| `customer_group`, `region` | `exports.py:222`, `market_intelligence.py:381` | None |
| `supplier`, `supplier_region`, `monthly_requirement_units` | `exports.py:244-246` | None |

**Read, but weaker than the column implies:**

| Field | Where read | Note |
|---|---|---|
| `requested_month` | `baseline_plan.py:76` | Sort tie-break in baseline only; Monte Carlo ignores it |
| `customer_priority` | `baseline_plan.py:76` | Baseline only; sim rations proportionally, so the two engines allocate shortages differently |
| `installation_capacity_units` | `operations.py:125` | Baseline decrements `install_remaining` but never checks it; never binds |
| `safety_stock_units` | `simulation.py` (30% floor) | Baseline starts from full on-hand and ignores it |
| `asp_usd` | `operations.py` `wavg` | Customer ASP averaged to family, so customer mix does not change revenue per unit |

---

## 3. Solved constrained plan vs visibility data

APEX runs its own rough-cut allocation (`baseline_plan.py`, `simulation.py`). The question is which inputs only exist if the planning system actually solves constraints, and which a "glass window" of parts, on-hand and POs can supply.

### 3.1 Assumes a solved constrained plan (flag)

| Item | Why it assumes a solve | Risk if planning system is visibility only |
|---|---|---|
| EMS capacity by site x month in std-equivalent units | Requires capacity modeled as a constraint with a per-family consumption rate | Must be sourced from EMS contracts and EMS planners, then normalized by APEX |
| Committed / promised dates (`committed_month`) | ATP or CTP output of a constrained solve | Not available; APEX derives its own ship dates |
| Allocation / priority rules | Only meaningful if the planner encodes them in a solve | APEX heuristic becomes the de facto rule; needs SIOP sign-off |
| Qualification matrix with effective dates | Useful as data either way, but only enforced inside a solve | Maintain as a master-data list |
| Pegging of supply to customer orders | Needed to say which customer is shorted by which part | Customer-level shortfall stays a heuristic estimate |
| Reconciliation of APEX baseline to planning system constrained plan | Needs a plan to reconcile to | Baseline cannot be validated against planning truth, only against actuals |
| Dated supplier commits (confirmed vs requested PO dates) | Often maintained in a solved supply plan | Use ERP PO confirmed dates if present |

### 3.2 Visibility data is enough

| Item | Source |
|---|---|
| On-hand by critical part (and location) | Planning system or ERP inventory snapshot |
| Open POs with dates and quantities | Planning system or ERP |
| Lead times, MOQ, safety stock policy | Planning system / item master |
| Usage per system for critical parts | BOM (planning system or PLM) |
| Demand plan customer x family x month, backlog split | Demand Planning / CRM |
| Standard costs, ASPs, rev-rec policy | ERP |
| Targets, opex, capex, DSO/DPO, revenue plan | FP&A |
| Overtime, fees, ramp limits, min lots | EMS contracts |
| Calibration history | Snapshots of plan vs actual from any system that keeps history |

Implication: with visibility data alone, APEX's rough-cut solver stays the only allocation logic, so the heuristic in `baseline_plan.py` and the proportional rationing in `simulation.py` must be reconciled (see `docs/engine-reconciliation.md`) and the rules approved by SIOP.

---

## 4. Questions for the planning-system (Kinaxis) team

1. Is EMS capacity modeled as a hard constraint, by site and month (or week)? In what unit (systems, hours, test cells), and does it vary by product family?
2. Is final integration and test at each EMS modeled as a separate constraint from board/subassembly build?
3. Which parts are flagged critical or constrained, and who maintains that list?
4. Are open POs held at line level with confirmed (not just requested) dates? How often are confirmations refreshed from suppliers?
5. Is on-hand available by location (each EMS site, consigned vs owned, hubs), and is it owned or EMS-held inventory?
6. Does the system run a constrained supply plan today, or is it used for visibility and exceptions only? If it solves, what is the solve frequency?
7. Is there pegging from supply (parts, EMS builds) to customer orders or forecast lines? At what grain: customer, order line, family?
8. Are customer allocation or priority rules encoded (fair share, tiered priority, strategic customers)? Who approves them?
9. Is the demand plan held at customer x family x month with the backlog vs forecast split and requested dates, or only at family level?
10. Is the qualification matrix (family x EMS site) maintained with effective dates for new qualifications?
11. Are safety stock, MOQ, lead time and lead-time variability maintained per part and supplier, and how current are they?
12. Can scenarios be created and exported (e.g. demand upside, capacity loss), and in what format (API, flat file, scheduled extract)?
13. What history is retained: plan snapshots by cycle (for forecast error by lag), push-out and cancel events, supplier on-time delivery?
14. What is the extract cadence and latency (daily, weekly, monthly), and is there a stable snapshot aligned to the SIOP cycle date?
15. Are EMS contract terms (overtime limits, ramp limits, minimum lots, reserved capacity) captured anywhere in the system, or only in contracts?
