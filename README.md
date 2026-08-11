# Apex Test Systems — SIOP Risk & Scenario Engine

A working Version-1 prototype of an executive decision-support system for the
monthly SIOP (Sales, Inventory & Operations Planning) cycle of a fictional
automated test equipment (ATE) manufacturer, **Apex Test Systems**. It connects
operating uncertainty — demand cycles, customer push-outs, component shortages,
EMS capacity, integration throughput, acceptance timing — to the financial
outcomes a COO, a VP of Global Operations Finance and a Director of SIOP
actually manage: revenue attainment probability, gross margin, inventory,
working capital and cash.

**All data is synthetic.** Company names, customers, suppliers, products and
every operating assumption are generated from a seeded random process. Nothing
is drawn from any real company.

## What it answers

- What is the probability of achieving the quarterly revenue plan? The margin target?
- How much revenue is at risk, and which families, customers, components and
  sites drive it?
- How likely are push-outs or shortages to move revenue between quarters?
- Can the EMS network support demand upside without excess inventory?
- What should leadership decide at the next SIOP meeting — and what does each
  action cost, protect, and trade off?

## Architecture

```
initial demand/supply tables  →  deterministic baseline plan  →  correlated
Monte Carlo simulation  →  financial translation  →  scenario & action
comparison  →  sensitivity ranking  →  recommendations  →  executive summary
→  dashboard + Excel export
```

| Module | Responsibility |
|---|---|
| `src/data_generator.py` | Seeded synthetic tables: demand (customer × family × month), products/BOM, 30 critical components, EMS sites & monthly capacity, integration capacity, financial plan |
| `src/validation.py` | Input validation (probabilities, bounds, horizon alignment, qualification integrity), PSD check/repair, friendly messages |
| `src/operations.py` | Dense planning arrays shared by baseline and simulator |
| `src/baseline_plan.py` | Deterministic monthly plan via a documented greedy heuristic |
| `src/correlations.py` | Common-factor model (8 factors, AR(1) paths); PSD by construction |
| `src/simulation.py` | Vectorized correlated Monte Carlo + full financial translation |
| `src/scenarios.py` | 8 prebuilt scenarios (exogenous world-states), custom scenario, 14 management actions across SIOP horizons (combinable into response packages), KPI summary & comparison — see `docs/ARCHITECTURE.md` for the (world, response) evaluation-context model |
| `src/market_intelligence.py` | Demand Confidence engine: market signals, external intelligence, customer confidence, and the confidence→simulation mapping |
| `src/sensitivity.py` | Spearman rank-correlation driver rankings, binding constraints, revenue-at-risk views |
| `src/recommendations.py` | Threshold-triggered, simulation-backed, scored recommendations |
| `src/executive_report.py` | Rules-based dynamic executive summary (LLM-ready abstraction, no API needed) |
| `src/visualizations.py` | Executive Plotly figures (validated accessible palette) |
| `src/exports.py` | 15-sheet formatted Excel workbook (plus a decision-of-record sheet when an evaluation context is active) |
| `app.py` | 11-tab Streamlit executive dashboard |

## Demand Confidence

The **Market Intelligence & Demand Confidence** page (after Demand & Backlog)
answers a different question than the demand plan itself: *how much confidence
should management have in that plan?* A composite of eight weighted demand
signals (push-out rate, forecast accuracy, book-to-bill, booking momentum,
cancellations, backlog growth/aging, expedites) plus synthesized external
intelligence produces a Demand Confidence level (Very High … Very Low). That
level maps to a visible table of simulation adjustments — demand-volatility
scaling, push-out probability, cancellation frequency — applied to the base
case, every scenario and every management action. Lower confidence widens
every probability distribution in the app; the page's sensitivity section
quantifies exactly what each level is worth in revenue, margin, EBITDA,
inventory, working capital and cash terms. Recommendations and the executive
summary reference the assessment where it changes how an action should be read.

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py     # dashboard at http://localhost:8501
pytest                   # automated tests
```

First launch generates synthetic CSVs into `data/generated/`. Use the sidebar
to change the seed, simulation depth (1,000 / 5,000 / 10,000 paths), scenario,
or to regenerate data. Results are cached and reproducible per seed.

## Synthetic company design

- **5 product families** — Zenith Compute Test, Vector Memory Test, Horizon
  Mobility Test, Atlas Automotive & Industrial Test, Nexus System-Level Test —
  mapped to end markets (AI/HPC, memory, mobile, automotive, industrial).
- **8 customers** across IDM, fabless compute, memory, automotive, mobile,
  OSAT and diversified groups in 5 regions. OSATs are *customers*; Apex builds
  through **EMS partners** (Americas, Malaysia, Taiwan, Eastern Europe) and two
  final-integration sites — it owns no wafer fab and no fab concepts are used.
- **30 critical components** (FPGAs, precision instrumentation, custom PCBs,
  power, interconnect, thermal, interface hardware…) with lead times, safety
  stock, allocation/disruption risk, expedite and alternate-source attributes.
- Scale: ~$2.5-2.9B annual revenue, ~95-110 systems/month, ASPs $1.5-3.4M,
  gross margins in the mid-50s — calibrated to resemble a diversified ATE
  maker's operating model without using any real company's data.

## Baseline planning logic

A transparent greedy heuristic (documented limitation: no cross-month
optimization):

1. Firm backlog before forecast; 2. higher customer priority; 3. earlier
requested date; 4. higher contribution margin per standard-equivalent unit;
5. only qualified EMS sites (least-contested site first, then cost);
6. component availability, EMS capacity (derated by schedule adherence and
labor) and final-integration capacity are hard constraints. Unmet demand rolls
forward and ages; every shortfall is logged with its binding constraint.

## Monte Carlo methodology

- **Correlation:** eight common factors (global semicap spending, AI/HPC,
  memory, mobile, auto/industrial, component tightness, logistics disruption,
  EMS labor/execution) follow AR(1) paths with 0.85 persistence. Every
  stochastic variable loads on factors; squared loadings sum ≤ 1, so the
  implied correlation matrix is **positive semidefinite by construction** (also
  verified by test). Loadings are editable in `config/default_config.yaml` and
  displayed in the app.
- **Distributions are bounded:** mean-one lognormal multipliers for demand,
  ASP, PPV/FX, freight; Bernoulli disruption events; beta-shaped acceptance
  slip; clipped probabilities — no impossible values.
- **Modeled uncertainty:** market and customer demand, pull-ins/push-outs/
  cancellations, ASP, component receipts/lateness/disruption/allocation,
  expedite recovery and premiums, EMS capacity/labor/adherence/yield/regional
  disruption, utilization-dependent adherence erosion, integration capacity,
  acceptance and site-readiness delays, cost variances.
- **Hybrid granularity (documented):** the deterministic baseline runs at
  customer/site/component detail; the simulator runs family × month with all
  30 components and site capacities, rationing scarce supply proportionally
  within each month. With zero shocks it reconciles to the baseline. 10,000
  paths run in ~1-2 seconds on a laptop.

## Financial translation

Revenue = recognized units × realized ASP (recognition at shipment, or one
month later for acceptance-based families, with stochastic slip). COGS =
material (PPV, FX, tightness variance) + EMS conversion + integration/test +
freight + warranty + scrap + rework + expedite premiums + overtime premiums.
Operating income subtracts opex and one-time action costs; EBITDA proxy adds
back depreciation; cash proxy = EBITDA − ΔWC − capex − cash taxes. Working
capital = inventory + receivables (DSO) − payables (DPO). E&O reserves apply
to component stock above 2.5 months forward usage plus aged finished goods.

## Scenarios & recommendations

Eight prebuilt scenarios — exogenous world-states only (base case, AI demand
surge, memory recovery, memory recovery delay, major customer push-out,
critical FPGA shortage, EMS Malaysia disruption, customer site-readiness
delay) — plus a fully custom scenario. Fourteen management actions across
three SIOP horizons (Execution / Tactical / Long-lead), combinable into costed
response packages, are each simulated with **common random numbers** against a
no-action reference at the same path count and seed so deltas isolate the
action's effect. Recommendations fire only from measurable risk thresholds,
cite the simulated impact of a real action, pass materiality gates, and expose
their scoring weights (EV 35%, probability 30%, revenue 20%, cash −15%).

## Test results

`pytest` — **69 passed** (data reproducibility & validity, baseline
feasibility vs component/EMS/integration constraints, financial identities,
simulation reproducibility & bounds, correlation PSD & co-movement, all 8
scenarios run, comparison reconciliation, equal-path-count action pricing,
recommendation traceability & materiality, dynamic summary — including the
confidence-override wording, Excel export integrity, and headless AppTest
runs of the full dashboard).

## Known limitations (Version 1)

- Monthly buckets; no weekly granularity.
- Monte Carlo is family-level; customer-level detail lives in the baseline.
- Greedy allocation, not optimization; proportional within-month rationing in
  the simulator.
- Simplified revenue recognition (0/1-month lag + stochastic slip), no
  balance-sheet FX, approximate overtime/reservation cost mechanics.
- Recommendations evaluate actions independently (no combined-action search).

## Version 2 candidates

1. Combined-action optimization (portfolio of actions under a budget).
2. Optional MILP allocation (PuLP/OR-Tools) beside the heuristic.
3. Customer-level Monte Carlo and weekly near-term buckets.
4. Bayesian updating of factor states from actuals (rolling SIOP learning).
5. Optional LLM narrative provider behind the existing abstraction.
6. PowerPoint export; distribution fitting from history.

## License and provenance

Copyright © 2026 Natalie Gendron. All rights reserved — see [LICENSE](LICENSE).

This is **independent portfolio work**, built to demonstrate an approach to executive SIOP risk
and scenario modeling. The repository is public so the work can be read and evaluated; it is
**not open source**. Commercial, production, or internal business use requires written
permission.

**Every figure in this repository is fictional.** The manufacturer, demand plans, supply
constraints, financials, and risk scenarios were invented for demonstration. Nothing here
represents or derives from the confidential information or actual economics of any real
organization — that was a deliberate design constraint, not an omission.
