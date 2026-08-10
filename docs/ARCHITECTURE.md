# Architecture Note — Apex SIOP Decision Engine

*Standing reference. Check every proposed feature against this note; if a
feature doesn't fit the model here, either the feature is wrong or this note
needs a deliberate revision — never a silent exception.*

## What the engine is for

SIOP (Sales, Inventory & Operations Planning) is a monthly decision cadence.
The executive meeting it culminates in exists to answer four questions, in
order:

1. **How much should we trust the plan?**
2. **What could happen to it?**
3. **What should we do about it?**
4. **Sign the result and record it.**

The engine is decision support for that meeting. Every page, chart, and
number serves one of the four questions; anything that serves none of them
doesn't belong.

## The core model

One equation describes everything the engine computes:

> **outcome distribution = Simulate(inputs, world, response)**
> — judged against a frozen **plan of record**.

### The two axes

- **World** — what happens *to* the business. Exogenous, carries no cost.
  Composed of the **base beliefs** (calibrated monthly by evidence: the
  Demand Confidence assessment, which widens or narrows demand variance,
  push-out and cancellation assumptions in *every* simulation) plus an
  optional **scenario** deviation (demand and/or supply: surges, shortages,
  disruptions, push-outs). Bullish/bearish management postures are scenarios
  too — beliefs about the world, not decisions.
- **Response** — what the business chooses to *do*. A **package** of zero or
  more management actions from the claim-sheet catalog
  (`config/management_actions.yaml`). Each action is analyst-authored
  (capability claimed, decision cost, SIOP horizon, effective-month latency
  ramp); the simulation computes the consequences. Package semantics: action
  overrides stack via the standard layering rules (adds sum, multiplies
  multiply, dict knobs merge per key, later actions win collisions), costs
  sum and are charged to Q1 operating income by the simulator.

**The axes must never be mixed.** A scenario with a decision cost, or a
custom-scenario slider that sets safety stock, is a category error — it hides
a choice inside the weather and breaks the governance story. (Four such
scenario/response hybrids were retired in the 2026-08 overhaul.)
`ScenarioSpec` is the shared *override-bundle* primitive with two typed uses;
sharing the dataclass is fine, sharing the axis is not.

### The evaluation context

The tuple **(world, response)** — set once in the sidebar, rendered
everywhere. Every named view in the app is a cell of the world × response
grid:

| | response = nothing | response = package |
| --- | --- | --- |
| **world = base** | the standing outlook | "base vs what we will do" |
| **world = scenario** | stress/posture, unmitigated | the recommended plan under that world |

### Two kinds of view (the plan-of-record / outcome rule)

Every chart and table is exactly one of:

- **Plan-of-record view** — a frozen anchor. Never moves with the context.
  (The plan, financial targets, the deterministic baseline, Demand &
  Backlog.)
- **Outcome view** — follows the evaluation context and *says so in its
  title* (context suffix) or in a banner.

### Three fixed reference frames

All comparisons in the app reduce to these; no view invents its own:

1. **vs plan** — the commitment frame: "do we still make the number?"
   (Executive Overview tile deltas. The plan never moves.)
2. **vs the base outlook (base, ∅)** — what the context changes against the
   standing risk-adjusted outlook. (The strip under the tiles.)
3. **The decomposition** — `(S,P) − (B,∅) = [(S,∅) − (B,∅)] + [(S,P) − (S,∅)]`:
   what the world does to us, plus what the response recovers *in that
   world*. (The bridge chart; all deltas on common random numbers.)

Corollaries: a package is priced *conditional on a world*; a package should
look sensible under the base case, not only under the posture that flatters
it; and package EV ≠ sum of individual action EVs (actions relieving the
same constraint overlap — the interaction gap is itself reported).

### The two clocks (time windows)

Every number carries one of three windows, and its label must say which:

- **Q1** — months 1-3 of the horizon: the shipment commitment, execution's
  question ("do we ship it?").
- **Fiscal year** — months 1-12: plan attainment, margin and inventory
  targets, and the window all action EVs are measured over (long-lead
  actions' benefits partly fall beyond it by construction — present that,
  don't "fix" it).
- **Horizon** — all 18 monthly buckets: trajectory and capacity views.

The Executive Overview groups its tiles by clock (quarter / full year /
year-end position) rather than offering a quarter-vs-year toggle — the
horizons aren't symmetric (year-end inventory, annual GM targets, and
FY-window EV have no meaningful quarterly form), so a toggle would fabricate
or suppress numbers. The clock is the third labeled dimension of every
figure, handled like the other two: **context × reference frame × clock**,
separated by element and named in the label, never toggled.

## Layer map

| Layer | Contents | Code |
| --- | --- | --- |
| 0 · Data | 7 synthetic input tables, validation gate; V2 seam: synthetic/uploaded selector toward real ERP/CRM feeds | `data_generator`, `validation` |
| 1 · Plan of record | Deterministic greedy baseline allocation; the frozen anchor | `baseline_plan` |
| 2 · Evidence | Demand Confidence: 8 weighted signals + curated external intel → sim-parameter backdrop merged into every run | `market_intelligence` |
| 3 · Evaluation context | (world, response); scenarios exogenous-only; actions/claim sheets; layering via `merge_confidence_params` | `scenarios`, `config/management_actions.yaml` |
| 4 · Simulation | Correlated common-factor Monte Carlo; capacity rationing; financial translation | `correlations`, `simulation`, `operations` |
| 5 · Views | Plan-of-record vs outcome views; three reference frames; KPI/compare | `app.py`, `visualizations`, `sensitivity` |
| 6 · Decision & governance | Rules-based risks, ranked recommendations, conditioned narrative, signed Excel readout (deliberately base-anchored) | `recommendations`, `executive_report`, `exports` |

## Tab order = the SIOP process

Executive answer first, then the meeting's supporting flow:

1. **Executive Overview** — the answer (outcome page, fully conditioned)
2. **Demand & Backlog** — demand review (plan-of-record)
3. **Market Intelligence & Demand Confidence** — demand evidence
4. **Supply & Component Risk**, 5. **Manufacturing Capacity** — supply review
6. **Financial Outcomes**, 7. **Risk Drivers** — integrated reconciliation
8. **Scenario Comparison**, 9. **Management Recommendations** — decision support
10. **Assumptions & Data**, 11. **Guide, Methodology & Export** — governance & reference

## Design decisions of record

- **Scenarios carry no costs.** Enforced by test. Responses live in the
  action catalog only.
- **The Excel export stays anchored to (base, ∅).** The signed readout must
  not silently absorb whatever context was on screen; the context appears in
  it only as an explicitly labeled comparison.
- **Tiles carry one delta (vs plan).** The vs-base frame is the strip, the
  decomposition is the bridge. Dual deltas per tile were considered and
  rejected as clutter; the three frames are separated by element, not
  toggled.
- **No plan-vs-base toggle.** Both frames are always visible; toggles create
  ambiguous screenshots.
- **Recommendations price actions in a *world*, never in a context that
  already contains a package** (marginal-value-on-top-of-package is a
  possible future refinement, priced explicitly if ever added).
- **The sim mean sits below the deterministic baseline by design** (capacity
  caps demand upside asymmetrically) — the risk-adjusted outlook story, not
  a bug.
- **Demand stays unconstrained** — the demand plan feeds requested dates,
  not supply-committed dates.

## The cycle (current and next)

Evidence already loops monthly: launchd jobs run a headless Claude routine
that opens a PR updating `data/external_intel.csv` (merge = review step;
Streamlit Cloud redeploys from `main`). The plan side does not loop yet: the
engine models one cycle, forever.

**Planned next capability — closing the loop:** at sign-off, snapshot the
decision (context evaluated, package approved, expected outcomes), roll the
approved package into the next cycle's plan of record, and show
cycle-over-cycle drift (what we expected last month vs what the world did).
Pairs with the V2 real-data seam; both are about the engine living across
months rather than inside one.
