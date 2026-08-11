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
  push-out and cancellation assumptions in *every* simulation; the sidebar
  can override the assessed level for what-ifs, default "Assessed") plus an
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
| **world = scenario** | stress/posture, unmitigated | the recommended response under that world |

### Two kinds of view (the plan-of-record / outcome rule)

Every chart and table is exactly one of:

- **Plan-of-record view** — a frozen anchor. Never moves with the context.
  (The plan, financial targets, the baseline supply plan, Demand &
  Backlog.)
- **Outcome view** — follows the evaluation context and *says so in its
  title* (context suffix) or in a banner.

Page template for tabs that mix the two: **outcome views lead the page**
(the decision question first), then an explicit divider and caption mark
the switch to plan-of-record/input reference below — never interleaved.

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

## Vocabulary of record

The same words, used the same way, everywhere — labels, captions,
narrative, docs. The executive-facing rendering of this table lives in
`docs/mental-model.html` (§4, embedded in the Guide tab); keep the two in
sync.

| Term | Means | Is not |
| --- | --- | --- |
| **Forecast** | The estimate of demand not yet booked — the uncertain, forward-looking slice of the demand plan. | Booked orders (standard S&OP: orders *consume* forecast; inside the demand time fence only orders count as demand). |
| **Backlog** | Booked orders not yet shipped/recognized — actual demand ("firm" in the booking sense). | A certainty: push-outs and cancellations still touch it. Nor the **past-due backlog**, which always carries its qualifier: in the real order book, past-due is the slice of backlog behind schedule; in the engine it is an outcome — simulated demand (forecast included) not yet served by its requested month. |
| **Demand plan** | The unconstrained demand statement: forecast + backlog, customer × family × month, at requested dates. | A supply commitment; confidence never rewrites it. |
| **Plan (of record)** | The frozen revenue commitment for the cycle, derived from the demand plan by the baseline supply plan — the yardstick behind every "vs plan" delta and P(plan). Industry mapping: the revenue line of the AOP ("one set of numbers"), and the internal number behind quarterly street guidance. | A forecast; it never moves inside a cycle, and it never means a response package. Nor guidance itself — guidance is the external, usually more conservative derivative; anchor on the internal commitment or P(plan) quietly becomes P(guidance). |
| **Outlook** | The simulated outcome distribution for a context — what the demand plan, pushed through supply and calibrated by evidence, is expected to produce. Always named by its context: the **standing base outlook** (base, ∅), a **scenario outlook** (unmitigated), the **conditioned outlook** (world + response). In graphics: the solid center line and band; the demand plan is never drawn in outcome space. | The plan — the outlook moves with the context, the plan never does; P(plan) is the measured gap between them. Nor "the decided version": decidedness is a sign-off event, not a context — whichever cell the meeting commits (package or none) becomes the **decision of record**. |
| **Baseline supply plan** ("the baseline") | The feasible allocation of the demand plan across sites and months — respecting components, capacity and integration — whose revenue becomes the plan of record. "Deterministic" describes how it is computed (zero shocks), not what it is. | The base case: the base *outlook* is a simulation, the baseline is an allocation — frozen mechanics, reference only. |
| **Targets** | Annual financial goals (gross margin, inventory) — the margin and inventory lines of the AOP. | The plan. Not a synonym for "AOP": the AOP's revenue line is the plan of record, so naming only these AOP would split the term. |
| **Scenario** (world) | Exogenous hypothesis about what happens *to* the business. | A decision; carries no cost. |
| **Demand Confidence** (world) | Assessed evidence calibrating demand variance, push-out and cancellation odds around the demand plan. The sidebar shorthand "trust in the forecast" names the least certain slice; the mechanics act on the whole demand plan. | A hypothesis or a toggle. |
| **Response** (package) | Costed, latency-ramped actions — what the business chooses to *do*. Each action carries a **decision cost**; package costs sum and are charged to Q1 operating income, so every EV is quoted net of cost. | Part of any world; the only axis where money is spent. |
| **Decision of record** | The committed outcome of the cycle: the (world, response) context the meeting signs — package or none — with its conditioned outlook and cost. Today: the export's Decision of Record sheet. The planned sign-off loop makes it a first-class state and rolls it into the next cycle's **plan of record**. | An outlook — outlooks are candidates; this term is reserved for the signed one. |

## Layer map

| Layer | Contents | Code |
| --- | --- | --- |
| 0 · Data | 7 synthetic input tables, validation gate; V2 seam: synthetic/uploaded selector toward real ERP/CRM feeds | `data_generator`, `validation` |
| 1 · Plan of record | Baseline supply plan (deterministic greedy allocation); the frozen anchor | `baseline_plan` |
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
- **Tile cards share one column grid; blanks are deliberate.** Columns are
  concepts (level, probability, margin, margin probability, downside), and a
  slot stays empty when the measure has no meaning on that clock — a blank
  beats a manufactured number (e.g. there is no P(Q1 margin target): the
  margin target is annual, and a quarterly probability against it would
  invent a commitment nobody made). Revenue-at-risk lives on the plan cards
  (quarter / full year), not the position card — it's a plan-attainment
  measure.
- **Working capital is a proxy, not a headline.** Its level is dominated by
  fixed DSO/DPO assumptions layered on inventory (no receivables aging, no
  payment terms, no balance-sheet FX), so it gets no overview tile; its
  *deltas* are driven by modeled inventory and remain in the action and
  scenario tables, and the narrative labels it a proxy. Revisit if real AR/AP
  feeds arrive in V2.
- **No plan-vs-base toggle.** Both frames are always visible; toggles create
  ambiguous screenshots.
- **Recommendations price actions in a *world*, never in a context that
  already contains a package** (marginal-value-on-top-of-package is a
  possible future refinement, priced explicitly if ever added). The world is
  the sidebar context's world, automatically — the page-local
  evaluation-world toggle was removed (2026-08-10) as a second source of
  truth; under a scenario the base-world EV stays visible as a comparison
  column instead. Both frames at once, never toggled.
- **Demand Confidence is calibration, not hypothesis** (2026-08-10). A
  scenario is a chosen hypothesis about how the world could differ; the
  confidence assessment is *evidence* — an estimate of how uncertain today's
  world already is, in the same category as the historical push-out and
  cancellation rates it scales. It therefore applies to every simulation by
  default and gets **no on/off toggle**: an off-switch for the uncertainty
  estimate is the switch people reach for exactly when confidence is low.
  The no-adjustment counterfactual stays visible instead — the Moderate row
  of the sensitivity table *is* the unadjusted world, and is labeled so.
  A sidebar **override** (default "Assessed") allows what-ifs at other
  levels; it belongs to the world axis and is labeled as an override
  everywhere it differs from the assessment (sidebar badge, header caption,
  Market Intelligence banner). Confidence never rewrites the demand plan —
  it reaches the simulation only through three knobs (`demand_sigma_mult`,
  `pushout_prob_add`, `cancel_prob_mult`), and external-intel impacts act
  one step further upstream: they move the 0-100 *score*, reaching the
  knobs only when they tip the score across a level threshold, at which
  point all three change together.
- **No third simulation count.** Headline results run at the sidebar mode's
  path count; action pricing runs at `min(n_sims, 2000)`; nothing else may
  introduce its own count. The confidence sensitivity table runs at the
  headline count with the same seeds (common random numbers), so the
  applied level's row reconciles exactly with the standing base outlook —
  a table that disagrees with the Overview for the same world reads as a
  bug, not a nuance. (Its previous hardcoded 1,500 paths were retired
  2026-08-10.)
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
**decision of record** (context evaluated, package approved, expected
outcomes — the term is already reserved in the vocabulary table), roll the
approved package into the next cycle's plan of record, and show
cycle-over-cycle drift (what we expected last month vs what the world did).
Pairs with the V2 real-data seam; both are about the engine living across
months rather than inside one.
