# Design: customer dimension and allocation policy in the Monte Carlo

Status: proposal for review (2026-09). Spike code: `spikes/customer_dimension/`.
Nothing in `src/` changes until this design is agreed.

## 1. Summary

- **Why.** The Monte Carlo (`src/simulation.py::run_simulation`) collapses
  demand to family x month and rations scarce supply pro-rata, so it cannot
  answer decision D6 ("who gets shorted, and what does it cost?") or show
  displacement for D7 (upside) and D9 (push-out). The deterministic baseline
  does keep customers, so today the two engines answer different questions.
- **What.** Carry a **demand line** (customer x family) through every path,
  split each line-month into a **firm** slice (backlog, timing risk only) and a
  **forecast** slice (volume and timing risk). Ration components and EMS
  capacity to slices under a selectable **allocation policy**, recognize
  revenue per line at customer ASP.
- **Cost.** Spike kernel, 5,000 paths, float32: **0.4 s with today's 15 lines**
  and **1.3 s at 200 lines / 109 customers** with tiered-priority allocation,
  175 MB peak. This fits the live recompute budget of about 3 s once the
  engine moves to float32 (section 5).
- **It changes the answer.** On the synthetic data, the policy choice moves
  expected FY gross margin by up to $15M and, more importantly, moves who is
  shorted: the top customer's P(FY fill below 95%) goes from 32% under
  proportional to 8% under strict priority, while two smaller customers go
  from under 25% to 42% and 59% (section 6).
- **Side finding, bigger than the policy lever.** The EMS water-fill in
  `run_simulation` walks sites in index order, so flexible sites are consumed
  before dedicated ones and capacity is stranded. Walking least-contested
  sites first raises mean FY revenue in the **current engine** by about $61M
  (2.4%) at 5,000 paths. This should be fixed first, as its own change.
- **Recommendation.** Express every policy as ordered **tiers** with pro-rata
  inside a tier. That form is vectorized, its cost depends on the number of
  tiers x families rather than on the number of customers, and it lets the
  deterministic baseline become the same engine run with zero shocks.

## 2. Design

### 2.1 Unit of analysis

| Concept | Definition | Count today | Real business (assumed) |
|---|---|---:|---:|
| Customer | Account with priority, group, ASP terms | 8 | 50 to 100 |
| Demand line | Customer x product family | 15 | 60 to 200 |
| Slice | Line x {firm, forecast} | 30 | 120 to 400 |

Order-line or serial-level detail is not needed (decision catalog, D6).
Per path, arrays are laid out **features first, simulations last**
(`(lines, n_sims)`); reductions over small axes then become row adds over
contiguous vectors, which was about 2x faster in the spike.

### 2.2 Demand shocks per line

Forecast units for line l (customer c, family f, market k), month m:

```
fcst[l,m] = plan_fcst[l,m] x (1 - cancel_p[l])
            x Market_k(m)        AR(1) factor path, existing FactorEngine
            x Family_f           persistent lognormal, small (spike: 0.05)
            x Customer_c         persistent lognormal, sigma = customer_idiosyncratic_sigma
            x Noise_l(m)         month-to-month order lumpiness (spike: 0.15)
```

- **Customer correlation choice.** `Customer_c = sqrt(rho) x Group_g +
  sqrt(1 - rho) x own_c`, with groups from `customer_group` (for example the
  two memory manufacturers). Options: independent (rho = 0), group-correlated
  (default 0.5), or fully shared. Finding: at today's sigmas the choice does
  not move top-3 revenue at risk (P10 of $1,568M to $1,570M across rho = 0,
  0.5, 0.9) because market variance dominates. Keep one group factor with a
  fixed rho, and do not expose it as a knob until calibration shows a customer
  sigma that matters.
- **Naming fix.** `customer_idiosyncratic_sigma` is applied per family in
  `run_simulation` today. In this design it becomes what its name says.

### 2.3 Firm backlog vs forecast

| | Firm backlog | Forecast |
|---|---|---|
| Volume shock | None (the order exists) | Market x family x customer x noise |
| Cancellation | Half the line's forecast rate (assumption, to calibrate) | Line rate |
| Push-out | Yes, whole order | Yes, whole order |
| Allocation | First, in every policy except proportional | After firm |

Firm backlog is the low-variance part of near-term revenue. Treating it like
forecast (as today) overstates near-quarter revenue risk.

### 2.4 Push-outs and cancellations at customer level

- **Push-out:** Bernoulli per line-month with the line's `push_out_prob`. The
  whole line-month quantity moves 1 month (70%) or 2 months (30%) and arrives
  as firm. One uniform draw decides both (u < p pushes; u < 0.3p slips two).
  Whole-order events matter at customer level: they create the lumpy quarters
  that drive P(shortfall) by customer. The family-level engine today moves an
  expected fraction, which is adequate for totals but not for customers.
- **Cancellation:** expected-value haircut per line (cheap; a Bernoulli event
  adds variance only where a single order is large relative to the quarter).
- **Unmet demand** carries to the next month as firm, keeps its line, and
  ages (late unit-months). Open question 1 asks whether some of it is lost.

### 2.5 Rationing to demand lines

Resources, each month and each path:

- **Critical components:** cumulative (unused supply carries forward), usage
  per system by family.
- **EMS sites:** monthly std-equivalent capacity, qualification by family.
  Final integration happens at the EMS in the target business, so the
  separate integration pool becomes an optional company-level cap (kept in the
  spike for reconciliation with today's data, off when integration collapses
  into EMS).

Two exact allocation primitives, both vectorized across paths:

1. **Pro-rata pass** over a set of slices. Pro-rata inside a family is
   identical at line and family level, so the heavy work (component scale per
   family, EMS water-fill across sites, least-contested site first) runs on
   `(families, n_sims)` arrays and each line gets its family's fill ratio.
   Cost does not grow with the number of lines, apart from one gather and one
   scatter.
2. **Sequential greedy** over slices in a fixed order: each slice takes the
   minimum of its want, its components' residual / usage, and its qualified
   EMS residual, then consumes. Exact replica of the baseline heuristic, but
   cost is linear in the number of slices.

### 2.6 Revenue recognition and ASP by customer

Revenue per line = shipped units x line ASP x ASP shock by customer, shifted
by the family's recognition lag and acceptance slip. Customer ASP is already
in `data.demand` (`asp_usd` varies by customer, up to about 12% within a
family). Cost stays by family (standard COGS), so GM by customer = revenue by
customer less family standard cost of units shipped. Acceptance slip can later
move to customer level using `site_readiness_prob`, which is already a
customer field.

## 3. Allocation policy lever

The allocation rule is a **decision**, so it lives on the Response axis
(`docs/ARCHITECTURE.md`) as a management action, not in the world. Its label
must state that it moves revenue between customers and months, and changes
margin only through mix.

| Policy | Rule | Implementation | When it is the right choice |
|---|---|---|---|
| Proportional | Everyone gets the same fill ratio within a family | 1 pro-rata pass | Neutral reference; today's engine |
| Strict priority | Firm first, then customer priority 1, 2, 3 | Tiers: 2 x priorities = 6 passes | Written priority list exists |
| Margin-weighted | Firm first, then contribution per constrained unit | Tiers: firm x margin bands (4 bands = 8 passes), or greedy by line | Maximize GM when contracts allow |
| Protect top-N | Top-N customers (all slices) first, then rest firm, rest forecast | 3 passes | Concentration defense, key accounts |
| Baseline replica | Firm, priority, contribution (sequential) | Greedy over slices | Reconciliation and zero-shock plan |

### 3.1 Vectorizing priority

- **Tier by tier with residual capacity.** Sort slices into ordered tiers once
  (static per policy), permute so each tier is a contiguous block, then run
  the pro-rata pass per tier against what earlier tiers left. Cost is
  proportional to the number of tiers, not customers. Lines in the same tier
  and family always get the same fill ratio, which is also the fairest
  statement of a tier rule.
- **Cumulative-capacity trick.** With one binding resource, strict order is a
  single `cumsum`: `alloc_i = clip(cap - cumsum_before_i, 0, want_i)`. It is
  exact and very fast, but not exact with components and EMS binding at once,
  which is the normal case here, so tiers are the general form.
- **Greedy over slices.** Exact for any order, but a Python loop over slices
  (about 12 small array operations each). Fine up to about 60 lines; too slow
  at 200 lines for live use.
- **Contribution per constrained unit.** The spike ranks by contribution per
  EMS std-equivalent unit (the baseline's tiebreak). Ranking by the resource
  that actually binds would need a shadow price per path. Recommendation:
  compute the binding resource once from the zero-shock run and rank on it;
  per-path ranking is not worth its cost unless calibration shows the binding
  resource changes often across paths.

### 3.2 Fidelity vs speed

| Form | Exactness | Cost driver | 5k paths, 200 lines, float32 |
|---|---|---|---:|
| Proportional | Exact for pro-rata | families | 0.99 s total, 0.15 s allocation |
| Tiered (6 tiers) | Exact for tier rule, pro-rata inside tier | tiers x families | 1.27 s total, 0.36 s allocation |
| Greedy by slice | Exact for any strict order | slices | 2.25 s total, 1.32 s allocation |

Within-tier pro-rata vs within-tier strict order changed expected FY revenue
by $5M (priority_tiered $2,579M vs strict greedy $2,584M), about 0.2%. That
is below the level at which a decision changes, so tiers are the production
form and greedy is kept for reconciliation and tests.

## 4. Prototype

`spikes/customer_dimension/`:

- `problem.py`: builds the 15-line problem from `generate_all` (temp dir) and
  `build_planning_arrays`; `scale_lines` splits customers into Dirichlet-share
  sub-customers (60 lines / 32 customers, 200 lines / 109 customers) with
  unchanged totals, so constraint tightness is unchanged.
- `kernel.py`: the Monte Carlo kernel (demand shocks, supply shocks,
  allocation, revenue) with all six policies; `dtype` option.
- `bench.py`: timings (best of 2) and peak memory (tracemalloc).
- `metrics.py`: policy comparison with common random numbers, customer and
  concentration tables, correlation sensitivity.

Supply shocks in the kernel are simplified stand-ins for the full engine's
(delay pipelines, expediting, adherence feedback and yield are omitted), so
the absolute dollars below are illustrative. Policy deltas are paired on the
same random numbers.

**Zero-shock reconciliation.** With shocks off, the strict greedy policy
ships 1,909 units vs 1,895 in `run_baseline` (0.7%; the baseline rounds down
to whole systems and also sorts on requested month). Family totals match exactly
for Zenith and Vector, within 3 units for Horizon, and are 4 and 7 units
higher for Atlas and Nexus.

## 5. Performance results

Machine: 4-core x86_64, Python 3.11, numpy 2.4. "Total" is the whole kernel
run (shocks, allocation, revenue); "alloc" is the allocation share.

| Lines (customers) | Paths | Policy | float64 total s | float32 total s | float32 alloc s | float32 peak MB |
|---|---:|---|---:|---:|---:|---:|
| 15 (8) | 1,000 | proportional | 0.07 | 0.04 | 0.02 | 5 |
| 15 (8) | 1,000 | priority tiered | 0.18 | 0.15 | 0.12 | 5 |
| 15 (8) | 1,000 | strict greedy | 0.08 | 0.06 | 0.03 | 5 |
| 15 (8) | 5,000 | proportional | 0.26 | 0.18 | 0.06 | 25 |
| 15 (8) | 5,000 | priority tiered | 0.60 | 0.41 | 0.29 | 25 |
| 15 (8) | 5,000 | strict greedy | 0.40 | 0.23 | 0.10 | 23 |
| 15 (8) | 10,000 | proportional | 0.60 | 0.35 | 0.13 | 50 |
| 15 (8) | 10,000 | priority tiered | 1.32 | 0.82 | 0.58 | 50 |
| 15 (8) | 10,000 | strict greedy | 0.81 | 0.46 | 0.22 | 46 |
| 60 (32) | 5,000 | proportional | 0.67 | 0.42 | 0.08 | 62 |
| 60 (32) | 5,000 | priority tiered | 1.05 | 0.59 | 0.29 | 60 |
| 60 (32) | 5,000 | strict greedy | 1.48 | 0.71 | 0.41 | 58 |
| 60 (32) | 10,000 | priority tiered | 2.33 | 1.15 | 0.58 | 120 |
| 60 (32) | 10,000 | strict greedy | 3.08 | 1.47 | 0.89 | 117 |
| 200 (109) | 1,000 | priority tiered | 0.43 | 0.31 | 0.12 | 34 |
| 200 (109) | 5,000 | proportional | 2.21 | 0.99 | 0.15 | 175 |
| 200 (109) | 5,000 | priority tiered | 2.40 | 1.27 | 0.36 | 169 |
| 200 (109) | 5,000 | strict greedy | 5.00 | 2.25 | 1.32 | 169 |
| 200 (109) | 10,000 | proportional | 4.99 | 2.24 | 0.33 | 351 |
| 200 (109) | 10,000 | priority tiered | 5.56 | 2.50 | 0.70 | 338 |
| 200 (109) | 10,000 | strict greedy | 10.75 | 4.73 | 2.95 | 338 |

Full output: `python spikes/customer_dimension/bench.py`.

Reading the table:

- Allocation is not the bottleneck at scale; per-line demand draws and
  accounting are. They scale with lines x paths; allocation under tiers does
  not.
- **float32 halves memory and cuts time by 15 to 55%** (more at larger sizes). Revenue changed by less
  than 0.03% (different random streams, not precision).
- **Budget.** Today's engine runs 5,000 paths in 2.1 s. The kernel replaces
  the demand and allocation sections, so the integrated cost at 15 lines
  should be about 2.1 s plus 0.2 to 0.4 s. At 200 lines, float32 throughout
  the engine is needed to stay near 3 s. This is an estimate; the integrated
  timing is a migration gate (step 9).
- **Beyond 200 lines,** fold tail customers (for example under 1% of revenue)
  into one "other" line per (tier, family). Under tiered pro-rata this is
  exact for the allocation, and it keeps line count bounded.

## 6. Metrics this enables

Definitions (per customer, FY = months 1 to 12):

| Metric | Definition | Answers |
|---|---|---|
| Revenue at risk (RaR) | Plan FY revenue minus P10 FY revenue | How exposed is each account, demand and supply combined |
| Supply gap | E[(FY demand units - FY shipped units)+ x ASP] | How much of the exposure is ours to fix |
| P(shortfall) | P(FY fill rate < 95%) | Service risk by account |
| Fill and delay | Mean FY fill; late unit-months per demand unit | Service level by account |
| Top-N concentration | Share, P10 and RaR of top 1, 3, 5 customers | Concentration exposure |
| Cost of who gets shorted | Per policy: E and P10 FY revenue and GM, delta vs proportional (paired), top-3 P(shortfall), accounts with P(shortfall) > 25% | Which rule to adopt |

**Cost of who gets shorted, base world** (5,000 paths, 15 lines, spike shocks):

| Policy | E[FY rev] $M | E[FY GM] $M | dGM vs proportional $M (P10, P90) | Top-3 P(short) | Accounts with P(short) > 25% |
|---|---:|---:|---:|---:|---:|
| Proportional | 2,558 | 1,511 | 0 | 32% | 2 of 8 |
| Priority tiered | 2,579 | 1,523 | +11.7 (0.0, +27.8) | 20% | 5 of 8 |
| Strict greedy | 2,584 | 1,526 | +14.9 (-0.6, +36.1) | 19% | 4 of 8 |
| Margin greedy | 2,579 | 1,523 | +12.3 (-4.8, +35.0) | 20% | 5 of 8 |
| Protect top-3 | 2,575 | 1,520 | +9.3 (-1.8, +25.2) | 4% | 5 of 8 |

**Who pays, base world:**

| Customer (priority, plan share) | Proportional P(short) | Strict priority P(short) | Supply gap $M, proportional to strict |
|---|---:|---:|---:|
| Titan Semiconductor (1, 27%) | 32% | 8% | 37.9 to 12.1 |
| Kestrel Compute (1, 18%) | 45% | 27% | 32.3 to 21.2 |
| Meridian Micro Devices (3, 10%) | 21% | 59% | 9.7 to 19.7 |
| Steinfeld Automotive Semi (2, 7%) | 2% | 42% | 1.1 to 9.6 |
| Silverpine Mobile (2, 5%) | 17% | 47% | 3.8 to 8.8 |

In a stressed world (the tightest component loses 25% of receipts from month
3), protect-top-3 is the best policy on both counts: GM +$14.2M vs
proportional and top-3 P(short) 15% vs 50%. Strict priority pushes Daehan
Memory (priority 2) to 85% fill and 4.7 months average delay. That is the
kind of consequence the SIOP meeting needs to see before signing a rule.

Two readings for management: first, most customer RaR comes from demand, not
supply (Titan: $182M RaR, $12M supply gap), so the allocation lever acts on
the smaller part. Second, the GM difference between rules is modest; the
redistribution of service between accounts is large. Both are only visible
with this dimension.

Full tables: `python spikes/customer_dimension/metrics.py`.

## 7. Migration from the current engine

Each step is a separate, reviewable change with a regression test.

0. **Fix the EMS water-fill site order** in `run_simulation` (least-contested,
   then cheapest, as the baseline already does). Measured on the current
   engine at 5,000 paths: mean FY revenue $2,595M to $2,657M (seeds 42 and 7
   agree within $3M). Independent of everything below; needs its own sign-off
   because it moves every reported number.
1. **Data contract.** Confirm the line grain: customer x family x month with
   firm and forecast split, ASP, priority, group, push-out and cancel
   probability (all present in `data.demand` today; see
   `docs/data-contract.md`).
2. **PlanningArrays gains line arrays** (`line_cust`, `line_fam`, firm,
   forecast, ASP, probabilities). Family arrays are derived by summing lines,
   so existing consumers keep working.
3. **Line-level demand step** in `run_simulation` (section 2.2 to 2.4),
   replacing the family loop and the expected-fraction push-outs.
4. **Replace the monthly shipment loop with the tiered kernel.** Default
   policy proportional. Regression: family totals equal today's engine within
   Monte Carlo noise after step 0.
5. **Revenue by line** at customer ASP; `SimulationResult` gains per-customer
   FY and quarterly revenue, GM, fill, and late unit-months as `(n, customers)`
   arrays (not per line per month, to bound memory).
6. **Allocation policy as a Response action** in
   `config/management_actions.yaml`, with an explicit label ("moves revenue
   between customers and months").
7. **One engine.** The deterministic baseline becomes the kernel with zero
   shocks, n = 1, and the baseline-replica greedy order. `run_baseline`'s row
   loop is retired once zero-shock results match it within rounding (today
   0.7%, explained by whole-system rounding and the requested-month key).
   After that, the baseline and the Monte Carlo cannot disagree on allocation.
8. **Views and exports:** customer table, concentration panel, cost-of-who-
   gets-shorted table (replacing plan-only `Customer Groups` export).
9. **Performance gate:** float32 per-path arrays, sims-last layout, and a test
   that fails if 5,000 paths exceed 3 s on the reference machine.

## 8. Risks and open questions

1. **Does a shorted order wait or leave?** The spike assumes all unmet
   demand waits, so shorting costs revenue timing, not revenue. If a share is
   lost (a customer buys from a competitor or cancels), the cost of shorting
   is much higher and customer-specific. This is the single assumption most
   likely to change the answer; it needs a lost-sale fraction per customer or
   group from Sales.
2. **Real allocation rules.** Is there a written priority rule, contractual
   allocation, or late-delivery penalties? Penalties would enter as a cost per
   late unit-month by customer, which the metrics already carry.
3. **Firm cancellation and push-out rates.** Half-rate cancellation for firm
   backlog is an assumption; calibrate from order history.
4. **Static margin ranking** may misrank when the binding resource changes
   across paths. Mitigation in section 3.1; test on real data.
5. **Customer sigma calibration.** At 0.10 customer volatility is dominated
   by market volatility. If history shows larger account-level swings, revisit
   the correlation option.
6. **Memory at scale.** 175 MB per 5,000-path run at 200 lines in float32.
   Fine for one user; for concurrent Streamlit sessions, keep outputs at
   customer x quarter and fold tail customers.
7. **Integration collapse into EMS.** The kernel treats integration as an
   optional cap; confirm whether any in-house integration or test capacity
   remains in the target business.
8. **Spike shocks are simplified.** Absolute dollars in sections 5 and 6 are
   illustrative; policy deltas and rankings are the point.
9. **Governance.** Showing per-customer shortfall invites manual overrides
   by account. The pattern of record stands: the rule is a Response the
   meeting approves, not an edit to the demand plan.
