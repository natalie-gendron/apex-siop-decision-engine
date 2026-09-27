# Engine reconciliation: baseline plan vs Monte Carlo simulation

Status: **implemented** (build step 1, 2026-09). Evidence for the analysis
below: `spikes/engine_reconciliation/` (seed 42, data generated to a temp dir),
run against the pre-step-1 engine (commit f7f2e44); the spike no longer runs
against current `src/`.

## Outcome of step 1

| Proposal | Decision | Where |
|---|---|---|
| 1. Shocks object | Built. `Shocks` scales ten named shock groups; `Shocks.zero()` switches all off. Draws are taken regardless of amplitude, so common random numbers survive | `src/shocks.py` |
| 2. Baseline = engine | Built. `run_baseline` wraps `run_simulation(shocks=Shocks.zero(), n_sims=1)`; the priority queue and its cost formulas are deleted. Allocation stays proportional within a family until the customer dimension lands (step 3) | `src/baseline_plan.py` |
| 3. Utilization penalty | Removed, not rebased. Driven by planned load capped at 100%, it becomes a constant derate of about 10% at every binding site, which is a calibration of the adherence input, not a mechanism | `src/simulation.py` |
| 4. Receipt delay | Structural delay is now a shock (`receipt_delay`), zero in the zero-shock run | `src/simulation.py` |
| 5. One policy each | Safety stock: 30% not usable, all physical stock valued. Unit cost: `operations.standard_unit_cost` for COGS and FG, rework from FPY. Damping kept in the one engine until step 4. Integer floors gone | `src/operations.py`, `src/simulation.py` |
| 6. Constraint log | Each unit logged once, in the month it first misses its requested date, against the ceiling that cut it. Installation capacity removed | `src/baseline_plan.py` |
| Site-fill order | Least-contested site first, then cheapest | `src/simulation.py` |
| Adherence | Each site uses its own scheduled adherence | `src/simulation.py` |

Guard: `tests/test_engine_reconciliation.py` (no spread at zero shocks; exact
equality with the baseline on units, revenue, COGS, inventory and cash; each
late unit logged once).

Effect, base world (Moderate confidence, 5,000 paths, seed 42), each change
added in turn from the old engine:

| Change | FY revenue $M | P(FY plan) | End inventory $M | E&O $M | Overtime EV $M | Reserve EMS EV $M |
|---|---:|---:|---:|---:|---:|---:|
| Old engine | 2,595.4 | 62.3% | 519.1 | 42.7 | +60.2 | +35.5 |
| Stock valued incl. unusable safety stock | 2,595.4 | 62.3% | 530.0 | 47.4 | +60.2 | +35.5 |
| Per-site adherence | 2,600.5 | 63.5% | 529.2 | 46.9 | +58.9 | +34.8 |
| Site-fill order | 2,658.4 | 74.8% | 518.8 | 42.0 | +39.5 | +22.2 |
| Penalty removed (= new engine) | 2,757.5 | 84.3% | 493.7 | 33.9 | -1.4 | -1.5 |

The baseline plan of record moves from $2,842.3M to $2,824.7M FY revenue
(-0.6%): the site-fill order recovers most of what proportional rationing
loses against the old priority greedy.

## Executive summary

1. At zero shocks the simulation lands $200.7M (7.1%) below the baseline on FY revenue ($2,641.6M vs $2,842.3M), 56.9 units short, $120.7M lower gross profit and $69.1M lower ending inventory. The docstring claim that it reconciles (`src/simulation.py` L9-10) is false.
2. Two differences explain all of the revenue gap: the utilization adherence penalty (+$114.7M, 45 units) and proportional rationing vs greedy allocation (+$82.5M, 11 units); per-site adherence adds $3.5M. With every difference switched off, the spike copy of the sim reproduces the baseline exactly (0.0% residual).
3. Ending inventory is a separate story: the sim's purchasing-response damping alone is worth $86M, and it decides whether the $600M inventory target looks met by a wide margin ($524M) or barely ($593M).
4. Engine mismatch, not uncertainty, drives most of the reported plan risk: P(FY plan) is 63% as-is, 81% without the penalty, and about 95% if the zero-shock gap is removed as a level shift. The penalty also inflates the value of EMS capacity levers about 2.3x.
5. Recommendation: one engine. The baseline becomes `run(shocks=ZERO, n=1)`, allocation becomes a vectorized customer-priority greedy (possible because the sort order is fixed across draws), and a zero-shock reconciliation test guards it.

## Method

- Zero-shock world (`harness.py`): all `config.uncertainty` sigmas and `market_demand_sigma` set to 0, EMS `regional_disruption_prob_monthly` 0, and sim params `cancel_prob_mult=0`, `pushout_prob_add=-1`, `pullin_prob_add=-1`, `comp_disrupt_mult=0`, `acceptance_delay_add=-1`. `utilization_adherence_penalty` and `site_disruption_impact` are not sigmas and stay at config values.
- `sim_variant.py` is a flag-switchable copy of `run_simulation`; `greedy.py` is a parametrized copy of the baseline allocator and financial translation. `check_equivalence.py` proves both copies match `src/` bit-for-bit (full uncertainty and zero shock) before any flag is flipped.
- `run_reconciliation.py` switches each difference off alone, cumulatively (table order), and "last out" (everything else off). Cumulative contributions sum exactly to the gap. `pplan_impact.py`, `customer_allocation.py` and `other_mechanisms.py` cover P(plan), who gets shorted, and mechanisms that are zero in this world.

## Differences and their contribution (zero-shock world, FY = months 1-12)

Cumulative contribution when switched off in table order, moving the sim toward the baseline. "Alone" gives the revenue effect of switching it off by itself from the sim as-is. GP and inventory in $M.

| # | Difference | Where | FY rev $M | Alone $M | FY units | FY GP $M | End-FY inv $M | Class |
|---|---|---|---|---|---|---|---|---|
| 1 | Utilization adherence penalty: adherence cut when unconstrained demand / capacity > 0.92; the ratio is unbounded (month 7: 1.31, adherence falls to the 0.60 floor) | sim L263-272 | +114.7 | +114.7 | +44.6 | +65.8 | -26.6 | Intentional mechanism, accidental driver, missing from baseline |
| 2 | Tightness, logistics and EMS factor shocks are not scaled by any sigma, so they remain at "zero sigma" (delay fraction, material +0.8%, freight +3.2%, FPY) | sim L157-158, L175-181, L231, L285, L400-405 | 0.0 | 0.0 | 0.0 | +6.5 | +1.6 | Accidental |
| 3 | Base receipt delay (LT-8)/100, clipped 1-25%, shifts receipts one month even with no shock | sim L174 | 0.0 | 0.0 | 0.0 | 0.0 | +6.8 | Accidental (contradicts PO facts as inputs) |
| 4 | Safety stock: sim holds 30% as a hard floor; baseline uses none (its docstring says "above safety stock") | sim L207, L223; baseline L9, L55 | 0.0 | 0.0 | 0.0 | 0.0 | +10.5 | Inconsistent on both sides |
| 5 | Adherence: sim applies the unweighted site-mean to every site; baseline uses each site's own | sim L267; `operations.py` L171 | +3.5 | +3.3 | +1.5 | +2.0 | -0.2 | Accidental |
| 6 | Allocation: proportional family-level rationing and EMS water-filling vs sequential customer-priority greedy with least-contested site routing | sim L305-355; baseline L67-151 | +82.5 | +65.3 | +10.8 | +46.1 | -9.3 | Documented approximation, material |
| 7 | FPY utilization penalty (cost only) | sim L283-285 | 0.0 | 0.0 | 0.0 | +1.2 | 0.0 | Accidental |
| 8 | COGS: sim rework on shipped units at fleet-mean conversion cost plus expedite cost; baseline rework_prob per family on recognized units | sim L415-439; baseline L170-175 | 0.0 | 0.0 | 0.0 | -0.9 | 0.0 | Accidental (two cost formulas) |
| 9 | Raw-material purchasing-response damping (half of stock above 2 months' need is not valued) | sim L450-454 | 0.0 | 0.0 | 0.0 | 0.0 | +86.0 | Intentional in sim, missing from baseline |
| 10 | RM valuation omits the safety-floor stock that is still on the shelf | sim L449 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 (+10.5 alone) | Accidental |
| 11 | FG valued without rework and scrap | sim L460 vs baseline L190-193 | 0.0 | 0.0 | 0.0 | 0.0 | +0.2 | Accidental |
| | **Total (sim as-is to baseline)** | | **+200.7** | | **+56.9** | **+120.7** | **+69.1** | Residual 0.0 |

Notes on the table:
- Rows 3 and 4 have no revenue effect only because critical components never bind in this dataset at zero shocks; they will bind in tighter data. Row 4's inventory effect is entirely the valuation bug in row 10 (the two are the same $10.5M).
- GM moves by less than 0.3 pts in every row (sim 58.4%, baseline 58.5%); the GP gap is volume. Rows 2 and 8 are each about +0.25 pts alone.
- Row 1, "last out" (everything else already off): +$131.4M. Capping the penalty's driver at 100% load (load cannot exceed capacity) recovers $67.4M of the $114.7M with the mechanism kept.
- Row 6 splits into about $42M volume and recognition timing and $41M family mix: when integration binds (units, not std-units), the greedy fills Zenith ($3.4M ASP) completely while proportional rationing cuts every family pro rata. Running the greedy with all customer priorities equal gives $2,845.2M, within $3M of the baseline: the gain comes from sequential routing and margin ordering, not from customer priority itself.

Baseline-only artifacts (not in the table because they sit inside the baseline):

| Item | Where | Effect |
|---|---|---|
| Integer `floor()` on component, integration and site ceilings | baseline L99, L103, L119 | -$10.0M FY revenue, -5.1 units |
| Constraint log mislabels: 53 of 183 FY units logged as "integration" were physically stopped by EMS capacity; carried shortfall is re-logged every month (252 units logged vs 42 true FY shortfall) | baseline L133-151 | Misleading "integration binds" narrative |
| Installation capacity (104/mo, below integration 114/mo) is decremented but never enforced; neither engine enforces it | baseline L57, L129 | Would cost -$61.2M FY revenue if real |

## Which constraint binds

FY revenue response to +30% capacity (zero shock, $M):

| Engine | Integration x1.3 | EMS x1.3 | Both x1.3 |
|---|---|---|---|
| Baseline (src) | +6.9 | +30.9 | +71.5 |
| Sim as-is | +8.1 | +189.2 | n/a |
| Sim, penalty off (full uncertainty) | +6.8 | +71.9 | n/a |
| Sim as-is (full uncertainty) | +2.7 | +168.7 | n/a |

The baseline is co-bound by EMS and integration, EMS first; the "integration is the binding constraint" story is a logging artifact (see above). The sim looks strongly EMS-bound because extra EMS capacity also lowers the demand/capacity ratio that drives the penalty, so an EMS lever is paid twice. Integration x1.3 is weak in both engines, not only in the sim.

## Mechanisms that are zero in this world but still separate the engines

- Expected-value events (baseline has none). Turned back on one at a time with sigmas still 0: cancellations -$27.9M, pull-ins +$33.3M, push-outs +$5.9M, acceptance slip +$4.8M, site disruption -$10.1M, component disruption $0. Push-outs and acceptance slips raise revenue only because they smooth lumpy demand under the penalty (without the penalty: -$2.4M and -$0.9M). A delay event that increases revenue is a symptom of difference 1.
- Horizon-edge push-out mirroring (sim L135-141) and acceptance-slip inflow at month 1 (L384-385): sim only, intentional steady-state assumptions.
- Labor-multiplier clips (sim L232, L278-279): at `ems_labor_sigma=0.05` the asymmetric integration clip [0.8, 1.05] biases mean integration capacity by -0.44%. Small, accidental.
- Revenue-recognition fill for the first lag month: identical formula in both engines (baseline L166, sim L380); not a difference.
- FPY never reduces output in either engine; the sim comment "good output = builds * fpy_eff" (L281) is wrong.

## Implications

- **P(plan) is mostly engine mismatch.** Full uncertainty, 3,000 draws, FY plan $2,556M (the `financial_plan` input): P(FY plan) = 63.0% as-is; 80.6% with the penalty removed; about 94.8% if the zero-shock gap ($200.7M) is added as a level shift (first-order estimate). About 32 of the 37 points of reported plan risk come from the engine disagreeing with itself, not from uncertainty.
- **Lever values are distorted.** EMS capacity actions read about 2.3x their penalty-free value (+$168.7M vs +$71.9M for x1.3); integration actions read lower. This violates "levers must do what their labels say".
- **Inventory target reads differ by engine.** Baseline $593M vs sim $524M against a $600M target; $86M of that is a sim-only behavioral assumption (damping).
- **Customer exposure is invisible in the sim.** With every other difference aligned, the greedy serves priority 1 and 2 customers 100% and pushes the whole FY shortfall onto the priority-3 customer (75% fill). Proportional rationing shorts the top three customers, who hold 70% of FY revenue, by 41 units (fill 92-96%). The sim cannot say who is shorted at all.
- **Vocabulary gap.** `docs/ARCHITECTURE.md` says the plan of record is derived from the baseline; in code the plan is an independent input (`data_generator.py` L360-374, demand revenue x 0.86) and the baseline sits 11% above it.

## Proposed single-engine design

**Principle.** One function computes the outcome for n draws. The baseline supply plan is that function run with `Shocks.zero()` and n = 1. There is no second allocator and no second financial translation.

1. **Shocks object.** Every stochastic term reads a named, owned amplitude: add sigmas for component tightness, logistics and EMS execution factor shocks (today unscaled), and route event probabilities (cancel, push, pull, disruptions, acceptance slip) through the same object. `Shocks.zero()` sets them all to 0. Test: zero-shock variance across draws is exactly 0.
2. **Allocation: vectorized customer-priority greedy.** The baseline's sort keys (backlog, customer priority, requested month, margin) do not depend on the draw, so the queue order is fixed. Iterate over customer x family cells in that order (15 per month plus carried backlog held as a state array per cell), and at each cell take `min(want, component ceiling, integration remaining, site capacity in fixed site order)` as numpy operations across all draws. That is about 18 x 15 x 4 vector operations per run: live-recompute speed at 5,000 draws, and it carries customer x family x month, which the concentration question needs. Proportional rationing survives only as an explicitly labeled allocation policy ("fair share") for comparison, never as a silent approximation.
3. **Utilization penalty.** Keep only if the owner of the adherence assumption confirms it changes a decision. If kept, drive it from planned load per site (at most 100%), not unconstrained demand against company capacity, and it then applies in the zero-shock run too, so the plan of record includes it.
4. **Supply timing.** No structural receipt delay at zero shocks: PO dates are facts. Expected lateness belongs in the shock model or as an owned input.
5. **One policy each** for safety stock (usable share, one number, used by both allocation and valuation), unit cost (one COGS and FG function), and inventory response (damping on or off in both). Drop integer floors or apply them everywhere.
6. **Constraint reporting.** Log the physical ceiling that stopped each cell and each unit of shortfall once. Decide whether installation capacity is a real constraint (worth $61M if so) or remove it from the arrays; per CLAUDE.md the target constraints are EMS capacity and critical components.
7. **Migration.** Build the kernel in `spikes/` first; `sim_variant.py` with `allocator="greedy"` already reproduces the baseline exactly and is the starting point. Promote to `src/` after this design is agreed.

## Reconciliation test to add

`tests/test_engine_reconciliation.py` (after the design lands):

```python
def test_zero_shock_sim_equals_baseline(data, config, baseline):
    zero = Shocks.zero(config)
    sim = run_simulation(data, zero.config, baseline, params=zero.params, n_sims=3)
    assert np.ptp(sim.revenue, axis=0).max() == 0              # no unscaled shocks
    np.testing.assert_allclose(sim.family_shipped[0], baseline.supply_units, atol=1e-6)
    for sim_arr, col in [(sim.revenue, "revenue_usd"), (sim.cogs, "cogs_usd"),
                         (sim.inventory, "inventory_usd")]:
        np.testing.assert_allclose(sim_arr[0], baseline.monthly[col], rtol=1e-9)
```

Until then, `python spikes/engine_reconciliation/check_equivalence.py` and `run_reconciliation.py` reproduce every number in this document in about 15 seconds.
