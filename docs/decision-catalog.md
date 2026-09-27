# APEX Decision Catalog

*Discovery work, 2026-09. Synthetic context only. Status: draft for review.*

This catalog lists the recurring decisions that ops leadership and the CFO's
capacity meeting face at an ATE manufacturer that builds through EMS partners,
and checks what APEX must model to translate each one into revenue, margin,
inventory, E&O and cash. The governing rule decides scope: **model an
operational mechanism only when leaving it out would change the financial
answer.** Everything else stays in the planning system.

Conventions: "sim" is `src/simulation.py::run_simulation` (drives every KPI,
comparison and recommendation); "baseline" is `src/baseline_plan.py::run_baseline`
(plan-of-record view only). Action keys refer to `config/management_actions.yaml`.

## Cross-cutting findings (apply to every decision below)

1. **No customer dimension in the sim.** Demand is collapsed to family x month
   and rationed proportionally (`run_simulation`, component and EMS
   water-filling loop). Customer priority exists only in the baseline sort
   (`_sorted_demand_queue`, `run_baseline`). Every financial output is
   therefore blind to who gets shorted.
2. **Purchases are a flat pipe.** Receipts are `open_po_units_per_month`
   tiled across all 18 months (`receipts_nominal` in `run_simulation`). No
   decision changes purchasing except through hand-authored multipliers.
   Lead time only sets a one-month slip fraction (`delay_frac`).
3. **Commercial terms are missing.** `capacity_reservation_fee_usd`,
   `reserved_capacity_units`, `flexible_capacity_units`, `min_order_qty`,
   `alt_source_available`, `max_ramp_pct_per_month` and `min_production_lot`
   are generated in `src/data_generator.py` (`_components`, `_ems`) but read
   by nothing in `src/`. Take-or-pay, NCNR liability and cancellation windows
   cannot be priced.
4. **Action cost is one-time and front-loaded.** `action_cost_usd` is spread
   over months 0-2 (`action_cost_m[:3]` in `run_simulation`). Permanent
   actions have no recurring cost, and ranking uses EV = delta gross profit
   minus action cost (`src/recommendations.py::build_recommendations`), which
   ignores cash, working capital and E&O.
5. **Resolved 2026-09 (build step 2): integration folded into EMS capacity.** **Final integration is modeled in-house.** A separate integration stage
   with calibration, FAT and install capacity (`_integration`,
   `integ_cap` in `run_simulation`) contradicts the target business, where
   integration happens at the EMS.
6. **AP and E&O are not tied to purchases.** AP = 0.75 x COGS x DPO/30; E&O
   = 25% of critical stock above 2.5 months of usage at month 12 plus 5% of
   FG, computed on `stock_path` rather than the damped `held_stock`. A buy-ahead
   raises inventory but not payables, so cash impact is overstated.

---

## D1. Authorize EMS overtime

- **Question:** Do we pay the EMS overtime premium this month or next to
  ship more systems in-quarter?
- **Owner:** VP Operations; CFO signs above a threshold.
- **Horizon:** 0-3 months.
- **Financial outputs:** quarter revenue, gross margin (premium), service to
  key customers.
- **Must model:** EMS site capacity with overtime ceiling and premium; whether
  components, not labor, bind (overtime is worthless if the FPGA binds);
  recognition lag to see which quarter the revenue lands in. **Not needed:**
  shift schedules, labor rosters, line balancing.
- **Uncertainty matters?** Yes, moderately: overtime value depends on whether
  demand holds and components arrive. A deterministic "is labor binding" check
  gets most of the answer.
- **Data:** overtime ceiling and premium (EMS contracts), capacity (planning
  system), component coverage (planning system).
- **APEX today: Partial.** `overtime_fraction`, `overtime_start_month` and
  `overtime_premium_pct` are modeled in `run_simulation`. Gaps: the one-time $5.0M
  `action_cost_usd` sits on top of the per-unit premium with no stated basis
  (possible double count); no
  per-site choice (all sites at 80%); customer who benefits is not visible.

## D2. Reserve or add EMS capacity (take-or-pay)

- **Question:** Do we commit to reserved EMS capacity for the next 2-4
  quarters, knowing we pay whether or not we use it?
- **Owner:** VP Operations with CFO; Supply Chain negotiates.
- **Horizon:** 3-12 months.
- **Financial outputs:** revenue upside, unabsorbed fee in downside, margin,
  and the probability-weighted spread between them.
- **Must model:** reserved capacity by site x month with fee per unused unit
  (the downside is the whole point); ramp limits; qualification by family.
  **Not needed:** EMS internal routing.
- **Uncertainty matters?** Yes, decisively. Take-or-pay is an option priced
  against demand variance; a deterministic answer always says "reserve" or
  "don't" with no view of the downside tail.
- **Data:** reservation fee, minimum commitment, ramp rate (EMS contracts);
  capacity (planning system).
- **APEX today: Partial.** `ems_capacity_add_ramp` adds units and
  `action_cost_usd: 4.5e6` is a lump sum. Gaps: `capacity_reservation_fee_usd`
  and `reserved_capacity_units` are generated but unused, so an unused
  reservation costs the same as a used one; no recurring fee; no ramp cap
  (`max_ramp_pct_per_month` unused).

## D3. Buy ahead or commit non-cancellable long-lead parts

- **Question:** Do we place NCNR orders now for parts arriving in 6-9 months,
  accepting E&O risk if demand does not show?
- **Owner:** VP Supply Chain; CFO for large commitments.
- **Horizon:** 6-18 months.
- **Financial outputs:** future-quarter revenue protected, inventory and cash
  outflow timing, E&O exposure under downside demand.
- **Must model:** purchases as a decision variable with lead time as a real
  delay (order in month t, receive t + LT); cancellation windows (NCNR vs
  cancellable portion); cash out at receipt, AP from purchases; E&O on the
  resulting excess. **Not needed:** full MRP netting, multi-level BOM.
- **Uncertainty matters?** Yes, decisively. The decision is a bet on demand
  six months out; value lies in the tails.
- **Data:** open POs and lead times (planning system), NCNR terms (supplier
  contracts, ERP), standard cost (ERP).
- **APEX today: Partial.** `comp_supply_ramp` with start month (e.g.
  `Commit long-lead component orders`, `Increase FPGA safety stock`) moves
  receipts. Gaps: timing is hand-authored, not derived from lead time;
  purchases never respond to demand; no NCNR liability or cancellation;
  AP ignores purchases; E&O only measured at month 12.

## D4. Expedite

- **Question:** Do we pay premium freight or broker prices to pull specific
  parts in?
- **Owner:** Supply Chain director; escalates to VP Ops above threshold.
- **Horizon:** 0-2 months.
- **Financial outputs:** in-quarter revenue recovered vs premium cost; margin.
- **Must model:** which component binds which family and which customer order;
  premium per unit; need-capped recovery. **Not needed:** carrier selection,
  broker sourcing.
- **Uncertainty matters?** Somewhat. Supply disruption drives need, but the
  call is usually made on a known shortage.
- **Data:** shortages and POs (planning system), premiums (ERP history,
  broker quotes).
- **APEX today: Supported (aggregate).** `expedite_recovery`,
  `expedite_recovery_by_comp` and need-capped `recovered` in
  `run_simulation`; premium via `comp_expedite_prem`. Gaps: no link to a
  specific customer order; recovery fraction is an analyst claim, not a
  quote.

## D5. Dual-source or qualify an alternate part or EMS site

- **Question:** Do we spend qualification cost and engineering time to open a
  second source or site?
- **Owner:** VP Operations with Engineering; CFO for capex or NRE.
- **Horizon:** 6-18 months.
- **Financial outputs:** protected revenue in disruption scenarios, margin
  (alternate part or site cost), qualification cost.
- **Must model:** qualification matrix with effective date; alternate source
  as additional supply with its own lead time, cost and capacity; disruption
  risk per source. **Not needed:** qualification project plan.
- **Uncertainty matters?** Yes, decisively. The value is almost entirely
  insurance against disruption; the base case shows little benefit.
- **Data:** qualification status (Engineering), alternate cost and capacity
  (supplier or EMS quotes), disruption history (Supply Chain).
- **APEX today: Partial.** `add_qualification` (site x family x month) works;
  `Dual-source the high-end FPGA` is a +30% receipts ramp. Gaps: a second
  source is modeled as more of the same supply, not an independent source
  with separate disruption risk, so the insurance value is understated;
  `alt_source_available` and `alt_source_qual_months` are unused; no cost
  delta for the alternate.

## D6. Allocate scarce supply across customers

- **Question:** When we cannot ship everything, who gets shorted, and what
  does that cost in revenue, concentration and relationship?
- **Owner:** VP Sales with VP Ops; CEO or CFO arbitrates key accounts.
- **Horizon:** 0-3 months, recurring.
- **Financial outputs:** revenue and margin by customer, revenue concentration,
  contractual penalties, quarter timing.
- **Must model:** customer x family x month demand; allocation policy
  (priority, margin, contract) as a selectable rule; per-customer ASP and
  acceptance terms. **Not needed:** order-line or serial-level allocation.
- **Uncertainty matters?** Partly. The policy choice is mostly deterministic;
  uncertainty matters for how often a key account gets hit.
- **Data:** demand by customer (CRM, demand plan), priorities and penalties
  (Sales, contracts), ASP by customer (ERP or CRM).
- **APEX today: Missing in the decision engine.** Priority sort exists only in
  `run_baseline`; the sim rations proportionally at family level. The
  `Customer Groups` export (`src/exports.py`) shows plan demand, not shortfall.
  This is the largest gap given concentration matters.

## D7. Accept or commit to a customer upside or pull-in

- **Question:** A customer asks for more systems or earlier dates. Can we
  commit, what does it cost, and whose shipments does it displace?
- **Owner:** VP Sales proposes; VP Ops commits; CFO if it needs spend.
- **Horizon:** 0-6 months.
- **Financial outputs:** incremental revenue and quarter timing, margin after
  enablers (overtime, expedite), displaced revenue from other customers.
- **Must model:** customer-level demand edit; constrained allocation to show
  displacement; enabler costs. **Not needed:** configuration-level feasibility.
- **Uncertainty matters?** Mostly no. The ask is known; a deterministic
  capacity and parts check answers it. Probability adds little here.
- **Data:** the request (CRM), constraints (planning system).
- **APEX today: Missing.** Only stochastic `pullin_prob` and scenario
  `demand_family_mult` exist in `run_simulation`. No way to enter a specific
  customer request or see displacement.

## D8. Inventory reduction and purchase push-outs

- **Question:** Do we push out or cancel open POs to release cash, and how
  much stockout risk do we accept?
- **Owner:** CFO initiates; VP Supply Chain executes.
- **Horizon:** 1-3 quarters.
- **Financial outputs:** cash and inventory released, E&O avoided, revenue
  at risk if demand firms, cancellation charges.
- **Must model:** PO schedule with cancellation windows by lead time; push-out
  as a shift in receipts (not a cut); E&O and AP responding to purchases.
  **Not needed:** supplier-by-supplier negotiation.
- **Uncertainty matters?** Yes. The tradeoff is cash now vs revenue risk if
  demand recovers.
- **Data:** open POs with dates and terms (ERP, planning system), inventory
  (ERP).
- **APEX today: Partial.** `Inventory reduction initiative` sets
  `comp_supply_mult: 0.94` and `safety_stock_mult: 0.6` for all 18 months. Gaps:
  a flat cut is not a push-out; parts inside lead time cannot be cancelled in
  reality but can here; `safety_stock_mult` lowers a hard floor
  (`safety_floor = comp_safety * mult * 0.3`), so cutting safety stock
  *adds* usable supply and improves service: the stockout exposure the label
  promises never appears, and raising safety stock (D10) cuts shipments.

## D9. Respond to a customer push-out or cancellation

- **Question:** A key customer pushes or cancels. Do we slow purchases, cut
  EMS commitments, or redeploy to other customers?
- **Owner:** VP Ops and CFO; Sales on customer terms.
- **Horizon:** 0-9 months.
- **Financial outputs:** revenue timing, cancellation fees received, E&O and
  inventory from committed supply, unabsorbed EMS reservation.
- **Must model:** customer-level demand edit; purchases that can or cannot be
  unwound (NCNR); reservation fees on unused capacity; redeployment to other
  customers. **Not needed:** detailed rescheduling.
- **Uncertainty matters?** The event itself is known; uncertainty matters for
  whether it becomes a cancellation.
- **Data:** customer notice (CRM), cancellation terms (contracts), POs (ERP).
- **APEX today: Partial.** `forced_pushout` (family, months, units) in
  `run_simulation` and scenario `Major Customer Push-Out`
  (`src/scenarios.py::prebuilt_scenarios`). Gaps: family not customer;
  modeled as world only, with no response lever (purchases do not respond);
  no customer cancellation fee; E&O impact is muted because supply keeps
  flowing unchanged.

## D10. Set component safety-stock and buffer policy

- **Question:** How many weeks of cover do we hold on constrained parts?
- **Owner:** VP Supply Chain; CFO on the working-capital envelope.
- **Horizon:** 1-4 quarters.
- **Financial outputs:** inventory and cash, E&O exposure, service level.
- **Must model:** buffer as a target that drives purchases, consumed in a
  shortage. **Not needed:** statistical safety-stock formulas beyond a
  weeks-of-cover target.
- **Uncertainty matters?** Yes. Buffers only have value under variance.
- **Data:** policy (Supply Chain), usage variance (planning system).
- **APEX today: Partial.** `safety_stock_mult` exists but, as noted in D8,
  it acts as a floor that reduces usable supply and does not drive buying.

---

## Summary gap table

| Decision | Current support | Biggest gap | Mechanism needed | Priority |
| --- | --- | --- | --- | --- |
| D1 EMS overtime | Partial | Possible double cost; no customer view | Customer allocation; per-site lever | M |
| D2 Reserve EMS capacity | Partial | Unused reservation costs nothing extra | Take-or-pay fee by month, ramp cap | H |
| D3 Buy ahead / NCNR | Partial | Purchases do not respond, no lead-time delay | Purchase decision with LT offset, NCNR, AP from purchases | H |
| D4 Expedite | Supported | No link to specific customer orders | Customer allocation | L |
| D5 Dual-source / qualify | Partial | Alternate treated as more of same supply | Independent source with own risk and cost | M |
| D6 Customer allocation | Missing | Sim has no customer dimension | Customer x family x month with policy rule | H |
| D7 Upside / pull-in | Missing | No way to enter a specific ask | Customer demand edit plus displacement view | H |
| D8 Inventory reduction | Partial | Flat cut, not a push-out; backwards safety-stock proxy | PO schedule with cancellation windows | H |
| D9 Customer push-out | Partial | Family-level, supply does not react | Customer edit plus purchase response and NCNR | H |
| D10 Buffer policy | Partial | Buffer does not drive purchasing | Buffer as purchase target | M |

## Mechanisms ranked by how many decisions need them (build list)

1. **Customer x family x month in the one engine, with a selectable
   allocation rule** (D1, D4, D6, D7, D9; indirectly all). Also removes the
   ~7% revenue gap between the two engines by collapsing them into one.
2. **Purchases as a response: order-to-receipt lead-time offset, driven by
   demand and buffer target** (D3, D8, D9, D10).
3. **Supplier and EMS commercial terms: NCNR and cancellation windows,
   take-or-pay fees on unused capacity, recurring vs one-time cost** (D2, D3,
   D8, D9).
4. **Cash chain tied to purchases: AP from receipts, E&O measured on the
   resulting excess at any month, not only month 12** (D3, D8, D9, D10).
5. **Specific-event demand edits at customer level (upside, pull-in,
   push-out, cancel)** with a deterministic run (D7, D9).
6. **Independent alternate sources and sites with their own risk, cost and
   capacity** (D5, partly D2).
7. **Done (build step 2).** **Remove the in-house integration stage** and fold integration into EMS
   capacity (affects D1, D2 answers; model simplification, not addition).

Explicitly not on the list: FPY driving output (FPY affects cost only today
and no decision above changes because of it), shift-level labor, multi-level
BOM, detailed MRP netting. These stay in the planning system.

## Open questions for the user

1. Does the capacity meeting actually decide D2 (reservations) and D3 (NCNR
   buys), or are those made by Supply Chain alone and reported afterwards?
2. How are customers prioritized today when supply is short: a written rule,
   margin, strategic tier, or case by case? Is there a contractual penalty for
   late delivery with any key account?
3. What does Kinaxis provide today: parts and PO visibility only, or solved
   constrained supply plans APEX could read as an input?
4. Do EMS contracts include take-or-pay or minimum volume commitments, and are
   overtime premiums fixed by contract or quoted each time?
5. Which parts are bought NCNR, and what share of the component spend is
   inside the cancellation window at any time?
6. Who owns the revenue plan APEX should compare against (FP&A), and at what
   grain: company, family, or customer?
7. Is there any integration, calibration or FAT step still performed in-house,
   or can that stage be removed entirely?
8. Which two or three of these decisions came up in the last quarter's
   capacity meetings? Those should be the first to build.
