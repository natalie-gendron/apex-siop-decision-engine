# Lever propagation spec (2026-09)

Executable version: `tests/test_propagation_spec.py`. Each row is one test. It runs
600 sims with seed 7 and common random numbers, so each difference between two runs
is caused by the lever. **xfail** means the lever does not propagate today. These are
strict xfails, so the suite fails as soon as a fix lands, and whoever fixes it must
update this table.

Governing rule: model a mechanism only when leaving it out would change the
financial answer.

| Lever / mechanism | Expected behavior | Status today | Code cause (`src/simulation.py`) | Changes a financial answer? | Recommendation |
|---|---|---|---|---|---|
| Demand +20%, EMS binding | Shipments rise less than demand; past-due rises; service falls | pass (EMS tightened 10% in the test so it binds; base-world EMS binds in one FY month) | n/a | Yes: revenue, service, expedite cost | Keep. (Before step 1, FY revenue fell under +20% demand because the utilization adherence penalty eroded output; the penalty is removed) |
| Demand -20% | Component inventory (cash tied up) rises through the lead-time window | pass (restated in step 4, user decision) | Open POs inside the lead time keep arriving; buyers cut beyond it, so the excess is worked off by year end and year-end E&O no longer rises | Yes: inventory, cash | Lasting E&O from a drop belongs to obsolescence risk (step 5) |
| Price (ASP) +10% | Revenue +10%; units and COGS unchanged | pass | n/a | Yes: revenue, GM | Keep |
| Relieve EMS capacity | Output rises, then plateaus as the next constraint binds | pass | n/a | Yes: value of capacity actions | Keep. Since build step 2 integration is part of EMS capacity, and the next constraint is critical components; the test asserts that |
| Overtime | Shipments and conversion cost (COGS per unit) rise | pass | n/a | Yes: revenue vs premium | Keep |
| First-pass yield down | Rework uses EMS capacity, so shipments fall when capacity binds | xfail | `fpy_eff` only feeds `rework_cost`; never touches `site_cap` or `shipped` | Yes: at 86% EMS utilization, lost output is worth far more than rework cost | Make it propagate: scale effective site capacity by yield |
| Lead time longer | Supply response is delayed, so shortage rises when demand increases | pass (step 4) | Fixed: lead time sets the open-PO window before planned orders can arrive | Yes: the lever gave the wrong sign on revenue | Done |
| Component purchasing response | Receipts follow sustained demand after the lead time | xfail, **deferred by the user as calibration** | Mechanism built in step 4 (order-up-to beyond the lead time; shortage clears with no demand noise, and with supply shocks alone). Under lumpy demand the data's ~0.6-month safety stock is too thin, so late shortage stays above months 5-7 | Yes | Revisit with calibrated safety stock and demand lumpiness |
| Safety stock policy up: service | Service does not fall | pass (step 4) | Fixed: the policy is the buffer target buyers order toward. Release of the held-back 30% under shortage is step 5 | Yes, slightly | Done for buying |
| Safety stock policy up: inventory | Average raw inventory rises | pass (step 1) | Fixed: raw inventory is physical stock including safety stock. Caveat: the rise is small (+$0.8M at 2x policy) and comes from fewer builds, not buy-ahead; the service row still fails until the policy drives purchases | Yes: inventory and cash, the main cost of the policy | Done for valuation; purchases in step 5 |
| Permanent capacity (take-or-pay, headcount) | Recurring cost every month after the action takes effect | xfail | `action_cost_m[:3]`: one-time Q1 cost only | Yes: understates the run-rate cost of the action, and the EV favors it | Make it propagate: add a `recurring_cost_usd_per_month` from the start month |
| Buy more components | AP rises with purchases; cash timing follows receipts | xfail | `ap = 0.75 * cogs * dpo / 30`, based on COGS, not receipts | Yes: cash impact of a buy-ahead is overstated by the full AP offset | Make it propagate: AP on receipt value |
| Obsolescence risk | E&O reflects component `obsolescence_risk` | xfail | `eo_reserve` uses one flat `eo_reserve_rate`; column never read | Yes, where excess sits in high-risk parts (for example buy-ahead of a single part) | Make it propagate: weight the reserve by risk. If excess is always diversified, remove the column instead |
| Customer-level outputs | Revenue by customer (concentration) | pass (step 3) | Fixed: demand lines customer x family; `customer_revenue` and customer metrics | Yes: who gets shorted drives concentration risk (design commitment) | Make it propagate per `docs/design-customer-dimension.md` |
| Zero-shock reconciliation | Sim with no shocks matches baseline FY revenue within 1% | pass (step 1) | Fixed: the baseline is the engine with `Shocks.zero()`; exact equality in `tests/test_engine_reconciliation.py` | Yes: the base case and the plan disagree ("two engines" defect) | Done |

## Summary

After build step 4, ten pass and five are xfail (one deferred by the user). At discovery, five passed and ten were xfail. Nine of the ten original xfails failed the governing rule, because
leaving the mechanism out changes a financial answer. The obsolescence row is the
only one where removing the input is a legitimate choice. No lever is purely cosmetic
enough to be relabeled "cost-only" and left alone. FPY comes closest, and at current
utilization, calling it cost-only would still hide lost revenue.

Priority for design work: (1) purchasing response with lead time, which fixes lead
time and part of safety stock, (2) FPY capacity, (3) recurring action cost,
(4) AP on receipts, (5) integration at the EMS, (6) customer dimension and one
engine.
