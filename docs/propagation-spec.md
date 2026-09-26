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
| Demand -20% | Component inventory and E&O rise | pass | n/a (flat receipts make this happen) | Yes: inventory, E&O, cash | Keep, but see purchasing response below: the size of the effect is overstated |
| Price (ASP) +10% | Revenue +10%; units and COGS unchanged | pass | n/a | Yes: revenue, GM | Keep |
| Relieve EMS capacity | Output rises, then plateaus as the next constraint binds | pass | n/a | Yes: value of capacity actions | Keep. Since build step 2 integration is part of EMS capacity, and the next constraint is critical components; the test asserts that |
| Overtime | Shipments and conversion cost (COGS per unit) rise | pass | n/a | Yes: revenue vs premium | Keep |
| First-pass yield down | Rework uses EMS capacity, so shipments fall when capacity binds | xfail | `fpy_eff` only feeds `rework_cost`; never touches `site_cap` or `shipped` | Yes: at 86% EMS utilization, lost output is worth far more than rework cost | Make it propagate: scale effective site capacity by yield |
| Lead time longer | Supply response is delayed, so shortage rises when demand increases | xfail | `lead_time_mult` only raises `delay_frac` (one-month slip); a bigger delayed pool means more expediting, so shortage **falls** | Yes: the lever gives the wrong sign on revenue | Make it propagate together with the purchasing response. Until then, remove it from decision UIs |
| Component purchasing response | Receipts follow sustained demand after the lead time | xfail | `receipts_nominal` is `open_po_units_per_month` repeated for 18 months | Yes: an 18-month demand scenario overstates shortage on the upside and inventory/E&O on the downside | Make it propagate: rough-cut reorder of forecast requirement, gated by lead time |
| Safety stock policy up: service | Service does not fall | xfail | `safety_floor` is subtracted from usable supply and no receipts are added | Yes, slightly: service and revenue move the wrong way | Make it propagate: a higher policy adds buy-ahead receipts, and the floor is released under shortage |
| Safety stock policy up: inventory | Average raw inventory rises | pass (step 1) | Fixed: raw inventory is physical stock including safety stock. Caveat: the rise is small (+$0.8M at 2x policy) and comes from fewer builds, not buy-ahead; the service row still fails until the policy drives purchases | Yes: inventory and cash, the main cost of the policy | Done for valuation; purchases in step 5 |
| Permanent capacity (take-or-pay, headcount) | Recurring cost every month after the action takes effect | xfail | `action_cost_m[:3]`: one-time Q1 cost only | Yes: understates the run-rate cost of the action, and the EV favors it | Make it propagate: add a `recurring_cost_usd_per_month` from the start month |
| Buy more components | AP rises with purchases; cash timing follows receipts | xfail | `ap = 0.75 * cogs * dpo / 30`, based on COGS, not receipts | Yes: cash impact of a buy-ahead is overstated by the full AP offset | Make it propagate: AP on receipt value |
| Obsolescence risk | E&O reflects component `obsolescence_risk` | xfail | `eo_reserve` uses one flat `eo_reserve_rate`; column never read | Yes, where excess sits in high-risk parts (for example buy-ahead of a single part) | Make it propagate: weight the reserve by risk. If excess is always diversified, remove the column instead |
| Customer-level outputs | Revenue by customer (concentration) | xfail | Demand aggregated to family x month; `SimulationResult` has no customer axis | Yes: who gets shorted drives concentration risk (design commitment) | Make it propagate per `docs/design-customer-dimension.md` |
| Zero-shock reconciliation | Sim with no shocks matches baseline FY revenue within 1% | pass (step 1) | Fixed: the baseline is the engine with `Shocks.zero()`; exact equality in `tests/test_engine_reconciliation.py` | Yes: the base case and the plan disagree ("two engines" defect) | Done |

## Summary

After build step 1, seven pass and eight are xfail. At discovery, five passed and ten were xfail. Nine of the ten original xfails failed the governing rule, because
leaving the mechanism out changes a financial answer. The obsolescence row is the
only one where removing the input is a legitimate choice. No lever is purely cosmetic
enough to be relabeled "cost-only" and left alone. FPY comes closest, and at current
utilization, calling it cost-only would still hide lost revenue.

Priority for design work: (1) purchasing response with lead time, which fixes lead
time and part of safety stock, (2) FPY capacity, (3) recurring action cost,
(4) AP on receipts, (5) integration at the EMS, (6) customer dimension and one
engine.
