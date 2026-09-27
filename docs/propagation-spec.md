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
| First-pass yield down | Rework uses EMS capacity, so shipments fall when capacity binds | pass (step 5) | Fixed: each failed unit takes `REWORK_SHARE` (50%) of a build slot, the same share that sets rework cost. Thin margin (-1.1% vs the 1% threshold) because EMS binds in fewer months since step 4 | Yes | Done |
| Lead time longer | Supply response is delayed, so shortage rises when demand increases | pass (step 4) | Fixed: lead time sets the open-PO window before planned orders can arrive | Yes: the lever gave the wrong sign on revenue | Done |
| Component purchasing response | Receipts follow sustained demand after the lead time | xfail, **deferred by the user as calibration** | Mechanism built in step 4 (order-up-to beyond the lead time; shortage clears with no demand noise, and with supply shocks alone). Under lumpy demand the data's ~0.6-month safety stock is too thin, so late shortage stays above months 5-7 | Yes | Revisit with calibrated safety stock and demand lumpiness |
| Safety stock policy up: service | Service does not fall | pass (step 4) | Fixed: the policy is the buffer target buyers order toward. Release of the held-back 30% under shortage is step 5 | Yes, slightly | Done for buying |
| Safety stock policy up: inventory | Average raw inventory rises | pass (step 1) | Fixed: raw inventory is physical stock including safety stock. Caveat: the rise is small (+$0.8M at 2x policy) and comes from fewer builds, not buy-ahead; the service row still fails until the policy drives purchases | Yes: inventory and cash, the main cost of the policy | Done for valuation; purchases in step 5 |
| Permanent capacity (take-or-pay, headcount) | Recurring cost every month after the action takes effect | pass (step 5) | Fixed: claim-sheet `recurring_cost_usd_per_month`; reserved EMS capacity carries the data's reservation fee ($59k a month) | Yes | Done |
| Buy more components | AP rises with purchases; cash timing follows receipts | pass (step 5) | Fixed: AP on purchases (parts received, other material, EMS conversion, freight). The test's lever became a safety-stock buy-ahead (user approval), since supply multipliers now scale capacity | Yes | Done |
| Obsolescence risk | E&O reflects component `obsolescence_risk` | pass (step 5) | Fixed: each part's excess is reserved at the policy rate plus its obsolescence probability on the remainder | Yes | Done |
| Customer-level outputs | Revenue by customer (concentration) | pass (step 3) | Fixed: demand lines customer x family; `customer_revenue` and customer metrics | Yes: who gets shorted drives concentration risk (design commitment) | Make it propagate per `docs/design-customer-dimension.md` |
| Zero-shock reconciliation | Sim with no shocks matches baseline FY revenue within 1% | pass (step 1) | Fixed: the baseline is the engine with `Shocks.zero()`; exact equality in `tests/test_engine_reconciliation.py` | Yes: the base case and the plan disagree ("two engines" defect) | Done |

## Summary

After build step 5, fourteen pass and one is xfail: component purchasing response, deferred by the user as calibration (the mechanism is built; the remaining gap is safety-stock sizing against lumpy demand). At discovery, five passed and ten were xfail. Nine of the ten original xfails failed the governing rule, because
leaving the mechanism out changes a financial answer. The obsolescence row is the
only one where removing the input is a legitimate choice. No lever is purely cosmetic
enough to be relabeled "cost-only" and left alone. FPY comes closest, and at current
utilization, calling it cost-only would still hide lost revenue.

All six design priorities from discovery are built (build steps 1 to 5,
2026-09): one engine and the customer dimension, integration at the EMS,
purchasing response with lead time, FPY capacity, recurring action cost, and
AP on purchases.
