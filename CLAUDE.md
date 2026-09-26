# APEX: working principles

APEX translates operational decisions into financial consequences (revenue,
gross margin, inventory, E&O exposure, cash) so management can make and align
on tradeoffs. It is decision support, not a planning system.

## The governing rule

**Model an operational mechanism only when leaving it out would change the
financial answer.** Every proposed model detail must name the decision and the
financial output it changes. If it can't, it doesn't go in.

## What APEX is not

- Not a demand forecast. Demand Planning owns the one consensus demand plan.
  APEX analyzes uncertainty around it and never produces a "better" forecast.
- Not an MRP or execution planning tool. Detailed planning truth lives in the
  planning system (e.g. Kinaxis). APEX keeps only rough-cut logic needed for
  financial translation and fast simulation.
- Not a second source of operational truth. Facts (demand plan, on-hand, POs,
  lead times, capacity, standard costs) are read-only inputs.
- Not the owner of the plan. The revenue plan and financial targets are
  inputs from FP&A, never computed inside APEX.

## Design commitments

- **One engine.** The deterministic baseline is the same engine run with zero
  shocks. Two engines giving two answers is a defect.
- **Customer-level detail matters** (revenue concentration). Carry customer x
  family x month wherever a financial answer depends on who gets shorted.
- **Target business constraints:** EMS partner capacity plus critical
  components. Final integration is performed at the EMS, not in-house.
- **Live recompute** is a requirement for the decision workspace.
- **Levers must do what their labels say.** A lever that only changes cost
  must be labeled cost-only, or made to propagate.
- **Probability only where uncertainty can change the decision.**
- **Agents propose, humans approve.** Agents never change the demand plan,
  supply plan, or approved assumptions directly. Pattern of record:
  `scripts/refresh_intel.sh` (structured output, bounded influence, human
  merges a PR).
- Assumptions carry an owner, a source and an as-of date.

## Reference docs

- `docs/ARCHITECTURE.md`: world x response model, vocabulary of record.
- `docs/decision-catalog.md`, `docs/engine-reconciliation.md`,
  `docs/data-contract.md`, `docs/design-customer-dimension.md`: discovery
  and design work (2026-09).

## Working rules

- Design work goes in `docs/`. Exploratory code goes in `spikes/`, never
  `src/`, until a design is agreed.
- Do not change `src/` behavior without an agreed design doc.
- No real company data, names or confidential information in this repo. All
  data is synthetic.
- Writing style for docs: executive-ready, concise, no em dashes.
- Tests: `python -m pytest -q` (UI tests need streamlit installed).
