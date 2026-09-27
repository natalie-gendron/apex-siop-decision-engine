# SIOP Operating Model: A Decision System, Not a Meeting

*A reference design for SIOP when consequences can be computed in seconds and
agents can do the preparation. Company-neutral. A starting framework to
test, not a finished answer.*

---

## 1. The design thesis

Traditional SIOP is a monthly batch process for reconciling numbers: product
review, demand review, supply review, pre-SIOP, executive meeting. That
design exists because computing consequences was slow and expensive, so
organizations met monthly to agree on one set of numbers and hoped the
decisions would follow.

Two things change that:

1. **Consequences are cheap to compute.** An engine can translate any
   operational choice into revenue, margin, inventory, E&O and cash, with
   uncertainty, in seconds.
2. **Preparation is cheap to automate.** Agents can detect change, frame
   decisions, assemble options and draft the record.

So the scarce resource is no longer analysis. It is **executive attention and
timely judgment**. The operating model should reorganize around that:

> **SIOP becomes a continuous decision system with a monthly governance
> heartbeat. The unit of work is the decision, not the plan.**

Proactive means three concrete things in this design:

- **Decide before the lead time closes.** Every decision has a date by which
  it must be made to still matter.
- **Decide the rule once, not the case every time.** Most recurring choices
  become policies; people handle only exceptions.
- **See the price before saying yes.** Every commitment is priced in money
  and risk before it is made, not discovered in the quarter's results.

---

## 2. Design principles

1. **Decisions are the unit of work.** Meetings, analytics and agents exist
   to produce well-framed, priced, recorded decisions.
2. **One set of facts, many priced options.** Facts come from systems of
   record and are never edited in the decision layer. Options are cheap;
   facts are singular.
3. **Policies before exceptions.** If a decision recurs, write the rule,
   price the rule, and delegate its execution. Meetings change rules and
   handle breaches.
4. **Decide at the last responsible moment, not the last possible one.**
   Lead times set the decision calendar.
5. **Everything in money.** Units, utilization and fill rates are inputs.
   Management decides on revenue, margin, inventory, E&O, cash and service
   to named customers.
6. **Probability only where uncertainty changes the choice.** Point
   estimates for execution; distributions for commitments that are hard to
   reverse.
7. **Model a mechanism only when leaving it out changes the financial
   answer.** Keeps the engine small, fast and trusted.
8. **Humans own judgment; agents own the grind.** Agents detect, frame,
   price, challenge and record. They never decide, and never change the
   demand plan, supply plan or approved assumptions.
9. **Close the loop.** Every material decision gets an outcome review. What
   we learn updates the assumptions.
10. **Traceability over persuasion.** Every number traces to facts,
    assumptions, a policy and an engine run. No slide carries a number that
    cannot be regenerated.

---

## 3. SIOP and SOE: the envelope

| | SIOP | SOE (Sales and Operations Execution) |
|---|---|---|
| Horizon | ~3 to 24 months | ~0 to 13 weeks, the committed window |
| Question | What envelope do we operate within? | How do we execute inside it this week? |
| Grain | Monthly, family and customer | Weekly or daily, order level |
| Produces | Policies, commitments (capacity, supply), thresholds | Schedules, allocations, expedites |
| Systems | Decision engine (economics) + planning system (feasibility) | Planning system + ERP |

**SIOP sets the envelope; SOE executes inside it; the decision layer works
on the envelope and its boundary, never inside the execution window.**

- Within policy: SOE handles it. No escalation, no meeting.
- Breaks the envelope (spend over threshold, a protected customer displaced,
  the quarter moves): escalate with priced options.
- After the fact: the cost of what SOE did reactively (expedites, overtime,
  displaced customers, revenue timing) is summed monthly and fed back to
  SIOP as evidence.

---

## 4. System architecture

| Layer | Holds | Written by | Changes |
|---|---|---|---|
| **L0 Facts** | Demand plan, on-hand, POs, lead times, EMS capacity, standard costs, AOP | Systems of record only | Continuously |
| **L1a Assumptions** | Beliefs about the world: demand uncertainty by horizon and customer, supplier reliability, upside conversion rates | Named owners, evidence-based | Monthly |
| **L1b Policies** | Standing decisions: allocation rule, buffer policy, expedite and overtime rules, approval thresholds, risk appetite | SIOP decision forum | Quarterly, or on breach |
| **L2 Engine** | Translation of operations into financials, with uncertainty | Model steward, under change control | Releases |
| **L3 Workspace** | Priced options, tradeoff cards, live what-ifs | Decision participants | Per decision |
| **L4 Record** | Decisions, rationale, actions, outcome reviews | Humans sign; the system stores | Per decision |

Every assumption and policy carries an **owner, source, as-of date and
review date**. Agents may propose changes to L1; only owners approve them.

---

## 5. Policies: the envelope, written down

Most organizations run on unwritten policies discovered in each crisis.
Writing them down is the single biggest step from reactive to proactive,
because it lets execution move fast without escalating everything.

| Policy | Decides | Example form | Owner |
|---|---|---|---|
| **Risk appetite** | How much exposure we accept | "Non-cancellable exposure max $X; P(meet tier-1 commitments) at least 90%" | CFO |
| **Customer service tiers** | Who is protected when supply is short | Tier 1 protected; tier 2 fair share; tier 3 residual | CFO + Sales |
| **Allocation rule** | How scarce supply is divided within tiers | Strict priority / protect top N / proportional | SIOP forum |
| **Commit rule** | When upside and pull-ins may be accepted | "Accept if no tier-1 displacement and incremental cost under $Y" | Sales + Ops |
| **Buffer policy** | Safety stock and buy-ahead for critical parts | Weeks of cover by part risk class | Supply + Finance |
| **Expedite and overtime rules** | When recovery spend is automatic | "Expedite if it protects a tier-1 shipment this quarter" | Ops |
| **Approval thresholds** | Who can commit what | See section 7 | CFO |

**Every policy is priced.** The engine shows what each policy option costs
and protects (revenue, margin, cash, service by tier) so the forum chooses a
rule knowing its consequence. Policies are reviewed quarterly, or whenever a
breach shows the rule no longer fits.

---

## 6. The decision calendar: proactive by construction

The most useful proactive artifact is simple: **a list of decisions that
must be made soon, ordered by when their window closes.**

Decide-by date = when the effect is needed, minus lead time, minus decision
and execution latency.

| Decision | Effect needed | Lead time | Decide by | Status |
|---|---|---|---|---|
| Commit long-lead FPGA buy | Month 8 upside | 30 weeks | This month | Framed, pricing |
| Reserve EMS capacity | Months 7 to 10 | 3 months notice | Next month | Watching |
| Qualify second EMS site | Month 12 | 9 months | Next month | Not started |

This turns "we'll deal with it when it happens" into "this is the last month
this choice is cheap." It is also the main agenda generator for the
decision forum.

---

## 7. Decision rights

Decision rights depend on two questions: **how much is at stake, and how
reversible is it?**

| | Reversible (two-way door) | Hard to reverse (one-way door) |
|---|---|---|
| **Small** | Execution team decides within policy. Logged, not escalated. | Director decides; priced option required. |
| **Large** | Director + Finance decide; priced options required. | **Decision forum.** Priced with uncertainty, challenged, recorded, reviewed. |

Illustrative thresholds, to be set by the CFO:

| Tier | Authority | Scope |
|---|---|---|
| 1 | Ops / supply lead | Within policy; under $A incremental cost; no tier-1 impact |
| 2 | Ops director + Finance partner | Under $B, or any tier-2 displacement |
| 3 | Decision forum (CFO chair) | Over $B; any tier-1 impact; any one-way door over $C; policy changes |

**Assumption ownership:**

| Assumption | Owner |
|---|---|
| Consensus demand plan (the facts) | Demand Planning |
| Demand uncertainty, customer conversion odds | Demand Planning or Market Intelligence |
| Supplier reliability, lead-time risk | Supply Chain |
| EMS capacity and terms | EMS / Operations management |
| Standard costs, revenue recognition, targets | FP&A |
| External market signals | Market Intelligence (agent-assisted) |

Agents have **no decision rights**. They have proposal rights only.

---

## 8. Cadence: three tempos

### Continuous (daily, mostly automated)

- The Sentinel agent watches facts for material change and the SOE boundary
  for envelope breaches.
- Breaches become framed decision requests routed by approval tier.
- The standing outlook refreshes as facts change, so the numbers are always
  current and nobody waits for month end.

### Weekly (SOE, 30 minutes, ops-led)

- Execution inside the envelope. Escalations only.
- Tier 2 decisions made here, with priced options.
- Nothing is presented that is within policy and on track.

### Monthly heartbeat (SIOP)

| Week | What happens | Who | Agent role |
|---|---|---|---|
| 1 | **Facts close and change detection.** What moved since last cycle, and why | Systems + agents | Sentinel builds the change bridge |
| 2 | **Assumption validation.** Owners approve or amend proposed assumption changes, asynchronously | Assumption owners | Challenger compares assumptions to history |
| 3 | **Decision preparation.** The decision calendar is refreshed; 3 to 5 decisions are framed, priced and challenged | Decision analysts | Analyst prices; Challenger red-teams |
| 4 | **Decision forum**, 60 to 90 minutes | CFO chair, COO, Sales lead | Chief of staff drafts the record live |
| +2 days | **Record and hand-off.** Decisions signed; planning-system changes made by their owners; actions assigned | Owners | Chief of staff tracks actions |

**The decision forum agenda** (no status reporting):

1. The outlook versus plan, and what changed (10 min, one page)
2. Envelope health: which policies are close to breach (5 min)
3. Three to five decisions from the decision calendar, each as a tradeoff
   card (45 to 60 min)
4. Policy changes proposed (10 min)
5. Outcome reviews of past decisions (5 min)

Rule: **if an item does not end in a decision, it does not take forum
time.** Information goes out beforehand, asynchronously.

### Quarterly

- Policy review, including risk appetite and service tiers.
- Model calibration: were last quarter's P10/P50/P90 ranges honest?
- Decision retrospective: which decisions paid off, which did not, and why.
- Link to the AOP and external guidance.

---

## 9. Analytics: a small set of standard views

Resist building dashboards. Build **five standard views** that every
decision uses the same way.

| View | Question it answers |
|---|---|
| **Outlook vs plan** | Where are we heading, with what confidence? P10/P50/P90 revenue, margin, inventory, cash; P(plan) |
| **Change bridge** | What changed since last cycle, split into demand, supply, assumptions and our own decisions? |
| **Decision calendar** | What must be decided soon, before its window closes? |
| **Envelope health** | Which policies or thresholds are likely to be breached in the next N months? |
| **Cost of reactivity** | What did decisions made inside the execution window cost us last month (expedites, overtime, displaced customers, revenue timing)? |

### The tradeoff card: one format for every decision

| Section | Content |
|---|---|
| Question | One sentence, with decide-by date and decision owner |
| Options | 2 to 4, always including "do nothing" |
| Consequences | For each option: revenue, gross margin, inventory, E&O exposure, cash, service by customer tier |
| Risk | Where uncertainty matters: range, probability of regret, what would change the answer |
| Assumptions | The three that drive the result, with their owners |
| Challenge | The strongest argument against the recommended option |
| Recommendation | Stated by a named human, not an agent |

Consistency is the point. Executives learn to read one format, and the
discussion moves to the tradeoff instead of the data.

---

## 10. Agents

### Roster

| Agent | Does | Never |
|---|---|---|
| **Sentinel** | Detects material change and envelope breaches; drafts decision requests; maintains the decision calendar | Edits facts; decides |
| **Analyst** | Orchestrates engine runs; builds tradeoff cards in plain language | Produces a number that did not come from an engine run |
| **Challenger** | Tests assumptions against history; finds missing options; argues the other side | Changes assumptions |
| **Chief of staff** | Builds the forum pack; drafts the record; tracks actions; runs outcome reviews | Signs a decision |
| **Intel analyst** | Curates external signals into bounded, reviewed assumption proposals | Moves an assumption beyond its bounds |

The Analyst and Challenger are deliberately separate so no agent grades its
own work.

### Controls

- **Proposal only.** Agents write proposals, drafts and flags. Humans approve
  changes to assumptions, policies and records.
- **Bounded influence.** An agent's proposed assumption change is capped (the
  pattern already used for external intelligence).
- **Traceable numbers.** Every number cites an engine run ID; every run cites
  its facts snapshot, assumption versions and policy versions.
- **No writes to systems of record.** Changes to the demand or supply plan are
  made by their owners in their systems.
- **Logged and reviewable.** Agent actions are logged and sampled in the
  quarterly review.

### Structured objects agents work with

`Signal`, `DecisionRequest`, `Option`, `PricedOption` (with run ID),
`Challenge`, `Decision`, `ActionItem`, `OutcomeReview`, `Assumption`,
`Policy`, `CycleSnapshot`. Without explicit schemas for these, agents
produce prose; with them, agents produce a system.

---

## 11. Team

A lean team. Agents absorb the preparation that traditionally needs a large
SIOP staff.

| Role | Accountability |
|---|---|
| **SIOP process owner** | Owns cadence, policy set, decision calendar, the forum agenda and the operating model itself |
| **Decision analysts** (1 to 3; finance and operations hybrids) | Frame decisions, own action claim sheets, supervise agents, present tradeoff cards |
| **Model steward** | Owns the engine, calibration, change control and data contract |
| **Assumption owners** | Demand Planning, Supply, EMS management, FP&A: validate and sign their assumptions monthly |
| **Decision forum** | CFO (chair), COO, Sales lead; others by agenda |

The skill profile that matters most is the **decision analyst**: someone who
can translate an operational choice into financial consequence and hold the
room to the tradeoff. That role is rare, and it is where AI multiplies
capacity the most.

---

## 12. What changes versus traditional SIOP

| Traditional | This design |
|---|---|
| Monthly batch reconciliation of numbers | Continuous outlook; monthly decisions |
| Five meetings, mostly status | Asynchronous preparation; one decision forum |
| Consensus on a plan | Commitment to decisions and policies |
| Scenarios built by hand, few and late | Options priced on demand; scenarios only where they change a choice |
| Units and utilization | Money, risk and customer service |
| Unwritten rules applied in each crisis | Written, priced, owned policies |
| Decisions in slides and memory | Decision record with outcome reviews |
| Analysts assemble data | Agents assemble; analysts frame and challenge |
| Reactive execution absorbs every change | SOE executes within an envelope; breaches escalate with prices |

---

## 13. A month in the life (illustration)

1. **Day 3.** The Sentinel sees open POs for a critical FPGA slip two weeks
   and AI-segment demand for months 6 to 9 rise 12%. The engine shows
   P(tier-1 shortfall in month 8) rising from 10% to 35%. The decision
   calendar shows the long-lead buy must be committed within 3 weeks.
2. **Week 2.** The Challenger notes the demand increase comes from two
   customers whose upside converted at 55% historically. Demand Planning
   confirms the uncertainty assumption.
3. **Week 3.** The Analyst prices four options: do nothing; commit a 20%
   non-cancellable buy; reserve EMS capacity; both. Tradeoff card: the buy
   protects $38M expected revenue at $6M downside E&O exposure; P(regret)
   25%.
4. **Week 4, forum.** The CFO approves the buy at 15% with a checkpoint in
   6 weeks. The allocation policy is reaffirmed: tier 1 protected.
5. **+2 days.** Supply places the orders in the planning system. The Chief
   of staff logs the decision and schedules the checkpoint and a 90-day
   outcome review.
6. **Meanwhile in SOE.** A tier-3 pull-in request arrives. It fits policy
   and is handled in the weekly SOE meeting. It never reaches the forum.

Nobody built a deck. The forum spent its time on one real tradeoff.

---

## 14. Measures that the system is working

| Measure | Direction |
|---|---|
| Share of spend and commitments decided inside the execution window that could have been decided in SIOP | Down |
| Cost of reactivity (expedite, overtime, displacement) | Down |
| Decisions made before their decide-by date | Up |
| Forum time spent on decisions vs status | Up |
| Calibration: actuals inside the P10 to P90 range about 80% of the time | Honest |
| Decisions with an outcome review | 100% of tier 3 |
| Plan attainment, E&O, service to tier-1 customers | Improve |

---

## 15. Failure modes to design against

| Failure mode | Guardrail |
|---|---|
| SIOP becomes reporting again | Forum rule: no decision, no agenda time |
| Scenario sprawl | Options only for decisions on the calendar; each needs an owner |
| A shadow demand plan | Demand is a read-only fact; uncertainty is owned by Demand Planning |
| A shadow MRP | The governing modeling rule; the planning system stays execution truth |
| Model distrust | One engine, published calibration, traceable runs, visible assumptions |
| Agent noise | Materiality thresholds on signals; the Sentinel tuned against false alarms |
| Policies nobody follows | Breaches measured and reviewed; cost of reactivity made visible |
| Decisions without follow-through | Actions tracked; outcome reviews scheduled at decision time |

---

## 16. Starting small

The full system is the destination, not the first step. A realistic path:

| Phase | Focus | Minimum to start |
|---|---|---|
| **0: Observe** (first 90 days) | Learn which decisions are made, by whom, when, and at what cost | A decision log (even a spreadsheet); a manual cost-of-reactivity estimate; interviews |
| **1: One forum** | Turn the existing capacity meeting into a decision meeting | Tradeoff cards for 2 decision types (capacity reservation, long-lead buys); decision record |
| **2: Write the envelope** | Service tiers, allocation rule, approval thresholds, risk appetite | Priced policy options; SOE escalation path |
| **3: Agents and cadence** | Automate detection, preparation, record, outcome reviews | Structured objects; engine run traceability; data contract in place |
| **4: Learn** | Calibration, retrospectives, policy refinement | Quarterly review loop |

Phases 0 and 1 need no new technology. They need a decision log, a
consistent tradeoff format and a forum willing to use them. The engine and
agents make it scale and make it fast; they are not the starting point.

---

## Open questions to test once inside

1. Where does the committed (SOE) window actually start, and how is it
   governed today?
2. Is there an existing delegation-of-authority matrix for operational
   spend and customer commitments?
3. What unwritten allocation and commit rules does the organization already
   follow?
4. Who would own writing and pricing the policy set?
5. Which existing meeting can become the decision forum, and who must be in
   it for a decision to count?
6. What does the planning system already provide toward the five standard
   views?
