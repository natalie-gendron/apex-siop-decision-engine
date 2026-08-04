You are a senior Python engineer, operations research practitioner, SIOP architect, and Operations Finance leader.

Build a complete, working Python application that demonstrates an AI-enabled SIOP Risk and Scenario Engine for a fictional automated test equipment company.

The fictional company should resemble the operating model of a diversified ATE manufacturer such as Teradyne or Advantest, but do not use confidential, proprietary, scraped, or company-specific internal data. All company names, product names, operating assumptions, and sample data must be synthetic.

The purpose is not to create a toy Monte Carlo demonstration. The purpose is to create a credible Version 1 prototype of an executive decision-support system that could be used by a VP of Global Operations Finance, a Director of SIOP, and a COO.

I want the finished project to show the full workflow from raw operating assumptions through simulation, financial translation, scenario comparison, executive recommendations, and exportable outputs.

Do not stop at planning or pseudocode. Create the complete project, write the files, install or document dependencies, run the application, test it, fix errors, and leave me with a working local prototype.

======================================================================
1. BUSINESS CONTEXT
======================================================================

Assume the fictional company, Apex Test Systems, designs and sells automated test equipment used by semiconductor manufacturers, integrated device manufacturers, fabless semiconductor companies, and outsourced semiconductor assembly and test providers.

Apex Test Systems does not manufacture semiconductor wafers. It designs complex capital equipment and relies heavily on external electronic manufacturing services partners, component suppliers, final integration sites, and global logistics providers.

The business has the following characteristics:

- Volatile semiconductor capital-equipment demand
- Large and sometimes lumpy customer orders
- Customer order pull-ins, push-outs, configuration changes, and cancellations
- Significant exposure to memory, compute, AI accelerators, mobile, automotive, industrial, and mixed-signal markets
- High-value systems with complex bills of material
- Long-lead electronic and electromechanical components
- Constrained items such as FPGAs, custom boards, power supplies, handlers, interface hardware, precision instrumentation, and specialized connectors
- Multiple EMS partners in different regions
- Final system integration, configuration, calibration, and factory acceptance testing
- Customer-specific configurations
- High inventory value and meaningful working-capital exposure
- Revenue timing that depends on manufacturing completion, shipment, installation, and customer acceptance
- Gross-margin variability caused by mix, configuration, supplier cost, expedite premiums, rework, freight, and pricing
- A monthly SIOP process that must reconcile demand, supply, capacity, inventory, revenue, gross margin, and cash

The application should model an 18-month rolling planning horizon at a monthly level, with special focus on the current quarter, next quarter, and full fiscal year.

The executive team needs to answer questions such as:

- What is the probability of achieving the quarterly revenue plan?
- What is the probability of achieving the gross-margin target?
- How much revenue is at risk, and what are the primary drivers?
- Which product families, customers, components, EMS sites, or regions create the greatest risk?
- How likely are customer push-outs or component shortages to move revenue between quarters?
- Can EMS partners support demand upside without creating excessive inventory?
- What happens if AI-related demand increases faster than expected?
- What happens if memory demand recovers later than planned?
- How much long-lead inventory should the company commit to?
- What is the expected inventory, excess-and-obsolete exposure, and working-capital requirement?
- Should the company reserve additional EMS capacity, authorize overtime, expedite components, dual-source production, or accept shipment risk?
- What is the financial trade-off between protecting revenue and increasing inventory?
- What decisions should leadership make during the next SIOP meeting?

======================================================================
2. REQUIRED DELIVERABLE
======================================================================

Create a local Streamlit application with a professional executive interface.

The application must:

1. Generate realistic synthetic sample data
2. Display and allow editing of major planning assumptions
3. Calculate a deterministic baseline plan
4. Run a correlated Monte Carlo simulation
5. Translate operational outcomes into financial outcomes
6. Identify constraints, risks, and opportunities
7. Compare management scenarios
8. Generate a dynamic executive summary
9. Generate evidence-based management recommendations
10. Export results to Excel
11. Include documentation and automated tests
12. Run successfully from a clean local environment

Use Python 3.11 or later.

Use practical, widely available packages. Preferred libraries include:

- streamlit
- pandas
- numpy
- scipy
- plotly
- openpyxl
- pydantic
- scikit-learn, only if it provides clear value
- pytest

Do not require a paid API, external database, cloud service, or proprietary data source.

The application must work without an external large-language-model API. The executive narrative should therefore be generated using transparent, rules-based natural-language generation. Architect the reporting layer so that an optional LLM provider could be added later, but do not make the working prototype dependent on one.

======================================================================
3. PROJECT STRUCTURE
======================================================================

Create a clean, modular repository similar to:

ate-siop-engine/
    app.py
    README.md
    requirements.txt
    pyproject.toml
    .gitignore
    config/
        default_config.yaml
    data/
        generated/
    src/
        __init__.py
        config.py
        models.py
        data_generator.py
        baseline_plan.py
        correlations.py
        simulation.py
        operations.py
        financials.py
        scenarios.py
        sensitivity.py
        recommendations.py
        executive_report.py
        visualizations.py
        exports.py
        validation.py
        utils.py
    tests/
        test_data_generator.py
        test_baseline_plan.py
        test_simulation.py
        test_financials.py
        test_scenarios.py
        test_recommendations.py

Use type hints, docstrings, validation, clear naming, and separation of concerns.

Avoid putting all logic into app.py.

Use dataclasses or Pydantic models for major input and output structures.

Set and expose random seeds so results can be reproduced.

======================================================================
4. SYNTHETIC COMPANY DESIGN
======================================================================

Create a fictional company with approximately:

Product families:
- Eagle Compute Test
- Vector Memory Test
- Horizon Mobility Test
- Atlas Automotive and Industrial Test
- Nexus System-Level Test

End-market demand drivers:
- AI accelerators and high-performance compute
- Memory
- Mobile and consumer
- Automotive
- Industrial and mixed signal

Regions:
- North America
- Taiwan
- South Korea
- Japan
- Europe
- Southeast Asia
- China

Customer groups:
- Large integrated device manufacturers
- Fabless compute leaders
- Memory manufacturers
- Automotive semiconductor suppliers
- OSAT customers
- Other diversified customers

EMS and integration network:
- EMS Americas
- EMS Malaysia
- EMS Taiwan
- EMS Eastern Europe
- Final Integration North America
- Final Integration Asia

The model may include OSAT companies as customers, but Apex Test Systems itself must use EMS partners for equipment manufacturing. Do not treat OSATs as Apex’s manufacturing partners.

Create at least:

- 5 product families
- 8 representative customer accounts or customer groups
- 4 EMS manufacturing sites
- 2 final integration sites
- 25 to 40 critical components or component categories
- 18 planning months
- Multiple demand and operating scenarios

Generate coherent synthetic CSV files automatically on first run and allow users to regenerate them with a selected random seed.

======================================================================
5. INPUT DATA MODEL
======================================================================

Create realistic input tables for the following areas.

A. Demand plan

At the customer, product-family, and month level, include:

- Base unit forecast
- Bookings
- Backlog
- Requested shipment month
- Committed shipment month
- Forecast confidence
- Historical forecast error
- Demand standard deviation
- Upside opportunity
- Downside risk
- Pull-in probability
- Push-out probability
- Cancellation probability
- Customer priority
- Market segment
- Region
- Average selling price
- Configuration complexity
- New-product flag
- Customer site-readiness probability
- Revenue-recognition method or timing assumption

Demand should be lumpy rather than smoothly distributed.

Model common demand factors so related markets move together. For example:

- AI compute demand affects Eagle Compute Test and Nexus System-Level Test
- Memory-cycle recovery affects Vector Memory Test
- Mobile demand affects Horizon Mobility Test
- Automotive demand affects Atlas Automotive and Industrial Test
- A general semiconductor capital-spending factor affects all product families

B. Product and bill-of-material assumptions

At the product-family level, include:

- Standard selling price
- Standard material cost
- Standard EMS conversion cost
- Standard integration and test cost
- Standard freight cost
- Standard warranty reserve
- Typical build cycle time
- Final integration time
- Factory acceptance test time
- Installation time
- Customer acceptance lag
- Configuration complexity
- Base first-pass yield
- Rework probability
- Scrap probability
- Critical component requirements
- Product-specific EMS qualification
- Alternate-source eligibility

C. Critical components

Include:

- Component category
- Supplier
- Supplier region
- Unit cost
- Lead time
- Lead-time variability
- On-hand inventory
- Open purchase orders
- Minimum order quantity
- Safety stock
- Allocation risk
- Disruption probability
- Expedite availability
- Expedite premium
- Alternate source
- Alternate-source qualification lag
- Products consuming the component
- Per-system usage quantity
- Obsolescence risk

Examples may include:

- FPGA
- Precision analog instrumentation
- Custom PCB assembly
- Power distribution module
- High-speed interconnect
- Thermal-management assembly
- Industrial computer
- Handler interface
- Probe interface hardware
- Precision mechanical frame
- Specialized cable set
- Custom power supply

D. EMS partner assumptions

For each EMS site and month, include:

- Available production capacity in standard-equivalent systems
- Reserved capacity
- Flexible capacity
- Maximum overtime capacity
- Capacity cost
- Overtime premium
- Capacity-reservation fee
- Manufacturing cycle time
- Schedule adherence
- First-pass yield
- Rework rate
- Scrap rate
- Labor availability
- Quality-escape probability
- Regional disruption probability
- Logistics lead time
- Eligible product families
- Ramp constraints
- Minimum production lot
- Currency exposure, represented simply as cost variability

E. Final integration assumptions

Include:

- Monthly integration capacity
- Calibration capacity
- Factory acceptance test capacity
- Labor availability
- First-pass completion rate
- Rework time
- Product-family eligibility
- Regional shipping lanes
- Installation resource capacity
- Customer acceptance timing

F. Financial assumptions

Include:

- Revenue plan by month and quarter
- Gross-margin target
- Inventory target
- Inventory-turn target
- Working-capital target
- Operating-expense assumptions
- Depreciation
- Capital expenditures
- Tax rate
- Cash conversion assumptions
- Freight rates
- Expedite premiums
- Price variability
- Purchase-price variance
- Foreign-exchange cost variability
- Warranty reserve
- E&O reserve rules
- Revenue-recognition timing
- Target EBITDA or operating-income plan

Use U.S. dollars throughout.

======================================================================
6. BASELINE PLANNING LOGIC
======================================================================

Before running Monte Carlo, build a deterministic monthly baseline plan.

The baseline must:

- Start with beginning backlog and forecast
- Convert demand into planned system builds
- Allocate builds across qualified EMS sites
- Consume component inventory and purchase-order receipts
- Respect component availability
- Respect EMS capacity
- Respect final integration and factory acceptance test capacity
- Respect installation capacity where applicable
- Calculate planned shipments
- Calculate expected revenue-recognition month
- Track unmet demand and backlog aging
- Track inventory by component, work in process, and finished goods
- Calculate revenue, cost of goods sold, gross margin, operating income, EBITDA proxy, and cash-flow proxy
- Calculate inventory turns and working-capital investment
- Identify baseline constraints

Use a transparent allocation heuristic for Version 1. For example:

1. Prioritize firm backlog over forecast
2. Prioritize higher customer priority
3. Prioritize earlier requested shipment date
4. Prioritize higher contribution margin when other factors are equal
5. Allocate only to qualified sites
6. Respect critical-component and capacity constraints

Document the heuristic and its limitations.

Do not over-engineer a full mixed-integer optimizer unless it can be added cleanly as an optional stretch feature. A working, understandable heuristic is preferable to a fragile optimization model.

======================================================================
7. MONTE CARLO SIMULATION
======================================================================

Implement a correlated Monte Carlo simulation.

The user should be able to select simulation count from:

- 1,000 for quick mode
- 5,000 for standard mode
- 10,000 or more for detailed mode

The architecture should be vectorized and efficient. Do not default to 100,000 simulations if that makes the interactive application unusably slow.

Model uncertainty in at least the following:

Demand:
- Market-cycle demand
- Customer-specific demand
- Pull-ins
- Push-outs
- Cancellations
- New-product ramp
- Customer site readiness
- Product mix
- ASP

Components:
- Supplier disruption
- Delivery lateness
- Lead-time variation
- Allocation
- Yield loss at component or assembly level
- Purchase-price variance
- Expedite availability and cost

EMS operations:
- Available capacity
- Schedule adherence
- Labor availability
- First-pass yield
- Rework
- Scrap
- Cycle time
- Regional disruption
- Logistics delay

Final integration and acceptance:
- Integration capacity
- Factory acceptance test delay
- Installation delay
- Customer acceptance delay

Finance:
- Selling price
- Material cost
- EMS conversion cost
- Freight
- Expedite premium
- Warranty cost
- FX-related cost variance

Use appropriate bounded distributions rather than naive unbounded normal distributions when variables cannot be negative or exceed logical limits.

Potential distribution choices:

- Beta for probabilities and bounded yields
- Triangular or lognormal for lead times and costs
- Bernoulli for discrete disruption events
- Poisson or negative binomial for lumpy order events
- Truncated normal for selected continuous variables
- Categorical distributions for discrete timing shifts

Implement common-factor correlations rather than treating every variable as independent.

At minimum, create factors for:

- Global semiconductor capital spending
- AI and high-performance-compute demand
- Memory cycle
- Mobile cycle
- Automotive and industrial cycle
- Electronic-component supply tightness
- Regional logistics disruption
- EMS labor and execution conditions

Examples of required relationships:

- AI-demand upside increases demand for compute and system-level test products
- Memory recovery increases memory-test demand
- Strong total demand increases EMS utilization
- Higher utilization can reduce schedule adherence and first-pass yield
- Component shortages increase lead times and expedite costs
- Longer component lead times can increase raw-material commitments while delaying finished-system output
- Customer pull-ins increase near-term capacity pressure
- Customer push-outs may create finished-goods inventory
- Lower first-pass yield increases rework, cycle time, and unit cost
- Expediting can reduce some shipment risk but raises cost
- Demand mix changes affect gross margin
- Customer site-readiness delays can shift revenue even when systems are complete

Create a documented correlation or factor-loading configuration that can be viewed in the application.

Ensure that correlations do not produce impossible values.

======================================================================
8. SIMULATION OUTPUTS
======================================================================

Capture simulation outcomes at the company, product-family, customer-group, site, month, quarter, and fiscal-year levels where practical.

Required company-level outputs include:

- Revenue distribution
- Gross-profit distribution
- Gross-margin distribution
- Operating-income distribution
- EBITDA proxy distribution
- Operating cash-flow proxy distribution
- Ending inventory distribution
- Average inventory
- Inventory turns
- Working capital
- Revenue at risk
- Margin at risk
- Probability of achieving revenue plan
- Probability of achieving gross-margin target
- Probability of exceeding inventory target
- Probability of stockout
- Probability of excess inventory
- Probability of missing a customer commitment
- Expected backlog
- Expected past-due backlog
- Expected expedite cost
- Expected rework cost
- Expected E&O reserve
- EMS utilization
- Final integration utilization
- Capacity shortfall
- Component-shortage incidence

Provide:

- Mean
- Median
- Standard deviation
- 5th percentile
- 25th percentile
- 75th percentile
- 95th percentile
- Confidence ranges
- Downside-at-risk values

Clearly distinguish:

- Expected result
- Plan
- Downside case
- Upside case
- Probability of meeting plan

======================================================================
9. SCENARIO ENGINE
======================================================================

Create a scenario manager that lets the user change assumptions and rerun the model.

Include prebuilt scenarios:

1. Base Case
2. AI Demand Surge
   - Compute and system-level demand increases
   - Select customer pull-ins increase
   - FPGA and high-speed interconnect pressure increases
3. Memory Recovery
   - Memory-test demand improves over several months
4. Memory Recovery Delay
   - Memory demand remains weak longer than expected
5. Major Customer Push-Out
   - A large customer shifts a meaningful order into the following quarter
6. Critical FPGA Shortage
   - Lead times increase
   - Receipts are delayed
   - Expedite premiums rise
7. EMS Malaysia Disruption
   - Capacity and schedule adherence decline temporarily
8. EMS Capacity Reservation
   - The company pays to reserve additional capacity
9. Overtime and Expedite Response
   - Additional output is enabled at higher cost
10. Dual-Source Qualification
   - Alternate supply becomes available after a qualification lag
11. Inventory Reduction Initiative
   - Purchasing is constrained and safety stocks are reduced
12. Customer Site-Readiness Delay
   - Completed systems wait longer for installation or acceptance

Allow a custom scenario with controls for:

- Demand by market or product family
- Pull-in probability
- Push-out probability
- Cancellation probability
- ASP
- Component lead time
- Supplier disruption probability
- EMS capacity
- EMS yield
- Schedule adherence
- Integration capacity
- Freight cost
- Expedite premium
- Safety-stock policy
- Capacity reservation
- Customer acceptance lag

The application should compare any selected scenario with the base case and show:

- Change in expected revenue
- Change in probability of hitting revenue plan
- Change in gross margin
- Change in inventory
- Change in working capital
- Change in expected expedite spending
- Change in service level
- Change in major risk drivers
- Risk reduced per dollar of incremental cost
- Incremental expected value where appropriate

======================================================================
10. MANAGEMENT ACTIONS AND TRADE-OFFS
======================================================================

Create a management-action comparison module.

Evaluate actions such as:

- Reserve additional EMS capacity
- Authorize overtime
- Expedite a critical component
- Increase selected component safety stock
- Reduce or cancel selected purchase commitments
- Shift production between EMS sites
- Prioritize high-value or high-margin configurations
- Dual-source a constrained component
- Pre-build standard subassemblies
- Delay lower-priority builds
- Reallocate scarce components among product families
- Increase installation resources
- Accept customer shipment risk rather than incur extraordinary cost

For each action, calculate where possible:

- Incremental cost
- Expected revenue protected
- Expected gross profit protected
- Reduction in probability of missing plan
- Change in inventory
- Change in working capital
- Change in customer service level
- Expected value
- Key assumptions
- Potential unintended consequences

Do not recommend actions merely because they improve one metric. Account for financial and operational trade-offs.

======================================================================
11. SENSITIVITY AND RISK-DRIVER ANALYSIS
======================================================================

Identify the largest drivers of outcomes.

Implement at least one robust sensitivity method, such as:

- Spearman rank correlation between sampled inputs and outcomes
- Standardized regression coefficients
- Permutation importance using a simple surrogate model

Use it to rank drivers of:

- Revenue
- Gross margin
- Ending inventory
- Working capital
- Customer-service performance

Include a tornado chart or ranked-driver chart.

Also identify:

- Most frequently binding components
- Most frequently constrained EMS sites
- Product families with the most revenue at risk
- Customer groups with the greatest timing risk
- Months with the greatest capacity risk
- Factors that most often cause revenue to shift between quarters

Avoid implying causality when the analysis only shows association. Label the method clearly.

======================================================================
12. EXECUTIVE DASHBOARD
======================================================================

Create a polished Streamlit dashboard designed for executives.

Recommended pages or tabs:

1. Executive Overview
2. Demand and Backlog
3. Supply and Components
4. EMS and Integration Capacity
5. Financial Outcomes
6. Risk Drivers
7. Scenario Comparison
8. Management Actions
9. Assumptions and Data
10. Export and Methodology

The Executive Overview should include:

- Expected quarterly revenue versus plan
- Probability of meeting revenue plan
- Expected gross margin versus target
- Probability of meeting margin target
- Expected inventory versus target
- Expected working capital
- Expected customer-service level
- Revenue at risk
- Top three risks
- Top three recommended actions
- Base versus selected-scenario comparison

Use executive-quality Plotly visuals, including:

- Revenue probability distribution with plan line
- Gross-margin probability distribution with target line
- Quarterly fan chart
- Inventory trajectory with uncertainty band
- Revenue bridge or waterfall
- Scenario comparison chart
- Capacity-utilization heat map
- Component-risk heat map
- Risk matrix
- Tornado or sensitivity chart
- Product-family revenue-at-risk chart
- EMS capacity versus demand chart
- Backlog-aging chart

Use a clean and restrained professional design.

Do not use excessive decoration, bright colors, unnecessary 3D charts, or clutter.

Use consistent units and labels, including:

- $M for major financial figures
- Percentages with one decimal place
- Units or systems where appropriate
- Clear distinction between monthly, quarterly, and annual views

======================================================================
13. DYNAMIC EXECUTIVE SUMMARY
======================================================================

Generate a concise, dynamic executive summary based on actual simulation results.

Do not hardcode specific percentages or conclusions.

The summary should sound like a strong VP of Global Operations Finance or Director of SIOP briefing a COO.

Use an outcome-first structure:

1. Overall outlook
2. Revenue and margin risk
3. Operational constraints
4. Inventory and cash implications
5. Recommended decisions

Example style only:

“The current plan has a 68% probability of achieving quarterly revenue and a 57% probability of achieving the gross-margin target. Expected revenue is $12.4M below plan at the median, with most downside concentrated in Eagle Compute Test and Vector Memory Test.

The largest operational risk is FPGA availability, followed by integration capacity in the final month of the quarter. Customer push-outs create additional timing risk and may increase finished-goods inventory by approximately $18M.

Reserving incremental EMS capacity alone provides limited benefit because component supply remains binding. The strongest combined response is to expedite selected FPGA receipts, shift eligible builds to EMS Taiwan, and add temporary final-integration capacity. In simulation, this combination improves the probability of achieving revenue plan by 14 percentage points at an expected incremental cost of $3.1M.”

The actual narrative must be generated from the model outputs.

The report should:

- Mention the most decision-relevant numbers
- Explain what changed versus the base case
- Separate facts from recommendations
- Avoid false precision
- Avoid generic statements
- Avoid calling every variance “significant”
- State uncertainty clearly
- Identify when no action has positive expected value
- Keep the main summary to approximately 250 to 450 words
- Provide an optional more detailed appendix narrative

Build this with deterministic rules and templates so no external AI API is needed.

Add an abstraction layer that could later support an optional LLM implementation.

======================================================================
14. RECOMMENDATION ENGINE
======================================================================

Create a transparent recommendation engine.

Each recommendation must be tied to:

- A detected risk or opportunity
- A measurable threshold
- A modeled management action
- Quantified expected benefits
- Quantified costs or trade-offs
- Confidence or evidence level

Example recommendation structure:

Recommendation:
Expedite FPGA receipts for Eagle Compute Test.

Why:
FPGA availability is the most frequently binding constraint and contributes to an estimated $22M of expected quarterly revenue risk.

Modeled impact:
- Revenue-plan attainment probability: +9 percentage points
- Expected revenue: +$8.5M
- Expected gross profit: +$3.9M
- Incremental expedite cost: $1.2M
- Ending inventory: +$2.1M

Caveat:
Benefit declines materially if the major customer push-out also occurs.

Recommendations should be ranked based on a combination of:

- Expected value
- Revenue or gross profit protected
- Probability improvement
- Cash requirement
- Execution feasibility
- Risk reduction
- Strategic importance

Make scoring assumptions visible.

Do not present a recommendation if the modeled benefit is unsupported or immaterial.

======================================================================
15. FINANCIAL LOGIC
======================================================================

Translate operations into finance carefully.

At minimum, calculate:

Revenue:
- Units recognized multiplied by realized ASP
- Recognition timing based on shipment, installation, or acceptance assumptions

Cost of goods sold:
- Material
- EMS conversion
- Integration and test
- Freight
- Expedite premiums
- Rework
- Scrap
- Warranty reserve
- Other product-level variable costs

Gross profit and gross margin

Operating income:
- Gross profit less operating expenses

EBITDA proxy:
- Operating income plus depreciation

Inventory:
- Raw materials
- Work in process
- Finished goods
- Completed systems awaiting shipment
- Completed systems awaiting acceptance

Working capital:
- Inventory
- Simplified receivables
- Simplified payables

Cash-flow proxy:
- EBITDA proxy
- Less working-capital increase
- Less capital expenditure
- Less cash taxes where applicable

E&O exposure:
- Excess component inventory
- Aging finished systems
- Product or component obsolescence risk
- Scenario-specific reserve logic

Explain all simplifications in the methodology page and README.

======================================================================
16. DATA VALIDATION AND ERROR HANDLING
======================================================================

Validate inputs and fail gracefully.

Examples:

- Probabilities must remain between 0 and 1
- Yields must remain within logical bounds
- Costs and capacities cannot be negative
- Dates must align to the planning horizon
- Product-to-site qualification must be valid
- Component usage must reference valid products
- Correlation matrices must be positive semidefinite or repaired safely
- Scenario changes must not create impossible values

Display user-friendly validation messages in Streamlit.

Log technical details separately.

======================================================================
17. PERFORMANCE
======================================================================

The standard simulation should finish in a reasonable time on a normal laptop.

Target:

- 1,000 simulations in quick mode within several seconds
- 5,000 simulations in standard mode within approximately 30 seconds where practical

Use vectorization, efficient arrays, and caching.

Use Streamlit caching appropriately for:

- Synthetic data
- Baseline calculations
- Simulation results keyed to assumptions and random seed

Provide visible progress indicators for longer runs.

Do not silently freeze the interface.

If full monthly, product, customer, component, and site granularity makes the Monte Carlo engine too slow, use a thoughtful hybrid approach:

- Preserve detailed deterministic planning
- Simulate the most decision-relevant aggregates and bottlenecks
- Document the simplification
- Maintain reconciliation to the baseline model

A fast, credible prototype is preferable to an unusable model.

======================================================================
18. EXPORTS
======================================================================

Create an Excel export containing:

- Executive summary
- KPI summary
- Scenario comparison
- Revenue distribution statistics
- Margin distribution statistics
- Inventory and working-capital statistics
- Monthly baseline plan
- Product-family results
- Customer-group results
- EMS capacity results
- Component-risk results
- Sensitivity rankings
- Recommendations
- Assumptions
- Methodology notes

Use professional formatting:

- Clear sheet names
- Frozen headers
- Appropriate number formats
- Filters
- Column widths
- Section headers
- Conditional formatting where useful

Also allow CSV downloads of underlying tables.

A PowerPoint export is optional and should only be attempted after the working application, testing, and Excel export are complete.

======================================================================
19. TESTING
======================================================================

Write automated tests.

At minimum, test:

- Synthetic data generation is reproducible
- Generated data meets validation rules
- Baseline supply never exceeds available components or capacity
- Financial statements reconcile arithmetically
- Simulation results are reproducible with the same seed
- Simulation outputs change when scenarios change
- Probabilities remain within valid bounds
- Percentiles are ordered correctly
- Revenue recognition follows timing rules
- Scenario comparisons reconcile to base results
- Recommendations reference actual modeled outcomes
- Excel export is created successfully

Run the tests and fix all failures.

Include the final test results in the README or a completion summary.

======================================================================
20. README
======================================================================

Create a comprehensive README that includes:

- Purpose of the application
- Business context
- Architecture
- Project structure
- Installation instructions
- Commands to run the app
- Commands to run tests
- Explanation of the synthetic data
- Explanation of the baseline planning logic
- Explanation of Monte Carlo methodology
- Explanation of correlations
- Explanation of financial calculations
- Explanation of scenario analysis
- Known limitations
- Potential Version 2 enhancements

Include exact commands, such as:

python -m venv .venv

Windows:
.venv\Scripts\activate

macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py

pytest

======================================================================
21. VERSION 1 SCOPE PRIORITIES
======================================================================

Implement in this priority order:

Priority 1:
- Working repository
- Synthetic data
- Deterministic baseline plan
- Monte Carlo engine
- Core financial outputs
- Streamlit executive dashboard
- Scenario comparison
- Dynamic executive summary
- Tests
- README

Priority 2:
- Sensitivity analysis
- Recommendation engine
- Management-action comparison
- Excel export
- Enhanced visualizations

Priority 3:
- Optional optimization
- Optional PowerPoint export
- Optional LLM integration interface
- Bayesian updating
- Machine-learning distribution fitting
- More advanced digital-twin functionality

Do not sacrifice a working Priority 1 application to pursue stretch goals.

======================================================================
22. DESIGN PRINCIPLES
======================================================================

Follow these principles:

- Build a decision tool, not a statistical science project.
- Tie operational uncertainty to financial outcomes.
- Make assumptions visible.
- Make recommendations traceable.
- Use realistic ATE and EMS terminology.
- Do not confuse EMS manufacturing partners with OSAT customers.
- Do not model Apex Test Systems as a wafer manufacturer.
- Do not use fab utilization, wafer starts, wafer yield, or internal wafer-fab concepts as Apex operating variables.
- Model EMS capacity, component supply, system integration, calibration, factory acceptance testing, installation, and customer acceptance.
- Treat revenue timing as a major source of uncertainty.
- Treat customer pull-ins and push-outs as important SIOP events.
- Represent market demand with common correlated factors.
- Quantify the trade-off between revenue protection and working-capital investment.
- Prefer explainability over unnecessary complexity.
- Avoid false precision.
- Keep the executive interface focused on decisions.
- Ensure that all conclusions come from actual calculations.
- Clearly label synthetic data and model limitations.

======================================================================
23. DEFINITION OF DONE
======================================================================

The project is complete only when:

- The repository and all required files exist
- Dependencies are documented
- Synthetic data can be generated
- The baseline plan runs
- The Monte Carlo simulation runs
- At least five prebuilt scenarios run
- The dashboard displays results without errors
- The executive summary changes dynamically with results
- Scenario comparisons work
- Recommendations cite modeled impacts
- Excel export works
- Automated tests pass
- The README explains how to run everything
- You have run the application and corrected errors

At completion, provide:

1. A concise summary of what was built
2. The final project structure
3. Exact commands to install and run it
4. Test results
5. Any known limitations
6. The best next enhancements

Begin by creating the project structure and a short implementation plan. Then proceed directly into building the working application. Do not wait for additional confirmation unless a genuinely blocking technical issue arises.