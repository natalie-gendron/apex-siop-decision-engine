"""Demand Confidence engine: determinism, structure, and simulation coupling."""
from __future__ import annotations

import numpy as np

from src.market_intelligence import (
    CONFIDENCE_LEVELS,
    CONFIDENCE_SIM_PARAMS,
    assumptions_in_business_language,
    build_demand_confidence,
    confidence_mapping_table,
    merge_confidence_params,
)
from src.simulation import fiscal_year, run_simulation


def test_confidence_deterministic_per_seed(data, config):
    a = build_demand_confidence(data, config)
    b = build_demand_confidence(data, config)
    assert a.score == b.score and a.level == b.level
    assert a.narrative == b.narrative
    assert [s.current for s in a.signals] == [s.current for s in b.signals]


def test_confidence_structure(data, config):
    dc = build_demand_confidence(data, config)
    assert dc.level in CONFIDENCE_LEVELS
    assert 0 <= dc.score <= 100
    assert len(dc.signals) == 8
    assert abs(sum(s.weight for s in dc.signals) - 1.0) < 1e-9
    assert len(dc.external) == 10
    assert all(e.stance in ("Favorable", "Neutral", "Watch", "Unfavorable")
               for e in dc.external)
    assert len(dc.customers) >= 8
    assert all(c.rating in ("High", "Moderate", "Low") for c in dc.customers)
    assert all(c.reasons for c in dc.customers)
    assert 0 <= dc.forecast_confidence_pct <= 100
    assert len(dc.narrative.split()) > 20


def test_lower_confidence_widens_distribution(data, config, baseline):
    """The core integration contract: lower confidence -> wider demand
    distribution, fatter downside tail, lower plan attainment."""
    results = {}
    for lvl in ("Very High", "Moderate", "Very Low"):
        r = run_simulation(data, config, baseline,
                           params=dict(CONFIDENCE_SIM_PARAMS[lvl]),
                           n_sims=1500, seed=42)
        fy = fiscal_year(r.revenue)
        results[lvl] = (fy.std(), np.percentile(fy, 5),
                        (fy >= baseline.revenue_plan_q[:4].sum()).mean())
    assert results["Very High"][0] < results["Moderate"][0] < results["Very Low"][0]
    assert results["Very High"][1] > results["Moderate"][1] > results["Very Low"][1]
    assert results["Very High"][2] > results["Moderate"][2] > results["Very Low"][2]


def test_merge_confidence_params():
    conf = {"demand_sigma_mult": 1.3, "pushout_prob_add": 0.04,
            "cancel_prob_mult": 1.3}
    scen = {"pushout_prob_add": 0.05, "cancel_prob_mult": 2.0,
            "freight_mult": 1.5}
    merged = merge_confidence_params(conf, scen)
    assert merged["pushout_prob_add"] == 0.09          # adds sum
    assert merged["cancel_prob_mult"] == 2.0 * 1.3     # mults multiply
    assert merged["demand_sigma_mult"] == 1.3          # confidence-only key kept
    assert merged["freight_mult"] == 1.5               # scenario-only key kept


def test_assumptions_table_business_language(data, config):
    dc = build_demand_confidence(data, config)
    df = assumptions_in_business_language(data, config, dc)
    assert len(df) >= 6
    assert set(df.columns) == {"Assumption", "Current setting", "What it means"}
    text = " ".join(df["What it means"])
    for banned in ("sigma", "lognormal", "np.", "stddev"):
        assert banned not in text.lower()


CURATED_CSV = """topic,stance,summary,sources,proposed_impact_pts,as_of
AI Infrastructure Spending,Favorable,Capex is strong.,src-a (a.com) ; src-b (b.com),9.0,2026-07-31
Automotive Outlook,Unfavorable,Demand weak.,src-c (c.com),-2.0,2026-07-31
"""


def test_curated_external_parse_and_clipping(data, config):
    from src.market_intelligence import MAX_TOPIC_IMPACT_PTS, parse_curated_external
    intel = parse_curated_external(CURATED_CSV)
    assert len(intel) == 2
    assert intel[0].sources == ["src-a (a.com)", "src-b (b.com)"]
    # a single topic cannot exceed the per-topic cap (9.0 -> 3.0)
    assert intel[0].proposed_impact_pts == MAX_TOPIC_IMPACT_PTS
    dc = build_demand_confidence(data, config, external=intel)
    assert dc.external_source == "curated"
    assert dc.external_adjustment_pts == 1.0   # +3.0 clipped, -2.0
    base = build_demand_confidence(data, config)
    assert abs((dc.score - base.score)
               - (dc.external_adjustment_pts - base.external_adjustment_pts)) < 0.11


def test_curated_external_rejects_bad_stance():
    import pytest as _pytest
    from src.market_intelligence import parse_curated_external
    bad = "topic,stance,summary\nAI,Amazing,text\n"
    with _pytest.raises(ValueError):
        parse_curated_external(bad)


def test_confidence_levels_all_mapped():
    assert set(CONFIDENCE_SIM_PARAMS) == set(CONFIDENCE_LEVELS)
    mults = [CONFIDENCE_SIM_PARAMS[lvl]["demand_sigma_mult"]
             for lvl in CONFIDENCE_LEVELS]
    assert mults == sorted(mults)  # monotonically wider as confidence falls


def test_confidence_mapping_table_matches_sim_params():
    df = confidence_mapping_table("Low")
    assert len(df) == len(CONFIDENCE_LEVELS)
    for _, row in df.iterrows():
        lvl = (row["Confidence level"].removeprefix("→ ")
               .split(" — ")[0])
        p = CONFIDENCE_SIM_PARAMS[lvl]
        assert row["Demand volatility"] == f"×{p['demand_sigma_mult']:.2f}"
        assert row["Push-out probability"] == \
            f"{p['pushout_prob_add'] * 100:+.0f} pts"
        assert row["Cancellations"] == f"×{p['cancel_prob_mult']:.2f}"
    marked = [r for r in df["Confidence level"] if r.startswith("→ ")]
    assert marked == ["→ Low"]
    assert any(r.endswith("— no adjustment") for r in df["Confidence level"])
    assert "What that means" in df.columns
    compact = confidence_mapping_table(None, plain_language=False)
    assert "What that means" not in compact.columns
    assert not any(r.startswith("→ ") for r in compact["Confidence level"])
