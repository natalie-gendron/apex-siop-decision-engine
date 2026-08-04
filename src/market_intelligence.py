"""Market Intelligence & Demand Confidence.

Synthesizes demand-health signals, external market intelligence and customer
confidence assessments into a single executive Demand Confidence level, and
maps that level to Monte Carlo assumption adjustments so confidence directly
drives simulation uncertainty.

Everything here is synthetic and rules-based: signals are seeded from the
input data seed (reproducible), commentary comes from templates, and the
confidence-to-simulation mapping is a visible, documented table — no black box.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .models import InputData

HISTORY_MONTHS = 12  # trailing months of signal history shown to executives

CONFIDENCE_LEVELS = ["Very High", "High", "Moderate", "Low", "Very Low"]

# Demand Confidence -> simulation assumption adjustments.
# demand_sigma_mult scales demand volatility; pushout_prob_add raises the
# monthly probability that orders slip; cancel_prob_mult scales cancellations.
CONFIDENCE_SIM_PARAMS: dict[str, dict[str, float]] = {
    "Very High": {"demand_sigma_mult": 0.70, "pushout_prob_add": -0.02,
                  "cancel_prob_mult": 0.80},
    "High":      {"demand_sigma_mult": 0.85, "pushout_prob_add": -0.01,
                  "cancel_prob_mult": 0.90},
    "Moderate":  {"demand_sigma_mult": 1.00, "pushout_prob_add": 0.00,
                  "cancel_prob_mult": 1.00},
    "Low":       {"demand_sigma_mult": 1.30, "pushout_prob_add": 0.04,
                  "cancel_prob_mult": 1.30},
    "Very Low":  {"demand_sigma_mult": 1.60, "pushout_prob_add": 0.08,
                  "cancel_prob_mult": 1.60},
}

CONFIDENCE_BUSINESS_MEANING: dict[str, str] = {
    "Very High": "Demand variance narrowed ~30%; push-outs and cancellations "
                 "below normal. Downside tail is small.",
    "High": "Demand variance narrowed ~15%; slightly fewer push-outs than normal.",
    "Moderate": "Baseline variance — the plan's stated volatility assumptions "
                "apply unchanged.",
    "Low": "Demand variance widened ~30%; push-outs 4 points more likely and "
           "cancellations 30% more frequent. Downside tail grows materially.",
    "Very Low": "Demand variance widened ~60%; push-outs 8 points more likely and "
                "cancellations 60% more frequent. Severe revenue shortfalls "
                "become plausible.",
}


@dataclass
class MarketSignal:
    name: str
    current: float
    unit: str                 # "%", "x", "count", "pts"
    trend: float              # change vs 3 months ago (same unit)
    score: float              # 0-100 contribution to confidence (higher = better)
    weight: float             # weight in the composite
    comment: str
    history: list[float] = field(default_factory=list)


@dataclass
class ExternalIntel:
    topic: str
    stance: str               # Favorable / Neutral / Watch / Unfavorable
    summary: str
    sources: list[str] = field(default_factory=list)   # citations (curated feed)
    proposed_impact_pts: float | None = None           # proposed confidence-score impact
    as_of: str = ""                                    # date of the assessment


@dataclass
class CustomerConfidence:
    customer: str
    group: str
    horizon_revenue: float
    rating: str               # High / Moderate / Low
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class DemandConfidence:
    score: float              # 0-100 composite
    level: str                # Very High .. Very Low
    narrative: str
    signals: list[MarketSignal]
    external: list[ExternalIntel]
    customers: list[CustomerConfidence]
    forecast_confidence_pct: float
    forecast_confidence_label: str
    forecast_confidence_reason: str
    external_adjustment_pts: float = 0.0   # external intel's contribution to the score
    external_source: str = "synthetic"     # "synthetic" or "curated"

    @property
    def sim_params(self) -> dict[str, float]:
        return dict(CONFIDENCE_SIM_PARAMS[self.level])


def _level_from_score(score: float) -> str:
    if score >= 78:
        return "Very High"
    if score >= 64:
        return "High"
    if score >= 46:
        return "Moderate"
    if score >= 32:
        return "Low"
    return "Very Low"


def _walk(rng: np.random.Generator, start: float, drift: float, vol: float,
          lo: float, hi: float) -> np.ndarray:
    """Bounded random walk for signal history."""
    steps = rng.normal(drift, vol, HISTORY_MONTHS - 1)
    path = start + np.concatenate([[0.0], np.cumsum(steps)])
    return np.clip(path, lo, hi)


def _score(value: float, bad: float, good: float) -> float:
    """Linear 0-100 score between a bad and good anchor."""
    return float(np.clip((value - bad) / (good - bad), 0, 1) * 100)


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def _build_signals(data: InputData, rng: np.random.Generator) -> list[MarketSignal]:
    dem = data.demand
    push_base = float(dem["push_out_prob"].mean())
    cancel_base = float(dem["cancel_prob"].mean())
    fc_err = float(dem["hist_forecast_error"].mean())

    signals: list[MarketSignal] = []

    btb = _walk(rng, rng.uniform(0.94, 1.10), 0.0, 0.025, 0.80, 1.25)
    signals.append(MarketSignal(
        "Book-to-Bill", round(btb[-1], 2), "x", round(btb[-1] - btb[-4], 2),
        _score(btb[-1], 0.85, 1.12), 0.16,
        "Bookings versus billings over the trailing month. Above 1.0 the order "
        "book is growing.", list(np.round(btb, 3))))

    mom = _walk(rng, rng.uniform(-2, 4), 0.0, 1.6, -12, 15)
    signals.append(MarketSignal(
        "Booking Momentum", round(mom[-1], 1), "%", round(mom[-1] - mom[-4], 1),
        _score(mom[-1], -8, 10), 0.14,
        "Three-month bookings growth rate.", list(np.round(mom, 2))))

    blg = _walk(rng, rng.uniform(-1, 3), 0.0, 1.2, -8, 10)
    signals.append(MarketSignal(
        "Backlog Growth", round(blg[-1], 1), "%", round(blg[-1] - blg[-4], 1),
        _score(blg[-1], -6, 8), 0.10,
        "Month-over-month change in total backlog.", list(np.round(blg, 2))))

    aging = _walk(rng, rng.uniform(10, 20), 0.1, 1.0, 4, 35)
    signals.append(MarketSignal(
        "Backlog Aging", round(aging[-1], 1), "%", round(aging[-1] - aging[-4], 1),
        _score(-aging[-1], -30, -8), 0.06,
        "Share of backlog past its originally committed month. Lower is healthier.",
        list(np.round(aging, 2))))

    push = _walk(rng, push_base * 100, rng.uniform(-0.15, 0.35), 0.5,
                 3, 25)
    signals.append(MarketSignal(
        "Customer Push-Out Rate", round(push[-1], 1), "%",
        round(push[-1] - push[-4], 1),
        _score(-push[-1], -20, -6), 0.20,
        "Share of committed orders slipping to a later month. The single "
        "strongest early-warning signal of demand softness.",
        list(np.round(push, 2))))

    canc = _walk(rng, cancel_base * 100, rng.uniform(-0.05, 0.1), 0.2, 0.2, 8)
    signals.append(MarketSignal(
        "Cancellation Rate", round(canc[-1], 2), "%",
        round(canc[-1] - canc[-4], 2),
        _score(-canc[-1], -6, -0.5), 0.12,
        "Share of orders cancelled per month.", list(np.round(canc, 2))))

    exp = _walk(rng, rng.uniform(6, 12), 0.0, 1.2, 2, 25)
    signals.append(MarketSignal(
        "Expedite Requests", round(exp[-1], 0), "count",
        round(exp[-1] - exp[-4], 0),
        _score(exp[-1], 2, 18), 0.04,
        "Customer requests to accelerate deliveries. Rising expedites signal "
        "demand urgency (and supply stress).", list(np.round(exp, 1))))

    acc = _walk(rng, (1 - fc_err) * 100, rng.uniform(-0.2, 0.2), 0.7, 60, 95)
    signals.append(MarketSignal(
        "Forecast Accuracy", round(acc[-1], 1), "%", round(acc[-1] - acc[-4], 1),
        _score(acc[-1], 65, 92), 0.18,
        "One-month-ahead unit forecast accuracy, trailing average.",
        list(np.round(acc, 2))))

    return signals


# ---------------------------------------------------------------------------
# External intelligence
# ---------------------------------------------------------------------------

_STANCE_TEMPLATES: dict[str, dict[str, str]] = {
    "Semiconductor Cycle": {
        "Favorable": "Industry capital-spending indices continue to expand; test "
                     "intensity per device is rising with advanced packaging.",
        "Neutral": "Capital-spending indicators are mixed; test demand is holding "
                   "near plan with no clear inflection either way.",
        "Watch": "Leading indicators have flattened; several fabs have paused "
                 "expansion decisions pending demand clarity.",
        "Unfavorable": "Capital-equipment orders are contracting across the "
                       "industry; test capacity additions are being deferred.",
    },
    "AI Infrastructure Spending": {
        "Favorable": "Hyperscaler capex guidance remains aggressive; compute and "
                     "system-level test demand continues to outrun supply.",
        "Neutral": "AI infrastructure spending remains strong but growth rates "
                   "are normalizing from exceptional levels.",
        "Watch": "Several hyperscalers signaled slower sequential capex growth; "
                 "monitor pull-in behavior for reversal risk.",
        "Unfavorable": "AI accelerator order digestion is underway; near-term "
                       "compute test demand is pausing.",
    },
    "Automotive Outlook": {
        "Favorable": "Vehicle electrification and ADAS content growth keep "
                     "automotive semiconductor demand resilient.",
        "Neutral": "Automotive semiconductor demand is stable; inventory at tier-1 "
                   "suppliers is near normal levels.",
        "Watch": "Auto OEM production schedules have softened in Europe; watch "
                 "for order deferrals from automotive customers.",
        "Unfavorable": "Automotive semiconductor customers are working down "
                       "excess inventory; test utilization is falling.",
    },
    "Industrial Demand": {
        "Favorable": "Industrial automation orders are recovering; distributor "
                     "inventories have normalized.",
        "Neutral": "Industrial demand is bouncing along the bottom of its cycle "
                   "with early signs of stabilization.",
        "Watch": "Industrial bookings remain weak; recovery timing keeps sliding "
                 "to the right in customer commentary.",
        "Unfavorable": "Industrial and mixed-signal demand continues to contract; "
                       "customers report elevated channel inventory.",
    },
    "Consumer Electronics": {
        "Favorable": "Smartphone unit forecasts have been revised upward; mobile "
                     "test reuse rates are falling.",
        "Neutral": "Consumer electronics demand is seasonal and broadly in line "
                   "with plan.",
        "Watch": "Handset build plans for the next cycle are trending below prior "
                 "expectations.",
        "Unfavorable": "Consumer device demand is weakening; mobile test capacity "
                       "is being redeployed rather than expanded.",
    },
    "Customer Inventory Commentary": {
        "Favorable": "Customers report lean finished-goods inventory and are "
                     "protecting equipment delivery slots.",
        "Neutral": "Customer inventory positions are mixed by segment but broadly "
                   "manageable.",
        "Watch": "Several large customers flagged rising inventory in earnings "
                 "commentary; slot protection is weakening.",
        "Unfavorable": "Broad customer inventory correction underway; equipment "
                       "delivery deferrals are the primary risk.",
    },
    "Competitor Announcements": {
        "Favorable": "Competitor lead times remain extended, supporting pricing "
                     "and share-capture opportunities.",
        "Neutral": "No competitor announcements this cycle that change the "
                   "demand picture.",
        "Watch": "A competitor announced capacity expansion targeting compute "
                 "test; monitor pricing pressure on new opportunities.",
        "Unfavorable": "Competitors are discounting to fill capacity, signaling "
                       "softer end demand than public forecasts imply.",
    },
    "Trade Policy": {
        "Favorable": "No new export-control actions this cycle; existing license "
                     "workflows are functioning.",
        "Neutral": "Trade-policy environment is unchanged; compliance overhead "
                   "remains manageable.",
        "Watch": "Draft export-control revisions under review could affect "
                 "advanced test shipments to select regions.",
        "Unfavorable": "New export restrictions announced; a portion of regional "
                       "backlog requires license review before shipment.",
    },
    "Tariffs": {
        "Favorable": "Tariff exclusions for test equipment components were "
                     "renewed, removing a modeled cost risk.",
        "Neutral": "Tariff regime unchanged; current landed-cost assumptions "
                   "remain valid.",
        "Watch": "Proposed tariff schedule changes could raise electronics "
                 "component costs; suppliers are pre-positioning inventory.",
        "Unfavorable": "New tariffs raise component landed costs; expect "
                       "purchase-price variance pressure over two quarters.",
    },
    "Macro Indicators": {
        "Favorable": "PMIs and semiconductor billings are both expanding; the "
                     "macro backdrop supports the demand plan.",
        "Neutral": "Macro indicators are mixed — manufacturing PMIs soft, "
                   "electronics billings stable.",
        "Watch": "Global PMIs slipped below 50 this month; historically this "
                 "leads equipment push-outs by one to two quarters.",
        "Unfavorable": "Macro deterioration is broad-based; financing conditions "
                       "are delaying customer capacity decisions.",
    },
}

_STANCE_ORDER = ["Favorable", "Neutral", "Watch", "Unfavorable"]
_STANCE_SCORE = {"Favorable": 1.0, "Neutral": 0.6, "Watch": 0.35, "Unfavorable": 0.0}


def _build_external(rng: np.random.Generator,
                    signal_score: float) -> list[ExternalIntel]:
    """Stances are random but biased toward coherence with internal signals."""
    tilt = (signal_score - 50) / 100.0          # -0.5 .. +0.5
    out = []
    for topic, templates in _STANCE_TEMPLATES.items():
        p = np.array([0.25 + 0.5 * tilt, 0.35, 0.25 - 0.2 * tilt, 0.15 - 0.3 * tilt])
        p = np.clip(p, 0.04, None)
        p = p / p.sum()
        stance = _STANCE_ORDER[int(rng.choice(4, p=p))]
        out.append(ExternalIntel(topic, stance, templates[stance]))
    return out


# ---------------------------------------------------------------------------
# Customer confidence
# ---------------------------------------------------------------------------

_REASON_LIBRARY = {
    "push_good": "Low push-out activity",
    "push_bad": "Increased push-outs",
    "cancel_good": "Minimal cancellation activity",
    "cancel_bad": "Elevated cancellation risk",
    "backlog_good": "Strong firm backlog coverage",
    "backlog_bad": "Thin backlog beyond the current quarter",
    "fc_good": "Stable, high-confidence forecasts",
    "fc_bad": "Historically volatile forecasts",
    "mom_good": "Improving bookings trend",
    "mom_bad": "Weak recent bookings trend",
}


def _build_customers(data: InputData,
                     rng: np.random.Generator) -> list[CustomerConfidence]:
    dem = data.demand.copy()
    dem["units"] = dem["base_forecast_units"] + dem["backlog_units"]
    dem["revenue"] = dem["units"] * dem["asp_usd"]
    out = []
    for (cust, group), g in dem.groupby(["customer", "customer_group"]):
        push = float(g["push_out_prob"].mean())
        cancel = float(g["cancel_prob"].mean())
        fconf = float(g["forecast_confidence"].mean())
        backlog_cov = float(g["backlog_units"].sum() / max(g["units"].sum(), 1))
        momentum = float(rng.normal(0, 1))     # synthetic recent-bookings tilt

        comp = {
            "push": _score(-push, -0.20, -0.05),
            "cancel": _score(-cancel, -0.05, -0.005),
            "backlog": _score(backlog_cov, 0.10, 0.45),
            "fc": _score(fconf, 0.55, 0.90),
            "mom": _score(momentum, -2, 2),
        }
        score = (0.30 * comp["push"] + 0.15 * comp["cancel"]
                 + 0.20 * comp["backlog"] + 0.20 * comp["fc"]
                 + 0.15 * comp["mom"])
        rating = "High" if score >= 65 else "Moderate" if score >= 42 else "Low"

        ranked = sorted(comp.items(), key=lambda kv: kv[1])
        reasons = []
        for key, val in reversed(ranked[-2:]):          # two strongest positives
            if val >= 60:
                reasons.append(_REASON_LIBRARY[f"{key}_good"])
        for key, val in ranked[:2]:                     # two weakest
            if val <= 45:
                reasons.append(_REASON_LIBRARY[f"{key}_bad"])
        if not reasons:
            reasons.append("Signals broadly in line with plan")

        out.append(CustomerConfidence(
            customer=cust, group=group,
            horizon_revenue=float(g["revenue"].sum()),
            rating=rating, score=round(score, 1), reasons=reasons))
    out.sort(key=lambda c: -c.horizon_revenue)
    return out


# ---------------------------------------------------------------------------
# Narrative and composite
# ---------------------------------------------------------------------------

def _narrative(level: str, score: float, signals: list[MarketSignal],
               external: list[ExternalIntel]) -> str:
    # only headline signals that carry real weight in the composite
    candidates = [s for s in signals if s.weight >= 0.08]
    weighted = sorted(candidates, key=lambda s: (s.score - 50) * s.weight)
    worst, best = weighted[0], weighted[-1]
    ext_fav = [e.topic for e in external if e.stance == "Favorable"]
    ext_bad = [e.topic for e in external
               if e.stance in ("Watch", "Unfavorable")]

    parts = [f"Demand confidence is **{level}** this month ({score:.0f}/100)."]
    pos_bits = []
    if best.score >= 60:
        pos_bits.append(f"{best.name.lower()} is supportive "
                        f"({best.current:g}{best.unit if best.unit != 'count' else ''})")
    if ext_fav:
        pos_bits.append(f"external conditions in {ext_fav[0]} remain favorable")
    if pos_bits:
        parts.append(("Strengths: " + " and ".join(pos_bits) + ".").capitalize())
    neg_bits = []
    if worst.score <= 50:
        direction = "rising" if worst.trend > 0 else "weak"
        neg_bits.append(f"{direction} {worst.name.lower()} "
                        f"({worst.current:g}{worst.unit if worst.unit != 'count' else ''})")
    if ext_bad:
        neg_bits.append(f"external watch items in {', '.join(ext_bad[:2])}")
    if neg_bits:
        parts.append("However, " + " and ".join(neg_bits)
                     + " increase downside uncertainty.")
    parts.append(f"This assessment feeds the simulation directly: "
                 f"{CONFIDENCE_BUSINESS_MEANING[level][0].lower()}"
                 f"{CONFIDENCE_BUSINESS_MEANING[level][1:]}")
    return " ".join(parts)


MAX_TOPIC_IMPACT_PTS = 3.0     # a single topic may move the score at most this much
MAX_TOTAL_IMPACT_PTS = 12.0    # external intel in total may move the score at most this much


def parse_curated_external(csv_text: str) -> list[ExternalIntel]:
    """Parse a curated external-intelligence CSV (the Claude/analyst feed).

    Expected columns: topic, stance, summary, sources (';'-separated),
    proposed_impact_pts, as_of. Stances are validated and proposed impacts
    clipped to +/-MAX_TOPIC_IMPACT_PTS so no single topic can swing the score.
    """
    import io
    df = pd.read_csv(io.StringIO(csv_text))
    required = {"topic", "stance", "summary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Curated external-intel file is missing columns: {sorted(missing)}")
    out = []
    for _, r in df.iterrows():
        stance = str(r["stance"]).strip().title()
        if stance not in _STANCE_ORDER:
            raise ValueError(f"Unknown stance '{r['stance']}' for topic '{r['topic']}'")
        raw_impact = r.get("proposed_impact_pts")
        impact = (float(np.clip(float(raw_impact), -MAX_TOPIC_IMPACT_PTS,
                                MAX_TOPIC_IMPACT_PTS))
                  if pd.notna(raw_impact) else None)
        sources = ([s.strip() for s in str(r["sources"]).split(";") if s.strip()]
                   if "sources" in df.columns and pd.notna(r.get("sources")) else [])
        out.append(ExternalIntel(
            topic=str(r["topic"]).strip(), stance=stance,
            summary=str(r["summary"]).strip(), sources=sources,
            proposed_impact_pts=impact,
            as_of=str(r["as_of"]).strip() if "as_of" in df.columns and pd.notna(r.get("as_of")) else ""))
    return out


def build_demand_confidence(data: InputData, config: AppConfig,
                            seed: int | None = None,
                            external: list[ExternalIntel] | None = None,
                            ) -> DemandConfidence:
    """Derive the full Demand Confidence assessment from the input data.

    Deterministic for a given data seed: the same synthetic company always
    produces the same intelligence picture. If a curated `external` list is
    supplied (analyst- or Claude-produced), it replaces the synthetic external
    templates; when its entries carry proposed_impact_pts, those proposals
    (clipped per topic and in total) become the external score adjustment.
    """
    base_seed = seed if seed is not None else data.seed
    rng = np.random.default_rng(base_seed * 7919 + 13)

    signals = _build_signals(data, rng)
    composite = float(sum(s.score * s.weight for s in signals)
                      / sum(s.weight for s in signals))
    if external is not None:
        ext_source = "curated"
        impacts = [e.proposed_impact_pts for e in external
                   if e.proposed_impact_pts is not None]
        if impacts:
            ext_adj = float(np.clip(sum(impacts), -MAX_TOTAL_IMPACT_PTS,
                                    MAX_TOTAL_IMPACT_PTS))
        else:
            ext_adj = float(np.mean([_STANCE_SCORE[e.stance] for e in external])
                            - 0.5) * 16
    else:
        ext_source = "synthetic"
        external = _build_external(rng, composite)
        ext_adj = float(np.mean([_STANCE_SCORE[e.stance] for e in external])
                        - 0.5) * 16
    score = float(np.clip(composite + ext_adj, 0, 100))
    level = _level_from_score(score)
    customers = _build_customers(data, rng)

    acc_signal = next(s for s in signals if s.name == "Forecast Accuracy")
    fc_pct = float(np.clip(0.55 * acc_signal.score + 0.30 * score
                           + 0.15 * float(data.demand["forecast_confidence"].mean()) * 100,
                           0, 100))
    fc_label = ("High" if fc_pct >= 70 else
                "Moderate" if fc_pct >= 50 else "Low")
    fc_reason = (
        f"Driven by trailing forecast accuracy of {acc_signal.current:.0f}% "
        f"({'improving' if acc_signal.trend >= 0 else 'declining'} vs three months "
        f"ago), overall demand-signal health of {composite:.0f}/100, and stated "
        f"planner confidence in the demand plan.")

    return DemandConfidence(
        score=round(score, 1), level=level,
        narrative=_narrative(level, score, signals, external),
        signals=signals, external=external, customers=customers,
        forecast_confidence_pct=round(fc_pct, 0),
        forecast_confidence_label=fc_label,
        forecast_confidence_reason=fc_reason,
        external_adjustment_pts=round(ext_adj, 1),
        external_source=ext_source)


# ---------------------------------------------------------------------------
# Simulation integration
# ---------------------------------------------------------------------------

def merge_confidence_params(base_params: dict[str, Any],
                            overrides: dict[str, Any]) -> dict[str, Any]:
    """Stack two parameter layers (e.g. confidence + scenario, or scenario +
    action).

    Additive scalar knobs sum, multiplicative scalar knobs multiply, dict-valued
    knobs merge per sub-key (the overrides layer winning on collisions), and any
    other collision resolves to the overrides layer. This lets a management
    action stack on top of a scenario backdrop instead of silently replacing it."""
    merged: dict[str, Any] = dict(overrides)
    for key, val in base_params.items():
        if key not in merged:
            merged[key] = val
        elif isinstance(val, dict) and isinstance(merged[key], dict):
            merged[key] = {**val, **merged[key]}
        elif key.endswith("_add") and np.isscalar(val):
            merged[key] = merged[key] + val
        elif key.endswith("_mult") and np.isscalar(val):
            merged[key] = merged[key] * val
        # otherwise the overrides layer's explicit setting wins
    return merged


def assumptions_in_business_language(data: InputData, config: AppConfig,
                                     dc: DemandConfidence) -> pd.DataFrame:
    """The assumptions feeding the Monte Carlo, in executive language."""
    unc = config.uncertainty
    dem = data.demand
    p = dc.sim_params
    rows = [
        ("Demand volatility",
         " · ".join(f"{m.split(' /')[0].split(' &')[0]} ±{s * 100:.0f}%"
                    for m, s in unc.market_demand_sigma.items()),
         "Typical month-to-month swing in end-market demand before any "
         "confidence adjustment."),
        ("Confidence adjustment to volatility",
         f"×{p['demand_sigma_mult']:.2f} ({dc.level} confidence)",
         "The current Demand Confidence level scales all demand volatility. "
         "Below 1.00 narrows the distribution; above 1.00 widens it."),
        ("Customer correlation",
         "Markets move together (semicap cycle links all five)",
         "Demand shocks are correlated through common market factors, so bad "
         "months cluster across product families rather than cancelling out."),
        ("Push-out probability",
         f"{dem['push_out_prob'].mean() * 100 + p['pushout_prob_add'] * 100:.1f}% "
         f"of orders per month "
         f"({'+' if p['pushout_prob_add'] >= 0 else ''}{p['pushout_prob_add'] * 100:.0f} "
         f"pts from confidence)",
         "Chance a committed order slips one to two months. Slipped revenue "
         "moves later or out of the horizon."),
        ("Pull-in probability",
         f"{dem['pull_in_prob'].mean() * 100:.1f}% of orders per month",
         "Chance a customer requests earlier delivery, adding near-term "
         "capacity pressure."),
        ("Cancellation probability",
         f"{dem['cancel_prob'].mean() * 100 * p['cancel_prob_mult']:.1f}% of "
         f"orders per month (×{p['cancel_prob_mult']:.1f} from confidence)",
         "Orders removed entirely; revenue is lost, and committed materials "
         "become excess-inventory exposure."),
        ("Forecast confidence by horizon",
         f"{dem['forecast_confidence'].mean() * 100:.0f}% average "
         "(declines with distance)",
         "Near months are mostly firm backlog; far months are forecast and "
         "carry proportionally more uncertainty."),
    ]
    return pd.DataFrame(rows, columns=["Assumption", "Current setting",
                                       "What it means"])
