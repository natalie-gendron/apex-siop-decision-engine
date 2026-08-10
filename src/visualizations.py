"""Executive Plotly figures.

Design system: validated light palette (categorical order blue, orange, aqua,
yellow, magenta — CVD-checked), one hue for sequential magnitude, blue-red
diverging, reserved status colors, recessive grid, direct labels where useful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .models import BaselineResult, SimulationResult
from .simulation import quarterly
from .utils import (
    PRODUCT_FAMILIES,
    fmt_money,
    month_labels,
    quarter_labels,
)

# validated palette (see dataviz reference; order is the CVD-safety mechanism)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ_BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
          "critical": "#d03b3b"}
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_C = "#c3c2b7"
SURFACE = "#fcfcfb"


def _layout(fig: go.Figure, title: str, xtitle: str = "", ytitle: str = "",
            height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  size=12, color=INK2),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        height=height, margin=dict(l=60, r=24, t=54, b=44),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font=dict(size=12, color=INK)),
    )
    fig.update_xaxes(title_text=xtitle, showgrid=False, linecolor=BASELINE_C,
                     tickfont=dict(color=MUTED, size=11), title_font=dict(size=12))
    fig.update_yaxes(title_text=ytitle, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor=BASELINE_C, tickfont=dict(color=MUTED, size=11),
                     title_font=dict(size=12))
    return fig


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def distribution_with_target(samples: np.ndarray, plan: float, title: str,
                             xlabel: str, value_fmt: str = "money",
                             target_label: str = "Plan") -> go.Figure:
    """Probability distribution with a plan/target reference line."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples, nbinsx=60, marker=dict(color=SERIES[0],
                                          line=dict(color=SURFACE, width=1)),
        opacity=0.9, name="Simulated outcomes",
        hovertemplate="%{x}<br>%{y} sims<extra></extra>"))
    p_meet = float((samples >= plan).mean())
    fig.add_vline(x=plan, line=dict(color=INK, width=2, dash="dash"))
    med = float(np.median(samples))
    fig.add_vline(x=med, line=dict(color=SERIES[1], width=2))
    if value_fmt == "money":
        plan_txt, med_txt = fmt_money(plan), fmt_money(med)
    else:
        plan_txt, med_txt = f"{plan:.1%}", f"{med:.1%}"
    fig.add_annotation(x=plan, y=1.06, yref="paper",
                       text=f"{target_label} {plan_txt} · P(≥) {p_meet:.0%}",
                       showarrow=False, font=dict(size=11, color=INK))
    fig.add_annotation(x=med, y=0.97, yref="paper", text=f"Median {med_txt}",
                       showarrow=False, font=dict(size=11, color=SERIES[1]))
    return _layout(fig, title, xlabel, "Simulations", legend=False)


def quarterly_fan_chart(result: SimulationResult, plan_q: np.ndarray,
                        title_suffix: str = "") -> go.Figure:
    """Quarterly revenue fan: P5-P95 and P25-P75 bands, median, plan markers."""
    rev_q = quarterly(result.revenue) / 1e6
    q = quarter_labels()
    p5, p25, p50, p75, p95 = (np.percentile(rev_q, p, axis=0)
                              for p in (5, 25, 50, 75, 95))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q + q[::-1], y=np.concatenate([p95, p5[::-1]]),
                             fill="toself", fillcolor="rgba(42,120,214,0.12)",
                             line=dict(width=0), name="P5–P95",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=q + q[::-1], y=np.concatenate([p75, p25[::-1]]),
                             fill="toself", fillcolor="rgba(42,120,214,0.25)",
                             line=dict(width=0), name="P25–P75",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=q, y=p50, mode="lines+markers",
                             line=dict(color=SERIES[0], width=2),
                             marker=dict(size=8), name="Median",
                             hovertemplate="%{x}: $%{y:.0f}M<extra>Median</extra>"))
    fig.add_trace(go.Scatter(x=q, y=plan_q / 1e6, mode="markers",
                             marker=dict(symbol="diamond", size=10, color=INK),
                             name="Plan",
                             hovertemplate="%{x}: $%{y:.0f}M<extra>Plan</extra>"))
    return _layout(fig, f"Quarterly revenue fan vs plan{title_suffix}",
                   "", "Revenue ($M)")


def inventory_trajectory(result: SimulationResult, target: float,
                         title_suffix: str = "") -> go.Figure:
    inv = result.inventory / 1e6
    m = month_labels()
    p5, p50, p95 = (np.percentile(inv, p, axis=0) for p in (5, 50, 95))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m + m[::-1], y=np.concatenate([p95, p5[::-1]]),
                             fill="toself", fillcolor="rgba(42,120,214,0.15)",
                             line=dict(width=0), name="P5–P95", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=m, y=p50, mode="lines",
                             line=dict(color=SERIES[0], width=2), name="Median",
                             hovertemplate="%{x}: $%{y:.0f}M<extra>Median</extra>"))
    fig.add_hline(y=target / 1e6, line=dict(color=INK, width=2, dash="dash"),
                  annotation_text=f"Target {fmt_money(target)}",
                  annotation_font=dict(size=11, color=INK))
    return _layout(fig, f"Month-end inventory trajectory vs target "
                        f"(18-month horizon){title_suffix}",
                   "", "Inventory ($M)")


def revenue_bridge(base_kpi: dict, scen_kpi: dict) -> go.Figure:
    """Waterfall from base expected FY revenue to scenario expected FY revenue."""
    base_rev = base_kpi["fy_revenue"]["mean"]
    scen_rev = scen_kpi["fy_revenue"]["mean"]
    delta = scen_rev - base_rev
    fig = go.Figure(go.Waterfall(
        x=["Base case", f"{scen_kpi['scenario']}", "Scenario total"],
        measure=["absolute", "relative", "total"],
        y=[base_rev / 1e6, delta / 1e6, None],
        text=[fmt_money(base_rev), fmt_money(delta), fmt_money(scen_rev)],
        textposition="outside",
        connector=dict(line=dict(color=BASELINE_C)),
        increasing=dict(marker=dict(color=SERIES[2])),
        decreasing=dict(marker=dict(color=STATUS["critical"])),
        totals=dict(marker=dict(color=SERIES[0])),
    ))
    return _layout(fig, "Expected FY revenue bridge", "", "Revenue ($M)",
                   legend=False)


def context_bridge(base_kpi: dict, world_kpi: dict, final_kpi: dict,
                   world_name: str, package_label: "str | None",
                   metric: str = "revenue") -> go.Figure:
    """Waterfall decomposing the evaluation context vs the base outlook:
    base → what the world (scenario) does → what the response package
    does → conditioned outlook. Either middle stage may be absent.

    metric="revenue" bridges expected FY revenue in $M; metric="margin"
    bridges expected FY gross margin in percentage points (each step is
    sequential conditioning, so the bars sum to the total exactly)."""
    if metric == "revenue":
        def val(k):
            return k["fy_revenue"]["mean"] / 1e6

        def fmt_abs(v):
            return fmt_money(v * 1e6)

        fmt_delta = fmt_abs
        title = "Expected FY revenue — world vs response decomposition"
        ytitle = "Revenue ($M)"
    elif metric == "margin":
        def val(k):
            return k["fy_gm"]["mean"] * 100

        def fmt_abs(v):
            return f"{v:.1f}%"

        def fmt_delta(v):
            return f"{v:+.1f} pts"

        title = "Expected FY gross margin — world vs response decomposition"
        ytitle = "Gross margin (%)"
    else:
        raise ValueError(f"Unknown bridge metric '{metric}'")
    base_v = val(base_kpi)
    x = ["Base outlook"]
    measure = ["absolute"]
    y = [base_v]
    text = [fmt_abs(base_v)]
    prev = base_v
    if world_name != "Base Case":
        d = val(world_kpi) - prev
        x.append(world_name)
        measure.append("relative")
        y.append(d)
        text.append(fmt_delta(d))
        prev += d
    if package_label:
        d = val(final_kpi) - prev
        x.append(package_label)
        measure.append("relative")
        y.append(d)
        text.append(fmt_delta(d))
    x.append("Conditioned outlook")
    measure.append("total")
    y.append(None)
    text.append(fmt_abs(val(final_kpi)))
    fig = go.Figure(go.Waterfall(
        x=x, measure=measure, y=y, text=text, textposition="outside",
        connector=dict(line=dict(color=BASELINE_C)),
        increasing=dict(marker=dict(color=SERIES[2])),
        decreasing=dict(marker=dict(color=STATUS["critical"])),
        totals=dict(marker=dict(color=SERIES[0])),
    ))
    return _layout(fig, title, "", ytitle, legend=False)


def scenario_comparison_chart(rows: list[dict]) -> go.Figure:
    """Grouped deltas vs base for selected scenarios."""
    names = [r["scenario"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=[r["d_fy_revenue"] / 1e6 for r in rows],
        name="Δ FY revenue ($M)", marker_color=SERIES[0],
        hovertemplate="%{x}: %{y:+.1f}M<extra>Δ revenue</extra>"))
    fig.add_trace(go.Bar(
        x=names, y=[r["d_gross_profit"] / 1e6 for r in rows],
        name="Δ FY gross profit ($M)", marker_color=SERIES[2],
        hovertemplate="%{x}: %{y:+.1f}M<extra>Δ gross profit</extra>"))
    fig.add_trace(go.Bar(
        x=names, y=[r["d_inventory"] / 1e6 for r in rows],
        name="Δ ending inventory ($M)", marker_color=SERIES[1],
        hovertemplate="%{x}: %{y:+.1f}M<extra>Δ inventory</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    return _layout(fig, "Full-year scenario deltas versus base case", "", "$M",
                   height=420)


# ---------------------------------------------------------------------------
# Heat maps and matrices
# ---------------------------------------------------------------------------

def utilization_heatmap(result: SimulationResult,
                        title_suffix: str = "") -> go.Figure:
    """EMS + integration expected utilization by month."""
    m = month_labels()
    rows = {
        "EMS network": result.ems_utilization.mean(axis=0),
        "Final integration": result.integration_utilization.mean(axis=0),
    }
    z = np.vstack(list(rows.values()))
    fig = go.Figure(go.Heatmap(
        z=z * 100, x=m, y=list(rows.keys()),
        colorscale=[[i / (len(SEQ_BLUES) - 1), c] for i, c in enumerate(SEQ_BLUES)],
        zmin=50, zmax=110, colorbar=dict(title="Util %", ticksuffix="%",
                                         tickfont=dict(color=MUTED)),
        hovertemplate="%{y} · %{x}: %{z:.0f}%<extra></extra>",
        xgap=2, ygap=2))
    return _layout(fig, f"Expected capacity utilization by month{title_suffix}",
                   "", "",
                   height=260, legend=False)


def component_risk_heatmap(components: pd.DataFrame,
                           binding: pd.DataFrame) -> go.Figure:
    """Component risk: binding frequency vs supply/disruption attributes."""
    top = binding.merge(components[["component", "allocation_risk",
                                    "disruption_prob_monthly", "lead_time_weeks"]],
                        on="component", how="left").head(10)
    metrics = {
        "Binding frequency": top["binding_frequency"] * 100,
        "Allocation risk": top["allocation_risk"] * 100,
        "Monthly disruption prob.": top["disruption_prob_monthly"] * 100,
        "Lead time (weeks)": top["lead_time_weeks"],
    }
    # normalize each row 0-1 for comparable shading; hover shows raw values
    z, text = [], []
    for name, vals in metrics.items():
        v = vals.to_numpy(float)
        rng = v.max() - v.min()
        z.append((v - v.min()) / rng if rng > 0 else v * 0)
        unit = "wk" if "weeks" in name else "%"
        text.append([f"{x:.1f}{unit}" for x in v])
    fig = go.Figure(go.Heatmap(
        z=np.array(z), x=top["component"], y=list(metrics.keys()),
        colorscale=[[i / (len(SEQ_BLUES) - 1), c] for i, c in enumerate(SEQ_BLUES)],
        showscale=False, text=np.array(text), texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="%{y} · %{x}: %{text}<extra></extra>", xgap=2, ygap=2))
    fig.update_xaxes(tickangle=30)
    return _layout(fig, "Component risk profile (top binding items; shading scaled per row)",
                   "", "", height=340, legend=False)


def risk_matrix(family_risk: pd.DataFrame, base_kpi: dict) -> go.Figure:
    """Probability-of-shortfall vs revenue impact by product family."""
    fig = go.Figure()
    exp, base = family_risk["expected_fy_revenue"], family_risk["baseline_fy_revenue"]
    prob = np.clip(1 - exp / base, 0.02, 0.98)          # probability proxy: expected shortfall share
    impact = family_risk["revenue_at_risk"] / 1e6
    for i, row in family_risk.iterrows():
        fam = row["product_family"]
        c = SERIES[PRODUCT_FAMILIES.index(fam)]
        fig.add_trace(go.Scatter(
            x=[prob.iloc[i] * 100], y=[impact.iloc[i]], mode="markers+text",
            marker=dict(size=14, color=c, line=dict(color=SURFACE, width=2)),
            text=[fam.replace(" Test", "")], textposition="top center",
            textfont=dict(size=10, color=INK2), name=fam,
            hovertemplate=(f"{fam}<br>Expected shortfall vs baseline: "
                           f"%{{x:.0f}}%<br>Revenue at risk: $%{{y:.0f}}M<extra></extra>")))
    return _layout(fig, "Risk matrix — expected FY shortfall vs FY revenue at risk",
                   "Expected shortfall vs baseline plan (%)",
                   "FY revenue at risk ($M)", height=420, legend=False)


def tornado_chart(ranking: pd.DataFrame, outcome: str) -> go.Figure:
    df = ranking.head(10).iloc[::-1]
    colors = [SERIES[0] if v >= 0 else SERIES[1] for v in df["spearman_rho"]]
    fig = go.Figure(go.Bar(
        x=df["spearman_rho"], y=df["driver"], orientation="h",
        marker=dict(color=colors, line=dict(color=SURFACE, width=1)),
        hovertemplate="%{y}: ρ = %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=BASELINE_C, width=1))
    fig = _layout(fig, f"Ranked drivers of {outcome} (Spearman rank correlation — "
                  "association, not causation)", "Rank correlation (ρ)", "",
                  height=420, legend=False)
    fig.update_yaxes(tickfont=dict(size=11, color=INK2))
    return fig


def family_risk_chart(family_risk: pd.DataFrame) -> go.Figure:
    df = family_risk.sort_values("revenue_at_risk")
    colors = [SERIES[PRODUCT_FAMILIES.index(f)] for f in df["product_family"]]
    fig = go.Figure(go.Bar(
        x=df["revenue_at_risk"] / 1e6, y=df["product_family"], orientation="h",
        marker=dict(color=colors, line=dict(color=SURFACE, width=1)),
        text=[fmt_money(v) for v in df["revenue_at_risk"]],
        textposition="outside", textfont=dict(size=11, color=INK2),
        hovertemplate="%{y}: $%{x:.0f}M at risk<extra></extra>"))
    fig = _layout(fig, "FY revenue at risk by product family (plan vs P5)",
                  "Revenue at risk ($M)", "", height=360, legend=False)
    fig.update_yaxes(tickfont=dict(size=11, color=INK2))
    return fig


def ems_capacity_vs_demand(baseline: BaselineResult) -> go.Figure:
    m = month_labels()
    cap = baseline.site_capacity.sum(axis=0).to_numpy(float)
    load = baseline.site_load.sum(axis=0).to_numpy(float)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m, y=load, name="Planned load (std units)",
                         marker_color=SERIES[0],
                         hovertemplate="%{x}: %{y:.0f} std units<extra>Load</extra>"))
    fig.add_trace(go.Scatter(x=m, y=cap, name="Effective capacity", mode="lines",
                             line=dict(color=INK, width=2, dash="dash"),
                             hovertemplate="%{x}: %{y:.0f} std units<extra>Capacity</extra>"))
    return _layout(fig, "EMS network: planned load vs effective capacity",
                   "", "Standard-equivalent systems")


def backlog_aging_chart(baseline: BaselineResult) -> go.Figure:
    aging = baseline.backlog_aging
    fig = go.Figure()
    buckets = [(0, "Current month"), (1, "1 month past due"), (2, "2+ months past due")]
    for i, (age, label) in enumerate(buckets):
        sub = (aging[aging["age_months"] >= age] if age == 2
               else aging[aging["age_months"] == age])
        by_month = sub.groupby("month")["units"].sum().reindex(month_labels()).fillna(0)
        fig.add_trace(go.Bar(x=by_month.index, y=by_month.values, name=label,
                             marker_color=SERIES[i],
                             hovertemplate="%{x}: %{y:.0f} systems<extra>" + label + "</extra>"))
    fig.update_layout(barmode="stack", bargap=0.3)
    return _layout(fig, "Demand queue by aging bucket (baseline plan)", "", "Systems")


def site_utilization_heatmap(baseline: BaselineResult) -> go.Figure:
    util = (baseline.site_load / baseline.site_capacity.clip(lower=1e-9)) * 100
    fig = go.Figure(go.Heatmap(
        z=util.values, x=month_labels(), y=util.index,
        colorscale=[[i / (len(SEQ_BLUES) - 1), c] for i, c in enumerate(SEQ_BLUES)],
        zmin=0, zmax=110,
        colorbar=dict(title="Util %", ticksuffix="%", tickfont=dict(color=MUTED)),
        hovertemplate="%{y} · %{x}: %{z:.0f}%<extra></extra>", xgap=2, ygap=2))
    return _layout(fig, "Baseline EMS site utilization", "", "", height=300,
                   legend=False)


def backlog_trajectory_chart(base_result: SimulationResult,
                             scen_result: SimulationResult,
                             scenario_name: str) -> go.Figure:
    """Expected past-due backlog by month: base case vs selected scenario.

    Past-due backlog = cumulative demand minus cumulative shipments (floored at
    zero), averaged across simulations; the band shows the scenario's P25-P75."""
    m = month_labels()

    def _backlog(res: SimulationResult) -> np.ndarray:
        cum = np.cumsum(res.units_demanded - res.units_shipped, axis=1)
        return np.clip(cum, 0, None)

    base_b = _backlog(base_result)
    scen_b = _backlog(scen_result)
    fig = go.Figure()
    p25, p75 = np.percentile(scen_b, 25, axis=0), np.percentile(scen_b, 75, axis=0)
    fig.add_trace(go.Scatter(
        x=m + m[::-1], y=np.concatenate([p75, p25[::-1]]), fill="toself",
        fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
        name="Scenario P25–P75", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=m, y=scen_b.mean(axis=0), mode="lines+markers",
        line=dict(color=SERIES[0], width=2), marker=dict(size=6),
        name=f"{scenario_name} (expected)",
        hovertemplate="%{x}: %{y:.0f} systems<extra>" + scenario_name + "</extra>"))
    if scen_result is not base_result:
        fig.add_trace(go.Scatter(
            x=m, y=base_b.mean(axis=0), mode="lines",
            line=dict(color=INK, width=2, dash="dash"), name="Base case (expected)",
            hovertemplate="%{x}: %{y:.0f} systems<extra>Base case</extra>"))
    title = ("Expected past-due backlog by month — base case"
             if scen_result is base_result else
             f"Expected past-due backlog by month — base vs {scenario_name}")
    return _layout(fig, title, "", "Systems past due", height=400)


def past_due_family_chart(df: pd.DataFrame,
                          title_suffix: str = "") -> go.Figure:
    """Stacked expected past-due backlog by product family (months × families)."""
    m = month_labels()
    fig = go.Figure()
    for i, fam in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=m, y=df[fam], mode="lines", stackgroup="one",
            line=dict(width=0.5, color=SERIES[i % len(SERIES)]), name=fam,
            hovertemplate="%{x}: %{y:.0f} systems<extra>" + fam + "</extra>"))
    return _layout(fig, f"Expected past-due backlog by family{title_suffix}",
                   "", "Systems past due", height=400)


def backlog_comparison_chart(base_curve: np.ndarray,
                             scen_curves: "dict[str, np.ndarray]") -> go.Figure:
    """Expected past-due backlog by month — one line per selected scenario
    (unmitigated) against the dashed base case."""
    m = month_labels()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m, y=base_curve, mode="lines",
        line=dict(color=INK, width=2, dash="dash"), name="Base case",
        hovertemplate="%{x}: %{y:.0f} systems<extra>Base case</extra>"))
    for i, (name, curve) in enumerate(scen_curves.items()):
        fig.add_trace(go.Scatter(
            x=m, y=curve, mode="lines+markers",
            line=dict(color=SERIES[i % len(SERIES)], width=2),
            marker=dict(size=5), name=name,
            hovertemplate="%{x}: %{y:.0f} systems<extra>" + name + "</extra>"))
    return _layout(fig,
                   "Expected past-due backlog by month — scenarios vs base",
                   "", "Systems past due", height=420)


def signal_history_chart(name: str, history: list[float], unit: str) -> go.Figure:
    """Trailing-12-month sparkline for one market signal."""
    x = [f"M-{i}" for i in range(len(history) - 1, 0, -1)] + ["Now"]
    fig = go.Figure(go.Scatter(
        x=x, y=history, mode="lines+markers",
        line=dict(color=SERIES[0], width=2), marker=dict(size=6),
        hovertemplate="%{x}: %{y:.2f}" + (unit if unit != "count" else "")
        + "<extra></extra>"))
    suffix = {"%": "%", "x": "×", "pts": " pts"}.get(unit, "")
    return _layout(fig, f"{name} — trailing 12 months", "",
                   f"{name}{f' ({suffix})' if suffix else ''}",
                   height=280, legend=False)


def confidence_sensitivity_chart(df, current_level: str) -> go.Figure:
    """FY revenue range (P5-P95, mean) at each Demand Confidence level."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(df["level"]) + list(df["level"])[::-1],
        y=list(df["p95"] / 1e6) + list(df["p5"] / 1e6)[::-1],
        fill="toself", fillcolor="rgba(42,120,214,0.15)",
        line=dict(width=0), name="P5–P95 range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=df["level"], y=df["mean"] / 1e6, mode="lines+markers",
        line=dict(color=SERIES[0], width=2), marker=dict(size=9),
        name="Expected FY revenue",
        hovertemplate="%{x}: $%{y:.0f}M<extra>Mean</extra>"))
    cur = df[df["level"] == current_level]
    if len(cur):
        fig.add_trace(go.Scatter(
            x=cur["level"], y=cur["mean"] / 1e6, mode="markers",
            marker=dict(symbol="diamond", size=16, color=SERIES[1],
                        line=dict(color=SURFACE, width=2)),
            name="Applied level",
            hovertemplate="Applied level: %{x}<extra></extra>"))
    return _layout(fig, "FY revenue uncertainty by Demand Confidence level",
                   "Demand Confidence", "FY revenue ($M)", height=400)


def monthly_revenue_chart(baseline: BaselineResult) -> go.Figure:
    m = baseline.monthly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m["month"], y=m["revenue_usd"] / 1e6,
                         name="Baseline revenue", marker_color=SERIES[0],
                         hovertemplate="%{x}: $%{y:.0f}M<extra>Baseline</extra>"))
    fig.add_trace(go.Scatter(x=m["month"], y=m["revenue_plan_usd"] / 1e6,
                             name="Plan", mode="lines",
                             line=dict(color=INK, width=2, dash="dash"),
                             hovertemplate="%{x}: $%{y:.0f}M<extra>Plan</extra>"))
    return _layout(fig, "Baseline monthly revenue vs plan", "", "Revenue ($M)")
