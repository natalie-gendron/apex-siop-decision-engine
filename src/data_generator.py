"""Synthetic data generation for Apex Test Systems.

All data is synthetic and generated from a seeded RNG. Nothing here is drawn
from any real company. Scale is calibrated to a diversified ATE manufacturer
with roughly $2.5B annual revenue and ~95 systems shipped per month.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .models import InputData
from .utils import (
    EMS_SITES,
    FAMILY_MARKET,
    INTEGRATION_SITES,
    N_MONTHS,
    PRODUCT_FAMILIES,
    month_labels,
)

# ---------------------------------------------------------------------------
# Static company design (synthetic)
# ---------------------------------------------------------------------------

CUSTOMERS = [
    # name, group, region, priority (1 = highest), families served
    ("Titan Semiconductor", "Large IDM", "North America", 1,
     ["Zenith Compute Test", "Atlas Automotive & Industrial Test", "Nexus System-Level Test"]),
    ("Kestrel Compute", "Fabless compute leader", "North America", 1,
     ["Zenith Compute Test", "Nexus System-Level Test"]),
    ("Northbridge Memory", "Memory manufacturer", "South Korea", 1,
     ["Vector Memory Test"]),
    ("Daehan Memory", "Memory manufacturer", "South Korea", 2,
     ["Vector Memory Test"]),
    ("Silverpine Mobile", "Fabless mobile", "Taiwan", 2,
     ["Horizon Mobility Test"]),
    ("Steinfeld Automotive Semi", "Automotive semiconductor", "Europe", 2,
     ["Atlas Automotive & Industrial Test"]),
    ("Pacific Crest Assembly & Test", "OSAT", "Taiwan", 2,
     ["Zenith Compute Test", "Horizon Mobility Test", "Vector Memory Test"]),
    ("Meridian Micro Devices", "Diversified", "Japan", 3,
     ["Atlas Automotive & Industrial Test", "Horizon Mobility Test", "Nexus System-Level Test"]),
]

# Average monthly units by family (company total), split across its customers.
FAMILY_MONTHLY_UNITS = {
    "Zenith Compute Test": 22.0,
    "Vector Memory Test": 20.0,
    "Horizon Mobility Test": 18.0,
    "Atlas Automotive & Industrial Test": 20.0,
    "Nexus System-Level Test": 12.0,
}

# Demand trend over the horizon by family (monthly compound growth).
FAMILY_TREND = {
    "Zenith Compute Test": 0.012,          # AI-driven growth
    "Vector Memory Test": 0.008,          # gradual memory recovery
    "Horizon Mobility Test": 0.000,
    "Atlas Automotive & Industrial Test": 0.002,
    "Nexus System-Level Test": 0.015,
}

PRODUCT_SPECS = {
    # asp, material%, conversion%, integration%, freight%, warranty%, build_cycle_mo,
    # integ_wk, fat_wk, install_wk, accept_lag_mo, complexity, fpy, rework, scrap
    "Zenith Compute Test": (3.4e6, 0.315, 0.045, 0.040, 0.014, 0.016, 2.0, 3.0, 2.0, 2.0, 1.0, 1.35, 0.90, 0.09, 0.008),
    "Vector Memory Test": (2.6e6, 0.300, 0.045, 0.038, 0.013, 0.014, 1.8, 2.5, 1.5, 1.5, 0.8, 1.15, 0.92, 0.07, 0.006),
    "Horizon Mobility Test": (1.7e6, 0.290, 0.042, 0.035, 0.012, 0.013, 1.5, 2.0, 1.0, 1.0, 0.5, 0.90, 0.94, 0.05, 0.005),
    "Atlas Automotive & Industrial Test": (1.5e6, 0.295, 0.043, 0.036, 0.013, 0.015, 1.5, 2.0, 1.5, 1.5, 0.7, 0.95, 0.93, 0.06, 0.005),
    "Nexus System-Level Test": (2.1e6, 0.310, 0.046, 0.042, 0.015, 0.016, 2.2, 3.5, 2.5, 2.0, 1.2, 1.25, 0.89, 0.10, 0.009),
}

FAMILY_EMS_QUAL = {
    "Zenith Compute Test": ["EMS Americas", "EMS Taiwan"],
    "Vector Memory Test": ["EMS Taiwan", "EMS Malaysia"],
    "Horizon Mobility Test": ["EMS Malaysia", "EMS Taiwan"],
    "Atlas Automotive & Industrial Test": ["EMS Eastern Europe", "EMS Malaysia", "EMS Americas"],
    "Nexus System-Level Test": ["EMS Americas", "EMS Taiwan"],
}

EMS_SPECS = {
    # base capacity (std-equivalent systems/mo), cost/std-unit, region, adherence, fpy,
    # rework, scrap, labor_avail, quality_escape, disruption_prob, logistics_wk,
    # overtime_max%, flex%, ot_premium, reservation_fee/unit, min_lot, ramp%
    "EMS Americas":       (37, 118e3, "North America", 0.955, 0.94, 0.05, 0.004, 0.97, 0.006, 0.010, 1.5, 0.12, 0.10, 0.45, 9.0e3, 2, 0.15),
    "EMS Malaysia":       (41, 92e3, "Southeast Asia", 0.940, 0.92, 0.07, 0.006, 0.95, 0.009, 0.020, 3.0, 0.15, 0.12, 0.40, 7.0e3, 3, 0.20),
    "EMS Taiwan":         (44, 101e3, "Taiwan", 0.960, 0.95, 0.05, 0.004, 0.97, 0.005, 0.018, 2.5, 0.12, 0.10, 0.42, 8.0e3, 3, 0.18),
    "EMS Eastern Europe": (23, 97e3, "Europe", 0.935, 0.91, 0.08, 0.007, 0.94, 0.010, 0.015, 2.5, 0.10, 0.08, 0.48, 7.5e3, 2, 0.12),
}

INTEGRATION_SPECS = {
    # integ cap (systems/mo), calibration cap, FAT cap, install crews (systems/mo),
    # labor_avail, first_pass, rework_days, families (all), acceptance_wk
    "Final Integration North America": (50, 54, 52, 44, 0.96, 0.92, 6, 3.5),
    "Final Integration Asia": (70, 74, 72, 60, 0.95, 0.91, 7, 3.0),
}

COMPONENT_SPECS = [
    # name, category, supplier, region, unit_cost, lt_wk, lt_std_wk, families, usage/system,
    # tightness (months of supply vs need in open POs: <1.0 = constrained), alloc_risk,
    # disrupt_prob, expedite_ok, exp_premium, alt_source, alt_qual_mo, obsolescence
    ("High-End FPGA", "FPGA", "Corvid Logic", "North America", 48e3, 30, 6,
     ["Zenith Compute Test", "Nexus System-Level Test"], 6, 0.97, 0.35, 0.030, True, 0.35, False, 6, 0.10),
    ("Mid-Range FPGA", "FPGA", "Corvid Logic", "North America", 14e3, 22, 4,
     ["Vector Memory Test", "Horizon Mobility Test", "Atlas Automotive & Industrial Test"], 4, 1.10, 0.15, 0.015, True, 0.25, True, 4, 0.08),
    ("Precision Analog Instrument Module", "Precision instrumentation", "Helvetia Instruments", "Europe", 62e3, 26, 5,
     ["Zenith Compute Test", "Atlas Automotive & Industrial Test", "Nexus System-Level Test"], 3, 1.02, 0.20, 0.020, True, 0.30, False, 8, 0.06),
    ("High-Speed Interconnect Set", "High-speed interconnect", "Straitline Interconnect", "Taiwan", 21e3, 18, 4,
     ["Zenith Compute Test", "Vector Memory Test", "Nexus System-Level Test"], 8, 0.95, 0.30, 0.025, True, 0.28, True, 3, 0.07),
    ("Custom Power Supply 12kW", "Custom power supply", "Voltaic Power Systems", "Japan", 17e3, 20, 4,
     ["Zenith Compute Test", "Vector Memory Test", "Nexus System-Level Test"], 4, 1.00, 0.22, 0.020, True, 0.22, True, 5, 0.05),
    ("Power Distribution Module", "Power distribution", "Voltaic Power Systems", "Japan", 6.5e3, 14, 3,
     PRODUCT_FAMILIES, 6, 1.15, 0.10, 0.012, True, 0.18, True, 3, 0.04),
    ("Digital Pin Card PCB Assembly", "Custom PCB assembly", "Meridian Circuits", "Taiwan", 28e3, 16, 3,
     ["Zenith Compute Test", "Vector Memory Test", "Horizon Mobility Test"], 12, 1.05, 0.18, 0.015, True, 0.20, True, 3, 0.09),
    ("Analog Pin Card PCB Assembly", "Custom PCB assembly", "Meridian Circuits", "Taiwan", 24e3, 16, 3,
     ["Atlas Automotive & Industrial Test", "Horizon Mobility Test"], 10, 1.10, 0.15, 0.015, True, 0.20, True, 3, 0.08),
    ("Backplane Assembly", "Custom PCB assembly", "Ridgeway Electronics", "North America", 31e3, 18, 4,
     PRODUCT_FAMILIES, 1, 1.08, 0.12, 0.012, True, 0.22, True, 4, 0.05),
    ("Thermal Management Assembly", "Thermal management", "CryoFlow Thermal", "South Korea", 12e3, 14, 3,
     ["Zenith Compute Test", "Nexus System-Level Test"], 2, 1.05, 0.15, 0.015, True, 0.18, True, 3, 0.05),
    ("Liquid Cooling Manifold", "Thermal management", "CryoFlow Thermal", "South Korea", 8.5e3, 12, 2,
     ["Zenith Compute Test"], 2, 1.10, 0.12, 0.012, True, 0.15, True, 2, 0.06),
    ("Industrial Control Computer", "Industrial computer", "Bastion Compute", "Taiwan", 9e3, 12, 2,
     PRODUCT_FAMILIES, 2, 1.20, 0.08, 0.010, True, 0.12, True, 2, 0.05),
    ("Handler Docking Interface", "Handler interface", "Dockside Automation", "Japan", 15e3, 16, 3,
     ["Horizon Mobility Test", "Atlas Automotive & Industrial Test"], 1, 1.08, 0.12, 0.012, True, 0.18, True, 4, 0.05),
    ("Probe Interface Board Set", "Probe interface hardware", "Cascade Interface Labs", "North America", 19e3, 14, 3,
     ["Zenith Compute Test", "Vector Memory Test"], 2, 1.02, 0.18, 0.015, True, 0.22, True, 4, 0.08),
    ("Precision Mechanical Frame", "Precision mechanical", "Steinmetz Precision", "Europe", 22e3, 12, 2,
     PRODUCT_FAMILIES, 1, 1.18, 0.06, 0.008, False, 0.0, True, 2, 0.03),
    ("Specialized Cable Set", "Specialized cables", "Straitline Interconnect", "Taiwan", 5.5e3, 10, 2,
     PRODUCT_FAMILIES, 3, 1.15, 0.10, 0.010, True, 0.15, True, 2, 0.04),
    ("High-Density Relay Matrix", "Relay / switching", "Kanto Relayworks", "Japan", 7.2e3, 15, 3,
     ["Atlas Automotive & Industrial Test", "Horizon Mobility Test"], 4, 1.06, 0.12, 0.012, True, 0.16, True, 3, 0.05),
    ("RF Front-End Module", "RF instrumentation", "Helvetia Instruments", "Europe", 26e3, 22, 4,
     ["Horizon Mobility Test"], 3, 1.00, 0.20, 0.018, True, 0.26, False, 7, 0.08),
    ("Timing Reference Module", "Precision instrumentation", "ChronoSync Devices", "North America", 11e3, 18, 3,
     ["Zenith Compute Test", "Vector Memory Test", "Nexus System-Level Test"], 2, 1.05, 0.14, 0.014, True, 0.20, True, 4, 0.06),
    ("High-Bandwidth ADC Module", "Data conversion", "ChronoSync Devices", "North America", 16e3, 24, 5,
     ["Zenith Compute Test", "Vector Memory Test"], 5, 0.98, 0.25, 0.020, True, 0.30, False, 6, 0.09),
    ("DC Source-Measure Unit", "Precision instrumentation", "Helvetia Instruments", "Europe", 13e3, 20, 4,
     ["Atlas Automotive & Industrial Test", "Industrial spare"], 6, 1.05, 0.15, 0.014, True, 0.22, True, 5, 0.05),
    ("Vacuum Wafer Chuck Assembly", "Precision mechanical", "Steinmetz Precision", "Europe", 9.5e3, 14, 3,
     ["Vector Memory Test"], 1, 1.12, 0.08, 0.010, True, 0.14, True, 3, 0.04),
    ("Optical Alignment Module", "Optics", "Lumen Optical Works", "Japan", 14e3, 18, 4,
     ["Nexus System-Level Test"], 2, 1.04, 0.14, 0.015, True, 0.20, False, 6, 0.07),
    ("System Chassis Enclosure", "Mechanical enclosure", "Ridgeway Electronics", "North America", 8e3, 10, 2,
     PRODUCT_FAMILIES, 1, 1.25, 0.05, 0.008, True, 0.10, True, 1, 0.02),
    ("EMI Shielding Kit", "Mechanical enclosure", "Ridgeway Electronics", "North America", 2.4e3, 8, 1,
     PRODUCT_FAMILIES, 4, 1.30, 0.04, 0.006, True, 0.08, True, 1, 0.02),
    ("High-Current Connector Set", "Specialized connectors", "Anchor Connect", "China", 3.1e3, 12, 3,
     ["Zenith Compute Test", "Atlas Automotive & Industrial Test", "Nexus System-Level Test"], 8, 1.05, 0.14, 0.015, True, 0.15, True, 2, 0.05),
    ("Fine-Pitch Connector Set", "Specialized connectors", "Anchor Connect", "China", 2.6e3, 12, 3,
     ["Horizon Mobility Test", "Vector Memory Test"], 10, 1.08, 0.12, 0.014, True, 0.14, True, 2, 0.05),
    ("Embedded Memory Module Set", "Memory / storage", "Bastion Compute", "Taiwan", 4.2e3, 14, 3,
     PRODUCT_FAMILIES, 4, 1.12, 0.10, 0.012, True, 0.12, True, 2, 0.06),
    ("Precision Air Bearing Stage", "Precision mechanical", "Steinmetz Precision", "Europe", 27e3, 20, 4,
     ["Nexus System-Level Test"], 1, 1.02, 0.12, 0.014, True, 0.24, False, 8, 0.05),
    ("Environmental Sensor Kit", "Sensors", "Kanto Relayworks", "Japan", 1.8e3, 8, 1,
     PRODUCT_FAMILIES, 6, 1.35, 0.05, 0.006, True, 0.08, True, 1, 0.03),
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _demand_plan(rng: np.random.Generator) -> pd.DataFrame:
    """Lumpy customer x family x month demand plan."""
    months = month_labels()
    rows = []
    for cname, group, region, priority, families in CUSTOMERS:
        for fam in families:
            n_cust_in_fam = sum(1 for c in CUSTOMERS if fam in c[4])
            share = rng.uniform(0.7, 1.3) / n_cust_in_fam
            base_rate = FAMILY_MONTHLY_UNITS[fam] * share
            asp_spec = PRODUCT_SPECS[fam][0]
            # customer-specific ASP reflecting configuration richness
            asp = asp_spec * rng.uniform(0.92, 1.12)
            new_product = fam in ("Nexus System-Level Test",) and rng.random() < 0.5
            for m, month in enumerate(months):
                trend = (1 + FAMILY_TREND[fam]) ** m
                lam = base_rate * trend
                # lumpy orders: negative-binomial-style dispersion plus occasional lump
                units = rng.poisson(lam * rng.gamma(shape=2.2, scale=1 / 2.2))
                if rng.random() < 0.06:  # occasional large fleet order
                    units += int(rng.integers(4, 10))
                horizon_conf = float(np.clip(0.95 - 0.035 * m + rng.normal(0, 0.02), 0.35, 0.98))
                backlog_frac = float(np.clip(1.05 - 0.22 * m + rng.normal(0, 0.05), 0.0, 1.0))
                backlog = int(round(units * backlog_frac))
                forecast = units - backlog
                rows.append({
                    "customer": cname,
                    "customer_group": group,
                    "region": region,
                    "customer_priority": priority,
                    "product_family": fam,
                    "market_segment": FAMILY_MARKET[fam],
                    "month": month,
                    "base_forecast_units": forecast,
                    "bookings_units": backlog,
                    "backlog_units": backlog,
                    "requested_month": months[max(0, m - int(rng.random() < 0.15))],
                    "committed_month": month,
                    "forecast_confidence": horizon_conf,
                    "hist_forecast_error": round(float(rng.uniform(0.08, 0.30)), 3),
                    "demand_std_units": round(units * rng.uniform(0.15, 0.35), 2),
                    "upside_units": int(round(units * rng.uniform(0.05, 0.30))),
                    "downside_units": int(round(units * rng.uniform(0.05, 0.25))),
                    "pull_in_prob": round(float(rng.uniform(0.02, 0.10)), 3),
                    "push_out_prob": round(float(rng.uniform(0.05, 0.18)), 3),
                    "cancel_prob": round(float(rng.uniform(0.005, 0.04)), 3),
                    "asp_usd": round(asp, -3),
                    "config_complexity": PRODUCT_SPECS[fam][11],
                    "new_product_flag": bool(new_product),
                    "site_readiness_prob": round(float(rng.uniform(0.88, 0.99)), 3),
                    "revrec_method": "acceptance" if PRODUCT_SPECS[fam][10] >= 1.0 else "shipment",
                })
    return pd.DataFrame(rows)


def _products() -> pd.DataFrame:
    rows = []
    for fam, spec in PRODUCT_SPECS.items():
        (asp, mat, conv, integ, freight, warranty, cycle, integ_wk, fat_wk, install_wk,
         accept_lag, complexity, fpy, rework, scrap) = spec
        rows.append({
            "product_family": fam,
            "market_segment": FAMILY_MARKET[fam],
            "list_asp_usd": asp,
            "material_cost_usd": round(asp * mat, -3),
            "ems_conversion_cost_usd": round(asp * conv, -3),
            "integration_test_cost_usd": round(asp * integ, -3),
            "freight_cost_usd": round(asp * freight, -3),
            "warranty_reserve_usd": round(asp * warranty, -3),
            "build_cycle_months": cycle,
            "integration_weeks": integ_wk,
            "fat_weeks": fat_wk,
            "install_weeks": install_wk,
            "acceptance_lag_months": accept_lag,
            "config_complexity": complexity,
            "base_first_pass_yield": fpy,
            "rework_prob": rework,
            "scrap_prob": scrap,
            "qualified_ems_sites": ";".join(FAMILY_EMS_QUAL[fam]),
            "alt_source_eligible": fam != "Nexus System-Level Test",
        })
    return pd.DataFrame(rows)


def _components(rng: np.random.Generator, demand: pd.DataFrame) -> pd.DataFrame:
    """Component master with supply pipeline sized against modeled demand."""
    per_month = (
        demand.assign(units=demand["base_forecast_units"] + demand["backlog_units"])
        .groupby(["product_family", "month"])["units"].sum().reset_index()
    )
    # buyers size purchase orders against the demand plan including its growth
    # trend, so use the back half of the horizon rather than the flat average
    later = per_month.groupby("product_family").apply(
        lambda g: g.sort_values("month")["units"].iloc[6:].mean(), include_groups=False)
    fam_monthly = later
    rows = []
    for spec in COMPONENT_SPECS:
        (name, cat, supplier, region, cost, lt, lt_std, fams, usage, tightness,
         alloc, disrupt, exp_ok, exp_prem, alt, alt_lag, obs) = spec
        valid_fams = [f for f in fams if f in PRODUCT_FAMILIES]
        monthly_req = float(sum(fam_monthly.get(f, 0.0) * usage for f in valid_fams))
        on_hand = int(round(monthly_req * rng.uniform(1.1, 1.9)))
        po_monthly = int(round(monthly_req * tightness * rng.uniform(0.97, 1.03)))
        rows.append({
            "component": name,
            "category": cat,
            "supplier": supplier,
            "supplier_region": region,
            "unit_cost_usd": cost,
            "lead_time_weeks": lt,
            "lead_time_std_weeks": lt_std,
            "on_hand_units": on_hand,
            "open_po_units_per_month": po_monthly,
            "min_order_qty": int(max(1, round(monthly_req * 0.25))),
            "safety_stock_units": int(round(monthly_req * rng.uniform(0.4, 0.8))),
            "allocation_risk": alloc,
            "disruption_prob_monthly": disrupt,
            "expedite_available": exp_ok,
            "expedite_premium_pct": exp_prem,
            "alt_source_available": alt,
            "alt_source_qual_months": alt_lag,
            "products_using": ";".join(valid_fams),
            "usage_per_system": usage,
            "obsolescence_risk": obs,
            "monthly_requirement_units": round(monthly_req, 1),
        })
    return pd.DataFrame(rows)


def _ems(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = month_labels()
    site_rows, cap_rows = [], []
    for site, spec in EMS_SPECS.items():
        (cap, cost, region, adherence, fpy, rework, scrap, labor, escape, disrupt,
         logistics_wk, ot_max, flex, ot_prem, resv_fee, min_lot, ramp) = spec
        eligible = [f for f, sites in FAMILY_EMS_QUAL.items() if site in sites]
        site_rows.append({
            "ems_site": site, "region": region, "cost_per_std_unit_usd": cost,
            "overtime_premium_pct": ot_prem, "capacity_reservation_fee_usd": resv_fee,
            "min_production_lot": min_lot, "max_ramp_pct_per_month": ramp,
            "eligible_families": ";".join(eligible),
            "logistics_lead_time_weeks": logistics_wk,
            "quality_escape_prob": escape,
            "regional_disruption_prob_monthly": disrupt,
            "cost_variability_pct": round(float(rng.uniform(0.01, 0.03)), 3),
        })
        for m, month in enumerate(months):
            seasonal = 1.0 + 0.03 * np.sin(2 * np.pi * (m + 3) / 12)
            base = cap * seasonal
            cap_rows.append({
                "ems_site": site, "month": month,
                "available_capacity_units": round(base, 1),
                "reserved_capacity_units": round(base * 0.85, 1),
                "flexible_capacity_units": round(base * flex, 1),
                "max_overtime_units": round(base * ot_max, 1),
                "capacity_cost_per_unit_usd": cost,
                "cycle_time_weeks": round(float(rng.uniform(6.0, 9.0)), 1),
                "schedule_adherence": round(float(np.clip(adherence + rng.normal(0, 0.008), 0.85, 0.99)), 3),
                "first_pass_yield": round(float(np.clip(fpy + rng.normal(0, 0.008), 0.80, 0.99)), 3),
                "rework_rate": rework,
                "scrap_rate": scrap,
                "labor_availability": round(float(np.clip(labor + rng.normal(0, 0.01), 0.85, 1.0)), 3),
            })
    return pd.DataFrame(site_rows), pd.DataFrame(cap_rows)


def _integration(rng: np.random.Generator) -> pd.DataFrame:
    months = month_labels()
    rows = []
    for site, spec in INTEGRATION_SPECS.items():
        integ, calib, fat, install, labor, first_pass, rework_days, accept_wk = spec
        for m, month in enumerate(months):
            rows.append({
                "integration_site": site, "month": month,
                "integration_capacity_units": integ,
                "calibration_capacity_units": calib,
                "fat_capacity_units": fat,
                "installation_capacity_units": install,
                "labor_availability": round(float(np.clip(labor + rng.normal(0, 0.01), 0.85, 1.0)), 3),
                "first_pass_completion": first_pass,
                "rework_days": rework_days,
                "eligible_families": ";".join(PRODUCT_FAMILIES),
                "shipping_lanes": "Global",
                "customer_acceptance_weeks": accept_wk,
            })
    return pd.DataFrame(rows)


def _financial_plan(demand: pd.DataFrame, products: pd.DataFrame,
                    plan_buffer: float) -> pd.DataFrame:
    """Monthly revenue plan: smoothed expected demand revenue with a plan buffer."""
    months = month_labels()
    dem = demand.copy()
    dem["units"] = dem["base_forecast_units"] + dem["backlog_units"]
    dem["revenue"] = dem["units"] * dem["asp_usd"]
    monthly_rev = dem.groupby("month")["revenue"].sum().reindex(months).fillna(0.0)
    # revenue recognizes roughly one month after committed ship for acceptance-based families;
    # the plan applies a simple one-half-month average lag via smoothing
    smoothed = monthly_rev.rolling(3, center=True, min_periods=1).mean() * plan_buffer
    return pd.DataFrame({
        "month": months,
        "revenue_plan_usd": smoothed.round(-5).values,
    })


def generate_all(seed: int = 42, out_dir: Path | None = None,
                 plan_buffer: float = 0.86) -> InputData:
    """Generate every synthetic table and write CSVs. Reproducible for a given seed."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir) if out_dir else DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    demand = _demand_plan(rng)
    products = _products()
    components = _components(rng, demand)
    ems_sites, ems_capacity = _ems(rng)
    integration = _integration(rng)
    fin_plan = _financial_plan(demand, products, plan_buffer)

    tables = {
        "demand_plan": demand, "products": products, "components": components,
        "ems_sites": ems_sites, "ems_capacity": ems_capacity,
        "integration_capacity": integration, "financial_plan": fin_plan,
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)

    return InputData(
        demand=demand, products=products, components=components,
        ems_capacity=ems_capacity, ems_sites=ems_sites,
        integration_capacity=integration, financial_plan=fin_plan, seed=seed,
    )


def load_or_generate(seed: int = 42, out_dir: Path | None = None) -> InputData:
    """Load existing CSVs if present (and seed matches marker), else generate."""
    out = Path(out_dir) if out_dir else DATA_DIR
    marker = out / ".seed"
    names = ["demand_plan", "products", "components", "ems_sites", "ems_capacity",
             "integration_capacity", "financial_plan"]
    if marker.exists() and marker.read_text().strip() == str(seed) and all(
            (out / f"{n}.csv").exists() for n in names):
        frames = {n: pd.read_csv(out / f"{n}.csv") for n in names}
        return InputData(
            demand=frames["demand_plan"], products=frames["products"],
            components=frames["components"], ems_capacity=frames["ems_capacity"],
            ems_sites=frames["ems_sites"],
            integration_capacity=frames["integration_capacity"],
            financial_plan=frames["financial_plan"], seed=seed,
        )
    data = generate_all(seed=seed, out_dir=out)
    marker.write_text(str(seed))
    return data
