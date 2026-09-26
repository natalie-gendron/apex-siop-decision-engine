"""The variable map must cover every input that can change a result."""
from __future__ import annotations

from dataclasses import fields

from src.config import FinancialAssumptions, UncertaintySettings
from src.shocks import Shocks
from src.simulation import default_params
from src.variable_map import LAYERS, mapped_names, variable_map

# data-table columns read by operations.build_planning_arrays and the engine
ENGINE_DATA_COLUMNS = {
    "base_forecast_units", "backlog_units", "asp_usd", "push_out_prob", "pull_in_prob",
    "cancel_prob", "site_readiness_prob", "acceptance_lag_months", "config_complexity",
    "qualified_ems_sites", "material_cost_usd", "ems_conversion_cost_usd",
    "integration_test_cost_usd", "freight_cost_usd", "warranty_reserve_usd", "scrap_prob",
    "build_cycle_months", "on_hand_units", "open_po_units_per_month", "usage_per_system",
    "products_using", "safety_stock_units", "lead_time_weeks", "allocation_risk",
    "disruption_prob_monthly", "expedite_available", "expedite_premium_pct",
    "unit_cost_usd", "available_capacity_units", "max_overtime_units",
    "schedule_adherence", "labor_availability", "first_pass_yield",
    "cost_per_std_unit_usd", "overtime_premium_pct", "regional_disruption_prob_monthly",
    "revenue_plan_usd",
}


def test_every_lever_is_mapped():
    assert set(default_params()) - mapped_names() == set()


def test_every_shock_switch_is_mapped():
    assert {f.name for f in fields(Shocks)} - mapped_names() == set()


def test_every_config_setting_is_mapped():
    names = set(UncertaintySettings.model_fields) | set(FinancialAssumptions.model_fields)
    assert names - mapped_names() == set()


def test_every_engine_data_column_is_mapped():
    assert ENGINE_DATA_COLUMNS - mapped_names() == set()


def test_map_is_well_formed():
    df = variable_map()
    assert set(df["Layer"]) <= set(LAYERS)
    assert not df["Variable"].duplicated().any()
    assert df.notna().all().all()
