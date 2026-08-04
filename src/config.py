"""Configuration loading and typed settings for the SIOP engine."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "generated"


class SimulationSettings(BaseModel):
    quick_sims: int = 1000
    standard_sims: int = 5000
    detailed_sims: int = 10000
    default_mode: str = "standard"


class FinancialAssumptions(BaseModel):
    gross_margin_target: float = Field(gt=0, lt=1)
    inventory_target_usd: float = Field(gt=0)
    inventory_turns_target: float = Field(gt=0)
    opex_monthly_usd: float = Field(ge=0)
    depreciation_monthly_usd: float = Field(ge=0)
    capex_monthly_usd: float = Field(ge=0)
    tax_rate: float = Field(ge=0, lt=1)
    dso_days: float = Field(ge=0)
    dpo_days: float = Field(ge=0)
    eo_reserve_rate: float = Field(ge=0, le=1)
    revenue_plan_buffer: float = Field(gt=0.5, le=1.1)


class UncertaintySettings(BaseModel):
    market_demand_sigma: dict[str, float]
    customer_idiosyncratic_sigma: float = Field(ge=0)
    asp_sigma: float = Field(ge=0)
    material_cost_sigma: float = Field(ge=0)
    fx_cost_sigma: float = Field(ge=0)
    conversion_cost_sigma: float = Field(ge=0)
    freight_sigma: float = Field(ge=0)
    ems_labor_sigma: float = Field(ge=0)
    utilization_adherence_penalty: float = Field(ge=0, le=0.5)
    site_disruption_impact: float = Field(ge=0, le=1)

    @field_validator("market_demand_sigma")
    @classmethod
    def _sigmas_nonneg(cls, v: dict[str, float]) -> dict[str, float]:
        if any(s < 0 for s in v.values()):
            raise ValueError("market demand sigmas must be non-negative")
        return v


class FactorModel(BaseModel):
    names: list[str]
    persistence: float = Field(ge=0, lt=1)
    loadings: dict[str, dict[str, float]]

    @field_validator("loadings")
    @classmethod
    def _loadings_bounded(cls, v: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        for var, load in v.items():
            total = sum(w * w for w in load.values())
            if total > 1.0 + 1e-9:
                raise ValueError(
                    f"loadings for '{var}' imply variance > 1 (sum of squares = {total:.3f}); "
                    "reduce loadings so the implied correlation matrix stays valid"
                )
        return v


class AppConfig(BaseModel):
    random_seed: int = 42
    n_months: int = 18
    simulation: SimulationSettings
    financial: FinancialAssumptions
    uncertainty: UncertaintySettings
    factors: FactorModel


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load and validate the YAML configuration."""
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppConfig(**raw)
