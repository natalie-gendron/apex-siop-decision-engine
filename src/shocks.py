"""Shock settings: one switch for every source of randomness in the engine.

The volatility assumptions themselves (sigmas, event probabilities) are owned
inputs in `config.uncertainty` and the data tables. A `Shocks` object scales
each named group of them. `Shocks()` applies them as calibrated;
`Shocks.zero()` switches every one off, so the engine becomes deterministic
and its single path is the baseline supply plan (one engine, two uses).

Scenario and response levers (params such as `pushout_prob_add`,
`forced_pushout`, `lead_time_mult`) are not shocks: they still apply in a
zero-shock run, as deterministic what-ifs.

Every draw is taken regardless of amplitude, so the random-number stream (and
common random numbers between runs) never depends on these settings.
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Shocks:
    """Amplitude per shock group: 1.0 = as calibrated, 0.0 = off."""

    demand: float = 1.0                # market and customer demand volatility
    timing_events: float = 1.0         # cancellation, push-out, pull-in rates and their noise
    price_cost: float = 1.0            # ASP, material, FX, conversion and freight volatility
    component_tightness: float = 1.0   # tightness factor: receipt delay, disruption odds, cost
    logistics: float = 1.0             # logistics factor: receipt delay, freight cost
    receipt_delay: float = 1.0         # structural receipt lateness by lead time
    component_disruption: float = 1.0  # supplier disruption events
    ems_execution: float = 1.0         # EMS labor and execution factor (capacity, yield)
    site_disruption: float = 1.0       # EMS regional disruption events
    acceptance_slip: float = 1.0       # customer acceptance / site-readiness slip

    @classmethod
    def zero(cls) -> "Shocks":
        """Every shock off: the deterministic engine."""
        return cls(**{f.name: 0.0 for f in fields(cls)})

    @property
    def is_zero(self) -> bool:
        return all(getattr(self, f.name) == 0.0 for f in fields(self))
