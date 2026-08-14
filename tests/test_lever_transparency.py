"""Every scenario and action must state its levers in business language.

The claim-sheet idea only works if the claim is legible: an executive who
cannot see that "EMS Malaysia Disruption" means 45% of one site's capacity
for three months cannot dispute it. These tests fail if a knob is added
without a translation (it would surface as a raw `key = value` dump, or a
numpy array printed into the UI).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.scenarios import (
    describe_overrides,
    management_actions,
    prebuilt_scenarios,
    scenario_assumptions_table,
)


def test_every_scenario_lever_is_translated():
    for name, spec in prebuilt_scenarios().items():
        text = describe_overrides(spec.overrides)
        for key in spec.overrides:
            assert f"{key} =" not in text, (
                f"{name}: '{key}' has no business-language translation")
        assert "array(" not in text and "[" not in text, (
            f"{name}: raw sequence leaked into the description — {text}")


def test_every_action_lever_is_translated():
    for name, spec in management_actions().items():
        text = describe_overrides(spec.overrides)
        for key in spec.overrides:
            assert f"{key} =" not in text, (
                f"{name}: '{key}' has no business-language translation")


def test_scenario_table_covers_every_scenario():
    df = scenario_assumptions_table()
    scen = prebuilt_scenarios()
    assert len(df) == len(scen)
    assert set(df["Scenario"]) == set(scen)
    for _, row in df.iterrows():
        assert row["What it assumes"].strip()
        assert row["Authored levers (the world)"].strip()


def test_base_case_states_it_has_no_overrides():
    df = scenario_assumptions_table().set_index("Scenario")
    assert "no overrides" in df.loc["Base Case", "Authored levers (the world)"]


@pytest.mark.parametrize("overrides, expected", [
    ({"ems_window_mult": {"EMS Malaysia": (1, 4, 0.55)}},
     "EMS Malaysia: capacity ×0.55 in months 2-4"),
    ({"forced_pushout": {"family": "Zenith Compute Test", "from_month": 1,
                         "to_month": 4, "units": 15}},
     "15 Zenith Compute Test systems moved from month 2 to month 5"),
    ({"adherence_delta": -0.02}, "schedule adherence -2 pts"),
    ({"acceptance_delay_add": 0.15}, "acceptance-slip probability +15 pts"),
])
def test_world_knobs_read_the_way_people_say_them(overrides, expected):
    assert describe_overrides(overrides) == expected


def test_demand_profile_is_summarized_not_dumped():
    ramp = np.concatenate([np.linspace(1.0, 1.3, 12), np.full(6, 1.3)])
    assert describe_overrides({"demand_market_mult": {"Memory": ramp}}) == \
        "Memory demand ×1 rising to ×1.3"
