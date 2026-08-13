"""Chart title wrapping — keeps the hover toolbar off the title text.

Plotly draws its toolbar in the top-right of the figure and does not wrap
titles, so a long single-line title runs under the icons (and clips in a
narrow column). `_wrap_title` breaks the title at a meaningful boundary
instead; these tests pin the boundaries so a future title edit can't
silently reintroduce the overlap.
"""
from __future__ import annotations

import re

import pytest

from src import visualizations as viz


def lines(title: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", part)
            for part in viz._wrap_title(title).split("<br>")]


def test_short_title_is_untouched():
    assert lines("Q1 revenue vs plan") == ["Q1 revenue vs plan"]


def test_context_suffix_moves_to_second_line():
    got = lines("FY gross margin vs target — EMS Malaysia Disruption "
                "+ response package (2 actions)")
    assert got == ["FY gross margin vs target",
                   "EMS Malaysia Disruption + response package (2 actions)"]


def test_trailing_parenthetical_moves_to_second_line():
    assert lines("Month-end inventory trajectory vs target (18-month horizon)") \
        == ["Month-end inventory trajectory vs target", "(18-month horizon)"]


def test_break_never_lands_inside_parentheses():
    """An em dash inside a parenthetical is not a title/context boundary."""
    got = lines("Ranked drivers of Q1 revenue (Spearman rank correlation — "
                "association, not causation)")
    assert got[0] == "Ranked drivers of Q1 revenue"
    assert got[1].count("(") == got[1].count(")") == 1


def test_every_line_fits_a_narrow_column():
    """Two charts side by side on a tablet leave ~30 characters per line at
    the title font size; every shipped title must survive that."""
    from src.baseline_plan import run_baseline  # noqa: F401  (import cost only)
    titles = [
        "Q1 revenue vs plan", "FY gross margin vs target",
        "Past-due backlog by month — base case",
        "Past-due backlog by family",
        "Risk matrix — FY shortfall vs FY revenue at risk",
        "FY revenue at risk by family (plan vs P5)",
        "Capacity utilization by month",
    ]
    for t in titles:
        assert len(lines(t)[0]) <= 34, f"{t!r} first line too long for a column"


@pytest.mark.parametrize("title", [
    "Full-year scenario deltas versus base case",
    "EMS network: planned load vs effective capacity",
    "FY revenue uncertainty by Demand Confidence level",
])
def test_wrapping_preserves_the_words(title):
    assert " ".join(lines(title)).split() == title.split()
