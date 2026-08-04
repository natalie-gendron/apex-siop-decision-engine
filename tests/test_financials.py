"""Financial identities must reconcile arithmetically inside the simulation."""
from __future__ import annotations

import numpy as np


def test_gross_profit_identity(base_result):
    np.testing.assert_allclose(base_result.gross_profit,
                               base_result.revenue - base_result.cogs, rtol=1e-9)


def test_ebitda_identity(base_result, config):
    fin = config.financial
    np.testing.assert_allclose(
        base_result.ebitda,
        base_result.operating_income + fin.depreciation_monthly_usd, rtol=1e-9)


def test_operating_income_identity(base_result, config):
    fin = config.financial
    np.testing.assert_allclose(
        base_result.operating_income,
        base_result.gross_profit - fin.opex_monthly_usd, rtol=1e-9)


def test_inventory_components_sum(base_result):
    np.testing.assert_allclose(
        base_result.inventory,
        base_result.raw_inventory + base_result.wip_inventory
        + base_result.fg_inventory, rtol=1e-9)


def test_cash_flow_identity(base_result, config):
    fin = config.financial
    dwc = np.diff(base_result.working_capital, axis=1,
                  prepend=base_result.working_capital[:, :1])
    expected = (base_result.ebitda - dwc - fin.capex_monthly_usd
                - np.clip(base_result.operating_income, 0, None) * fin.tax_rate)
    np.testing.assert_allclose(base_result.cash_flow, expected, rtol=1e-9)


def test_family_revenue_sums_to_total(base_result):
    np.testing.assert_allclose(base_result.family_revenue.sum(axis=2),
                               base_result.revenue, rtol=1e-9)


def test_margins_in_plausible_band(base_result):
    """Company gross margin should stay in a plausible ATE band on average."""
    fy_rev = base_result.revenue[:, :12].sum(axis=1)
    fy_gp = base_result.gross_profit[:, :12].sum(axis=1)
    gm = fy_gp / fy_rev
    assert 0.45 < gm.mean() < 0.65
