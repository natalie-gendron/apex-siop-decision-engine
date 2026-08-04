"""Common-factor correlation model.

Every stochastic variable loads on one or more independent standard-normal
factors (config: `factors.loadings`). A variable with loadings {f1: a, f2: b}
receives the shock  a*F1 + b*F2 + sqrt(1 - a^2 - b^2) * idiosyncratic.
Because factors are independent and loadings satisfy sum(a^2) <= 1, the implied
correlation matrix is positive semidefinite by construction — correlations can
never produce impossible covariance structures.

Factor paths follow an AR(1) process through the 18 months so that market
cycles persist rather than resetting each month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FactorModel


class FactorEngine:
    """Draws correlated shocks for named variables from the factor model."""

    def __init__(self, model: FactorModel):
        self.model = model
        self.factor_names = model.names
        self.var_names = list(model.loadings.keys())
        self._loadings = np.zeros((len(self.var_names), len(self.factor_names)))
        for i, var in enumerate(self.var_names):
            for fname, w in model.loadings[var].items():
                if fname not in self.factor_names:
                    raise ValueError(f"Unknown factor '{fname}' in loadings for '{var}'")
                self._loadings[i, self.factor_names.index(fname)] = w
        self._idio = np.sqrt(np.clip(1.0 - (self._loadings ** 2).sum(axis=1), 0.0, 1.0))

    def draw_factor_paths(self, rng: np.random.Generator, n_sims: int,
                          n_months: int) -> np.ndarray:
        """AR(1) standard-normal factor paths, shape (n_sims, n_months, n_factors)."""
        rho = self.model.persistence
        k = len(self.factor_names)
        paths = np.empty((n_sims, n_months, k))
        paths[:, 0, :] = rng.standard_normal((n_sims, k))
        innov = rng.standard_normal((n_sims, n_months - 1, k)) * np.sqrt(1 - rho ** 2)
        for t in range(1, n_months):
            paths[:, t, :] = rho * paths[:, t - 1, :] + innov[:, t - 1, :]
        return paths

    def shock(self, var: str, factor_paths: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
        """Correlated standard-normal shock for a named variable,
        shape (n_sims, n_months)."""
        i = self.var_names.index(var)
        systematic = factor_paths @ self._loadings[i]
        idio = rng.standard_normal(factor_paths.shape[:2])
        return systematic + self._idio[i] * idio

    def implied_correlation(self) -> pd.DataFrame:
        """The correlation matrix implied by the loadings (for display)."""
        corr = self._loadings @ self._loadings.T
        np.fill_diagonal(corr, 1.0)
        return pd.DataFrame(corr, index=self.var_names, columns=self.var_names)

    def loadings_table(self) -> pd.DataFrame:
        """Factor-loading matrix (variables x factors) for display."""
        return pd.DataFrame(self._loadings, index=self.var_names,
                            columns=self.factor_names)
