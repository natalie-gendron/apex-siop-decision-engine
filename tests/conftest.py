"""Shared fixtures: one generated dataset, baseline and base simulation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline_plan import run_baseline          # noqa: E402
from src.config import load_config                  # noqa: E402
from src.data_generator import generate_all         # noqa: E402
from src.simulation import run_simulation           # noqa: E402

SEED = 42
N_SIMS = 1500


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def data(config, tmp_path_factory):
    out = tmp_path_factory.mktemp("generated")
    return generate_all(seed=SEED, out_dir=out)


@pytest.fixture(scope="session")
def baseline(data, config):
    return run_baseline(data, config)


@pytest.fixture(scope="session")
def base_result(data, config, baseline):
    return run_simulation(data, config, baseline, n_sims=N_SIMS, seed=SEED)
