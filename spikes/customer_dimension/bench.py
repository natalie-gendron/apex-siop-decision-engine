"""Spike benchmark: allocation kernel runtime and memory vs paths and demand lines.

Usage:  python spikes/customer_dimension/bench.py [--quick]

Reports, per (lines, paths, policy): wall time of the full kernel run (demand
shocks + supply shocks + allocation + revenue), the allocation-only share, and
peak traced memory (numpy allocations, via tracemalloc, measured in a separate
run so tracing does not distort timing). Timing is the best of `REPEATS` runs.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from kernel import SimOptions, simulate  # noqa: E402
from problem import load_problem, scale_lines  # noqa: E402

POLS = ["proportional", "priority_tiered", "strict_greedy"]
DTYPES = ["float64", "float32"]
REPEATS = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    base, cfg = load_problem()
    line_counts = [15, 60] if args.quick else [15, 60, 200]
    paths = [1000, 5000] if args.quick else [1000, 5000, 10000]
    print(f"python {platform.python_version()}, numpy {np.__version__}, {platform.machine()}")
    print("| lines | customers | paths | policy | dtype | total s | alloc s | peak MB |")
    print("|---:|---:|---:|---|---|---:|---:|---:|")
    for n_lines in line_counts:
        prob = base if n_lines == base.n_lines else scale_lines(base, n_lines)
        for n in paths:
            for pol, dt in [(p, d) for p in POLS for d in DTYPES]:
                opt = SimOptions(dtype=dt)
                best_t, best_a = 1e9, 0.0
                for _ in range(REPEATS):
                    r = simulate(prob, cfg, pol, n, opt=opt)
                    if r.t_total < best_t:
                        best_t, best_a = r.t_total, r.t_alloc
                del r
                tracemalloc.start()
                simulate(prob, cfg, pol, n, opt=opt)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                print(f"| {prob.n_lines} | {prob.n_cust} | {n:,} | {pol} | {dt} | "
                      f"{best_t:.2f} | {best_a:.2f} | {peak / 1e6:.0f} |", flush=True)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"bench wall time {time.perf_counter() - t0:.0f}s")
