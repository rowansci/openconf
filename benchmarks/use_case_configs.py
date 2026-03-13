#!/usr/bin/env python3
"""Benchmark use-case configurations for openconf.

Four regimes:
  1. max_rapid  - FastROCS-style, 1B-scale virtual screening
  2. ensemble        - Normal conformer ensemble (e.g. logP / property prediction)
  3. spectroscopic   - Boltzmann-weighted NMR/IR ensemble (all accessible conformers)
  4. docking         - Docking pose recovery (bioactive conformation recall)

Run with:
    cd /path/to/openconf
    pixi run -e dev python benchmarks/use_case_configs.py
"""

import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from openconf.api import generate_conformers
from openconf.config import ConformerConfig, PrismConfig

# ---------------------------------------------------------------------------
# Representative molecules
# ---------------------------------------------------------------------------

MOLECULES = {
    "butylbenzene (13 heavy, 3 rotors)": "CCCCc1ccccc1",
    "ibuprofen     (18 heavy, 5 rotors)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "celecoxib     (26 heavy, 4 rotors)": "CC1=CC=C(C=C1)S(=O)(=O)NC2=CC(=NN2C)C3=CC=C(C=C3)F",
    "maraviroc     (34 heavy, 7 rotors)": (
        "Cc5nnc(n5[C@@H]1C[C@H]4CC[C@@H](C1)N4CC[C@H](NC(=O)C2CCC(F)(F)CC2)c3ccccc3)C(C)C"
    ),
}

# ---------------------------------------------------------------------------
# Config presets
# ---------------------------------------------------------------------------

USE_CASE_CONFIGS = {
    # ── 1. Max rapid ─────────────────────────────────────────────────
    # FastROCS-style: enumerate as many molecules/second as possible.
    # A handful of diverse conformers per molecule is sufficient for shape
    # similarity. Skip final MMFF refinement, use coarser seeds, batch the
    # minimization for multi-core efficiency.
    "max_rapid": ConformerConfig(
        max_out=5,
        pool_max=100,
        n_steps=30,
        energy_window_kcal=20.0,
        seed_n_per_rotor=2,
        seed_prune_rms_thresh=1.5,
        do_final_refine=False,
        minimize_batch_size=16,
        dedupe_period=15,
        shake_period=10,
        final_select="diverse",
        random_seed=42,
    ),
    # ── 2. Normal ensemble ────────────────────────────────────────────────
    # Balanced setting for property prediction (logP, pKa, ML descriptors).
    # A medium-sized, well-diversified ensemble at low computational cost.
    "ensemble": ConformerConfig(
        max_out=50,
        pool_max=500,
        n_steps=200,
        energy_window_kcal=10.0,
        seed_n_per_rotor=3,
        seed_prune_rms_thresh=1.0,
        do_final_refine=True,
        minimize_batch_size=8,
        final_select="diverse",
        random_seed=42,
    ),
    # ── 3. Spectroscopic / Boltzmann ensemble ────────────────────────────
    # NMR, IR, VCD: need all thermally-accessible conformers with accurate
    # relative energies for Boltzmann weighting. Tight energy window
    # (~3 kcal ≈ 99 % population at 300 K), energy-ranked final selection,
    # and full MMFF refinement.
    "spectroscopic": ConformerConfig(
        max_out=100,
        pool_max=1000,
        n_steps=400,
        energy_window_kcal=5.0,
        seed_n_per_rotor=5,
        seed_prune_rms_thresh=0.5,
        do_final_refine=True,
        minimize_batch_size=8,
        parent_strategy="softmax",
        final_select="energy",
        random_seed=42,
    ),
    # ── 4. Docking pose recovery ──────────────────────────────────────────
    # The bioactive conformation is often not the global MMFF minimum.
    # Use uniform parent sampling for broad diversity, a wide energy window
    # to keep strained conformers, and skip the final refine step (docking
    # programs do their own minimization inside the binding site).
    "docking": ConformerConfig(
        max_out=250,
        pool_max=2500,
        n_steps=500,
        energy_window_kcal=18.0,
        seed_n_per_rotor=4,
        seed_prune_rms_thresh=0.8,
        do_final_refine=False,
        minimize_batch_size=8,
        parent_strategy="uniform",
        final_select="diverse",
        prism_config=PrismConfig(energy_window_kcal=18.0),
        random_seed=42,
    ),
}

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def bench(smiles: str, config: ConformerConfig, n_runs: int = 3) -> dict:
    """Return mean/min wall time and conformer count over n_runs."""
    times = []
    n_confs = 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        ens = generate_conformers(smiles, config=config)
        times.append(time.perf_counter() - t0)
        n_confs = ens.n_conformers
    return {
        "mean_s": float(np.mean(times)),
        "min_s": float(np.min(times)),
        "n_confs": n_confs,
    }


def sep(title: str) -> None:
    """Print a section separator."""
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print("═" * 64)


if __name__ == "__main__":
    print("=" * 64)
    print("  OpenConf — Use-Case Configuration Benchmark")
    print("=" * 64)

    results: dict[str, dict[str, dict]] = {}

    for use_case, config in USE_CASE_CONFIGS.items():
        sep(use_case)
        results[use_case] = {}
        for label, smi in MOLECULES.items():
            r = bench(smi, config)
            results[use_case][label] = r
            print(f"  {label}:  {r['mean_s']:.3f}s  (min {r['min_s']:.3f}s)  →  {r['n_confs']} confs")

    # Summary table
    sep("Summary table  (mean wall time, seconds)")
    mol_labels = list(MOLECULES.keys())
    col_w = max(len(k) for k in mol_labels) + 2
    use_cases = list(USE_CASE_CONFIGS.keys())

    header = f"{'use case':<20}" + "".join(f"{m:<{col_w}}" for m in mol_labels)
    print(header)
    print("-" * len(header))
    for uc in use_cases:
        row = f"{uc:<20}"
        for mol in mol_labels:
            row += f"{results[uc][mol]['mean_s']:<{col_w}.3f}"
        print(row)

    print("\nDone.")
