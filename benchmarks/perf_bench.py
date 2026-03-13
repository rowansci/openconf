#!/usr/bin/env python3
"""Micro-benchmark for openconf performance bottlenecks.

Run with:
    cd /path/to/openconf
    pixi run -e dev python benchmarks/perf_bench.py

Measures each hotspot in isolation, then end-to-end timing.
"""

import sys
import time
import timeit

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Add project root to path if running directly
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from openconf.config import ConformerConfig
from openconf.dedupe import _mol_to_arrays
from openconf.perceive import prepare_molecule
from openconf.pool import ConformerPool
from openconf.propose.hybrid import _has_clash

SMILES = {
    "small  (butylbenzene, ~13 heavy)": "CCCCc1ccccc1",
    "medium (ibuprofen, ~18 heavy)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "large  (celecoxib, ~26 heavy)": "CC1=CC=C(C=C1)S(=O)(=O)NC2=CC(=NN2C)C3=CC=C(C=C3)F",
    "xlarge (maraviroc, ~34 heavy)": "Cc5nnc(n5[C@@H]1C[C@H]4CC[C@@H](C1)N4CC[C@H](NC(=O)C2CCC(F)(F)CC2)c3ccccc3)C(C)C",
}


def make_mol(smiles: str) -> Chem.Mol:
    """Prepare a molecule from SMILES with explicit Hs."""
    mol = Chem.MolFromSmiles(smiles)
    return prepare_molecule(mol, add_hs=True)


def embed_confs(mol: Chem.Mol, n: int = 1) -> list[int]:
    """Embed n conformers and return their IDs."""
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params))
    return ids


# ---------------------------------------------------------------------------
# Hotspot 1: _has_clash
# ---------------------------------------------------------------------------


def bench_has_clash(mol: Chem.Mol, n_calls: int = 500) -> float:
    """Return microseconds per call."""
    conf_ids = embed_confs(mol, n=1)
    if not conf_ids:
        return float("nan")
    conf_id = conf_ids[0]
    elapsed = timeit.timeit(lambda: _has_clash(mol, conf_id), number=n_calls)
    return elapsed / n_calls * 1e6


# ---------------------------------------------------------------------------
# Hotspot 2: _mol_to_arrays
# ---------------------------------------------------------------------------


def bench_mol_to_arrays(mol: Chem.Mol, n_confs: int = 100, n_reps: int = 5) -> float:
    """Return milliseconds per call (across n_confs conformers)."""
    conf_ids = embed_confs(mol, n=n_confs)
    if not conf_ids:
        return float("nan")

    elapsed = timeit.timeit(lambda: _mol_to_arrays(mol, conf_ids), number=n_reps)
    return elapsed / n_reps * 1e3


# ---------------------------------------------------------------------------
# Hotspot 3: ConformerPool.insert
# ---------------------------------------------------------------------------


def bench_pool_insert(pool_max: int = 2000, n_inserts: int = 5000) -> float:
    """Return milliseconds for n_inserts insertions into a pool capped at pool_max."""
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    mol = make_mol(smiles)
    config = ConformerConfig(pool_max=pool_max, energy_window_kcal=100.0)

    rng = np.random.default_rng(0)
    energies = rng.uniform(0, 20, n_inserts).tolist()

    pool = ConformerPool(mol, config)
    t0 = time.perf_counter()
    for i, e in enumerate(energies):
        pool.insert(i, e, source="bench")
    return (time.perf_counter() - t0) * 1e3


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def bench_e2e(smiles: str, config: ConformerConfig, n_runs: int = 3) -> dict:
    """Return mean/min runtime and conformer count over n_runs."""
    from openconf.api import generate_conformers

    times = []
    n_confs = 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        ens = generate_conformers(smiles, config=config)
        times.append(time.perf_counter() - t0)
        n_confs = ens.n_conformers
    return {"mean_s": np.mean(times), "min_s": np.min(times), "n_confs": n_confs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def sep(title: str) -> None:
    """Print a section separator."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("  OpenConf Performance Micro-Benchmark")
    print("=" * 60)

    # ── Hotspot 1: _has_clash ──────────────────────────────────────
    sep("Hotspot 1: _has_clash  (μs per call, 500 calls)")
    for label, smi in SMILES.items():
        mol = make_mol(smi)
        us = bench_has_clash(mol, n_calls=500)
        n_atoms = mol.GetNumAtoms()
        print(f"  {label}  [{n_atoms} atoms]: {us:7.1f} μs")

    # ── Hotspot 2: _mol_to_arrays ──────────────────────────────────
    sep("Hotspot 2: _mol_to_arrays  (ms per call, 100 confs)")
    for label, smi in SMILES.items():
        mol = make_mol(smi)
        ms = bench_mol_to_arrays(mol, n_confs=100)
        print(f"  {label}: {ms:7.2f} ms")

    # ── Hotspot 3: ConformerPool.insert ────────────────────────────
    sep("Hotspot 3: ConformerPool.insert  (ms total, 5000 inserts, pool_max=2000)")
    ms = bench_pool_insert()
    print(f"  {ms:.1f} ms")

    # ── End-to-end ─────────────────────────────────────────────────
    sep("End-to-end  (n_steps=300, max_out=50, 3 runs each)")
    cfg = ConformerConfig(n_steps=300, max_out=50, random_seed=42)
    for label, smi in SMILES.items():
        r = bench_e2e(smi, cfg, n_runs=3)
        print(f"  {label}: {r['mean_s']:.2f}s  ({r['n_confs']} confs)")

    print("\nDone.")
