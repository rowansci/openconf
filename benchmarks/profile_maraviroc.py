#!/usr/bin/env python3
"""Profile openconf on maraviroc to find remaining bottlenecks.

Run with:
    pixi run -e dev python benchmarks/profile_maraviroc.py
"""

import cProfile
import io
import pstats
import time

import numpy as np

from openconf.api import generate_conformers
from openconf.config import ConformerConfig

# Maraviroc: ~50 heavy atoms, 9 rotatable bonds, a piperidine ring
# Kekulé form that RDKit accepts; replace with canonical SMILES if available
MARAVIROC = "Cc5nnc(n5[C@@H]1C[C@H]4CC[C@@H](C1)N4CC[C@H](NC(=O)C2CCC(F)(F)CC2)c3ccccc3)C(C)C"

CFG = ConformerConfig(n_steps=500, max_out=100, random_seed=42)

# ── Warm timing run ──────────────────────────────────────────────────────────
print("Timing maraviroc (3 runs, n_steps=500):")
times = []
for i in range(3):
    t0 = time.perf_counter()
    ens = generate_conformers(MARAVIROC, config=CFG)
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    print(f"  run {i + 1}: {elapsed:.2f}s  ({ens.n_conformers} confs)")
print(f"  mean={np.mean(times):.2f}s\n")

# ── cProfile ────────────────────────────────────────────────────────────────
print("Profiling (1 run, n_steps=500):")
pr = cProfile.Profile()
pr.enable()
ens = generate_conformers(MARAVIROC, config=CFG)
pr.disable()

buf = io.StringIO()
ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
ps.print_stats(30)
print(buf.getvalue())

# ── Also print tottime (exclusive) top 20 ───────────────────────────────────
print("\nTop 20 by tottime (exclusive time in function):")
buf2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=buf2).sort_stats("tottime")
ps2.print_stats(20)
print(buf2.getvalue())
