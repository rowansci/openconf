#!/usr/bin/env python
"""Fair conformer benchmark: multiple baselines, budgets, seeds, shared postprocessing.

Each method's raw output passes through the same canonical MMFF minimization
(for ETKDG variants) and RMSD-based pruning, so results reflect method quality
rather than differences in postprocessing.

Methods:
-------
  etkdg_raw        ETKDGv3 only, no minimization
  etkdg_mmff       ETKDGv3 + shared MMFF minimization
  etkdg_oversample ETKDGv3 x5 oversampled, shared MMFF, energy-ranked to N
  openconf         OpenConf (internal MMFF + PRISM, wide-window docking config)

Tracks
------
  A  RMSD / TFD to Iridium reference conformer
  B  Lowest MMFF energy found (energy gap vs pooled best-known)
  C  Ensemble diversity after shared RMSD pruning

Quick test (~3 min on M2, 20 molecules):
    pixi run -e bench python scripts/fair_benchmark.py --n-molecules 20 --budgets 50 --seeds 2

Recommended first benchmark (~30 min):
    pixi run -e bench python scripts/fair_benchmark.py --n-molecules 100 --budgets 10,50,200 --seeds 3

Full benchmark:
    pixi run -e bench python scripts/fair_benchmark.py --budgets 10,50,200 --seeds 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_lib import (
    MolRecord,
    MolRunRecord,
    RunResult,
    add_energy_gaps,
    bootstrap_ci,
    compute_best_rmsd,
    compute_best_tfd,
    load_iridium,
    pairwise_rmsd_stats,
    run_etkdg_mmff,
    run_etkdg_oversample,
    run_etkdg_raw,
    run_openconf,
    shared_rmsd_prune,
)

# ──────────────────────────────────────────────────────────────────────────────
# Method registry
# ──────────────────────────────────────────────────────────────────────────────

METHODS = {
    "etkdg_raw": run_etkdg_raw,
    "etkdg_mmff": run_etkdg_mmff,
    "etkdg_oversample": run_etkdg_oversample,
    "openconf": run_openconf,
}

DEFAULT_PRUNE_THRESHOLD = 1.0  # Å

# ──────────────────────────────────────────────────────────────────────────────
# Core run logic
# ──────────────────────────────────────────────────────────────────────────────


def run_single(
    mol_record: MolRecord,
    method_name: str,
    budget: int,
    seed: int,
    prune_threshold: float,
) -> MolRunRecord:
    """Execute one (method, budget, seed) combination and compute all metrics."""
    fn = METHODS[method_name]
    raw: RunResult = fn(mol_record.smiles, budget, seed)

    if raw.failure or raw.mol is None or not raw.conf_ids:
        return MolRunRecord(
            mol_id=mol_record.mol_id,
            method=method_name,
            budget=budget,
            seed=seed,
            runtime_s=raw.runtime_s,
            n_raw=raw.n_raw,
            n_shared_pruned=0,
            best_rmsd=float("inf"),
            best_tfd=None,
            lowest_energy_kcal=float("inf"),
            median_pairwise_rmsd=None,
            failure=raw.failure or "no_conformers",
        )

    # Shared RMSD pruning (same threshold for all methods)
    pruned_ids = shared_rmsd_prune(raw.mol, raw.conf_ids, raw.energies, prune_threshold)

    # Track A: RMSD and TFD vs Iridium reference
    best_rmsd = compute_best_rmsd(raw.mol, mol_record.ref_mol, pruned_ids)
    best_tfd = compute_best_tfd(raw.mol, mol_record.ref_mol, pruned_ids)

    # Track B: lowest energy in pruned set
    energy_map = dict(zip(raw.conf_ids, raw.energies))
    pruned_energies = [energy_map.get(cid, float("inf")) for cid in pruned_ids]
    lowest_energy = min(pruned_energies) if pruned_energies else float("inf")

    # Track C: ensemble diversity (capped to avoid O(N²) blowup)
    med_pairwise = pairwise_rmsd_stats(raw.mol, pruned_ids)

    return MolRunRecord(
        mol_id=mol_record.mol_id,
        method=method_name,
        budget=budget,
        seed=seed,
        runtime_s=raw.runtime_s,
        n_raw=raw.n_raw,
        n_shared_pruned=len(pruned_ids),
        best_rmsd=best_rmsd,
        best_tfd=best_tfd,
        lowest_energy_kcal=lowest_energy,
        median_pairwise_rmsd=med_pairwise,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

_FIELDS = [
    "mol_id",
    "method",
    "budget",
    "seed",
    "runtime_s",
    "n_raw",
    "n_shared_pruned",
    "best_rmsd",
    "best_tfd",
    "success_05",
    "success_10",
    "success_15",
    "lowest_energy_kcal",
    "energy_gap_kcal",
    "median_pairwise_rmsd",
    "failure",
]


def write_results_csv(records: list[MolRunRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    "mol_id": r.mol_id,
                    "method": r.method,
                    "budget": r.budget,
                    "seed": r.seed,
                    "runtime_s": f"{r.runtime_s:.3f}",
                    "n_raw": r.n_raw,
                    "n_shared_pruned": r.n_shared_pruned,
                    "best_rmsd": f"{r.best_rmsd:.4f}" if r.best_rmsd < float("inf") else "inf",
                    "best_tfd": f"{r.best_tfd:.4f}" if r.best_tfd is not None else "",
                    "success_05": r.success_05,
                    "success_10": r.success_10,
                    "success_15": r.success_15,
                    "lowest_energy_kcal": (
                        f"{r.lowest_energy_kcal:.3f}" if r.lowest_energy_kcal < float("inf") else "inf"
                    ),
                    "energy_gap_kcal": (f"{r.energy_gap_kcal:.3f}" if r.energy_gap_kcal < float("inf") else "inf"),
                    "median_pairwise_rmsd": (
                        f"{r.median_pairwise_rmsd:.3f}" if r.median_pairwise_rmsd is not None else ""
                    ),
                    "failure": r.failure or "",
                }
            )


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────


def _agg_by_mol(records: list[MolRunRecord], method: str, budget: int) -> dict[str, list[float]]:
    """Aggregate seed runs per molecule. Returns per-mol best-RMSD, mean-time, etc."""
    recs = [r for r in records if r.method == method and r.budget == budget and not r.failure]
    mol_ids = sorted({r.mol_id for r in recs})
    agg: dict[str, list] = {
        "rmsd": [],
        "time": [],
        "unique": [],
        "div": [],
        "s10": [],
        "s15": [],
        "gap": [],
    }
    for mol_id in mol_ids:
        mol_recs = [r for r in recs if r.mol_id == mol_id]
        # best over seeds for RMSD (as per spec: best over 5 seeds per molecule)
        best_rmsd = min(r.best_rmsd for r in mol_recs)
        agg["rmsd"].append(best_rmsd)
        agg["time"].append(float(np.mean([r.runtime_s for r in mol_recs])))
        agg["unique"].append(float(np.mean([r.n_shared_pruned for r in mol_recs])))
        divs = [r.median_pairwise_rmsd for r in mol_recs if r.median_pairwise_rmsd is not None]
        if divs:
            agg["div"].append(float(np.mean(divs)))
        agg["s10"].append(float(best_rmsd < 1.0))
        agg["s15"].append(float(best_rmsd < 1.5))
        gaps = [r.energy_gap_kcal for r in mol_recs if r.energy_gap_kcal < float("inf")]
        if gaps:
            agg["gap"].append(float(np.min(gaps)))
    return agg


def print_summary(
    records: list[MolRunRecord],
    methods: list[str],
    budgets: list[int],
) -> None:
    """Overall performance table (Track A + C)."""
    print("\n" + "=" * 110)
    print("FAIR BENCHMARK — TRACK A (recovery) + TRACK C (diversity)")
    print("=" * 110)

    for budget in budgets:
        all_recs = [r for r in records if r.budget == budget]
        n_mols_total = len({r.mol_id for r in all_recs})
        print(f"\n--- Budget N={budget}  ({n_mols_total} molecules) ---")
        hdr = (
            f"{'Method':<22} {'N mol':<7} {'Med RMSD':<10} {'<1.0Å%':<8} {'<1.5Å%':<8} "
            f"{'Med t(s)':<9} {'Fail%':<7} {'Med uniq':<9} {'Med div':<9}"
        )
        print(hdr)
        print("-" * 110)

        for method in methods:
            all_m = [r for r in records if r.method == method and r.budget == budget]
            fail_pct = 100 * sum(1 for r in all_m if r.failure) / len(all_m) if all_m else 100.0
            agg = _agg_by_mol(records, method, budget)

            if not agg["rmsd"]:
                print(f"{method:<22} {'0':<7} {'N/A':>10}")
                continue

            med_rmsd = float(np.median(agg["rmsd"]))
            lo, hi = bootstrap_ci(agg["rmsd"], n_boot=500)
            pct_10 = 100 * float(np.mean(agg["s10"]))
            pct_15 = 100 * float(np.mean(agg["s15"]))
            med_t = float(np.median(agg["time"]))
            med_uniq = float(np.median(agg["unique"]))
            med_div = float(np.median(agg["div"])) if agg["div"] else float("nan")

            print(
                f"{method:<22} {len(agg['rmsd']):<7} "
                f"{med_rmsd:<10.3f} {pct_10:<8.1f} {pct_15:<8.1f} "
                f"{med_t:<9.2f} {fail_pct:<7.1f} {med_uniq:<9.1f} "
                f"{med_div:<9.3f}"
                f"  [{lo:.3f}-{hi:.3f} 95% CI RMSD]"
            )

    print("=" * 110)


def print_energy_summary(
    records: list[MolRunRecord],
    methods: list[str],
    budgets: list[int],
) -> None:
    """Track B: energy gap table."""
    print("\n" + "=" * 80)
    print("FAIR BENCHMARK — TRACK B (energy gap to best-known)")
    print("=" * 80)

    for budget in budgets:
        print(f"\n--- Budget N={budget} ---")
        print(f"{'Method':<22} {'N mol':<7} {'Med gap':<10} {'<0.5 kcal%':<12} {'<1.0 kcal%':<12}")
        print("-" * 65)
        for method in methods:
            agg = _agg_by_mol(records, method, budget)
            gaps = agg["gap"]
            if not gaps:
                print(f"{method:<22} {'0':<7}")
                continue
            med_gap = float(np.median(gaps))
            pct_05 = 100 * sum(g < 0.5 for g in gaps) / len(gaps)
            pct_10 = 100 * sum(g < 1.0 for g in gaps) / len(gaps)
            print(f"{method:<22} {len(gaps):<7} {med_gap:<10.3f} {pct_05:<12.1f} {pct_10:<12.1f}")

    print("=" * 80)


def print_stratified(
    records: list[MolRunRecord],
    mol_map: dict[str, MolRecord],
    budget: int,
    methods: list[str],
) -> None:
    """Stratified Track A results by rotatable bond count and macrocycle flag."""
    strata = {
        "rb 0-3": lambda m: m.n_rotatable <= 3,
        "rb 4-6": lambda m: 4 <= m.n_rotatable <= 6,
        "rb 7-9": lambda m: 7 <= m.n_rotatable <= 9,
        "rb 10+": lambda m: m.n_rotatable >= 10,
        "macrocycle": lambda m: m.is_macrocycle,
    }

    print(f"\n--- Stratified results @ N={budget} ---")
    for stratum_name, pred in strata.items():
        mol_ids = {mol_id for mol_id, m in mol_map.items() if pred(m)}
        stratum_recs = [r for r in records if r.budget == budget and r.mol_id in mol_ids and not r.failure]
        if not stratum_recs:
            continue
        print(f"\n  {stratum_name} (n={len(mol_ids)} molecules)")
        print(f"  {'Method':<22} {'Med RMSD':<10} {'<1.0Å%':<10}")
        for method in methods:
            m_recs = [r for r in stratum_recs if r.method == method]
            mol_ids_m = sorted({r.mol_id for r in m_recs})
            if not mol_ids_m:
                continue
            rmsds = [min(r.best_rmsd for r in m_recs if r.mol_id == mid) for mid in mol_ids_m]
            pct_10 = 100 * sum(v < 1.0 for v in rmsds) / len(rmsds)
            print(f"  {method:<22} {float(np.median(rmsds)):<10.3f} {pct_10:<10.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-molecules", type=int, default=None, help="Limit dataset size")
    parser.add_argument(
        "--budgets",
        type=str,
        default="10,50,200",
        help="Comma-separated conformer budgets (default: 10,50,200)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds 0..N-1 (default: 3)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(METHODS.keys()),
        help=f"Comma-separated methods (default: {','.join(METHODS.keys())})",
    )
    parser.add_argument(
        "--prune-threshold",
        type=float,
        default=DEFAULT_PRUNE_THRESHOLD,
        help="Shared RMSD pruning threshold in Å (default: 1.0)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    seeds = list(range(args.seeds))
    methods = [m.strip() for m in args.methods.split(",")]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        print(f"Error: unknown methods: {unknown}. Available: {list(METHODS)}")
        sys.exit(1)

    data_dir = args.data_dir or str(Path(__file__).parent.parent / "static" / "iridium")

    print("=" * 80)
    print("FAIR CONFORMER BENCHMARK")
    print("=" * 80)
    print(f"Dataset:  {data_dir}")
    print(f"Methods:  {methods}")
    print(f"Budgets:  {budgets}")
    print(f"Seeds:    {seeds}")
    print(f"Prune:    {args.prune_threshold} Å shared RMSD threshold")

    mol_records = load_iridium(data_dir, args.n_molecules)
    mol_map = {r.mol_id: r for r in mol_records}
    macrocycles = sum(1 for m in mol_records if m.is_macrocycle)
    print(f"Loaded:   {len(mol_records)} molecules ({macrocycles} macrocycles)")

    total_runs = len(mol_records) * len(methods) * len(budgets) * len(seeds)
    print(f"Runs:     {total_runs} total")
    print("=" * 80)

    all_records: list[MolRunRecord] = []
    run_idx = 0
    for mol_rec in mol_records:
        for method in methods:
            for budget in budgets:
                for seed in seeds:
                    run_idx += 1
                    prefix = f"[{run_idx:4d}/{total_runs}] {mol_rec.mol_id:<8} {method:<22} N={budget:<4} s={seed}"
                    print(prefix, end=" ", flush=True)

                    rec = run_single(mol_rec, method, budget, seed, args.prune_threshold)
                    all_records.append(rec)

                    if rec.failure:
                        print(f"FAIL({rec.failure})")
                    else:
                        tfd_str = f" tfd={rec.best_tfd:.3f}" if rec.best_tfd is not None else ""
                        print(f"rmsd={rec.best_rmsd:.3f}{tfd_str} uniq={rec.n_shared_pruned} t={rec.runtime_s:.2f}s")

    # Post-process: energy gaps require pooled best-known across all methods
    add_energy_gaps(all_records)

    # Write CSV
    output_path = Path(args.output) if args.output else Path(__file__).parent / "fair_benchmark_results.csv"
    write_results_csv(all_records, output_path)
    print(f"\nResults: {output_path}")

    # Print summaries
    print_summary(all_records, methods, budgets)
    print_energy_summary(all_records, methods, budgets)

    # Stratified at the middle budget
    mid_budget = budgets[len(budgets) // 2]
    print_stratified(all_records, mol_map, mid_budget, methods)


if __name__ == "__main__":
    main()
