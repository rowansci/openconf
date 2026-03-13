#!/usr/bin/env python
"""Benchmark accuracy scaling with conformer count (25 → 50 → 100 → 200 → 400).

Usage:
    pixi run -e bench python scripts/scaling_benchmark.py --n-molecules 30
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench_utils import compute_min_rmsd, load_reference, pct_below, run_etkdg, run_openconf


def benchmark_molecule_scaling(sdf_path: Path, conf_counts: list[int]) -> dict | None:
    """Benchmark scaling for a single molecule."""
    ref = load_reference(sdf_path)
    if ref is None:
        return None
    ref_mol, smiles, n_heavy, n_rotatable = ref

    result: dict[str, Any] = {
        "mol_id": sdf_path.stem.split("_")[0],
        "smiles": smiles,
        "n_heavy_atoms": n_heavy,
        "n_rotatable": n_rotatable,
    }

    for n in conf_counts:
        # OpenConf
        mol, oc_time = run_openconf(smiles, n)
        oc_rmsd = compute_min_rmsd(mol, ref_mol) if mol else float("inf")
        oc_n = mol.GetNumConformers() if mol else 0

        # ETKDG
        mol, et_time = run_etkdg(smiles, n)
        et_rmsd = compute_min_rmsd(mol, ref_mol) if mol else float("inf")
        et_n = mol.GetNumConformers() if mol else 0

        result.update(
            {
                f"openconf_{n}_rmsd": oc_rmsd,
                f"openconf_{n}_time": oc_time,
                f"openconf_{n}_n": oc_n,
                f"etkdg_{n}_rmsd": et_rmsd,
                f"etkdg_{n}_time": et_time,
                f"etkdg_{n}_n": et_n,
            }
        )

    return result


def write_results(results: list[dict], conf_counts: list[int], output_path: Path):
    """Write results to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["mol_id", "smiles", "n_heavy_atoms", "n_rotatable"]
        for n in conf_counts:
            header.extend(
                [
                    f"openconf_{n}_rmsd",
                    f"openconf_{n}_time",
                    f"openconf_{n}_n",
                    f"etkdg_{n}_rmsd",
                    f"etkdg_{n}_time",
                    f"etkdg_{n}_n",
                ]
            )
        writer.writerow(header)

        for r in results:
            row = [r["mol_id"], r["smiles"], r["n_heavy_atoms"], r["n_rotatable"]]
            for n in conf_counts:
                for method in ["openconf", "etkdg"]:
                    rmsd = r.get(f"{method}_{n}_rmsd", float("inf"))
                    row.extend(
                        [
                            f"{rmsd:.4f}" if rmsd < float("inf") else "inf",
                            f"{r.get(f'{method}_{n}_time', 0):.3f}",
                            r.get(f"{method}_{n}_n", 0),
                        ]
                    )
            writer.writerow(row)


def print_summary(results: list[dict], conf_counts: list[int]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)
    print(f"{'Confs':<8} {'OC RMSD':<10} {'ET RMSD':<10} {'OC Time':<10} {'ET Time':<10} {'OC <1Å':<8} {'ET <1Å':<8}")
    print("-" * 80)

    for n in conf_counts:
        oc_rmsds = [
            r[f"openconf_{n}_rmsd"] for r in results if r.get(f"openconf_{n}_rmsd", float("inf")) < float("inf")
        ]
        et_rmsds = [r[f"etkdg_{n}_rmsd"] for r in results if r.get(f"etkdg_{n}_rmsd", float("inf")) < float("inf")]
        oc_times = [r[f"openconf_{n}_time"] for r in results if r.get(f"openconf_{n}_time", 0) > 0]
        et_times = [r[f"etkdg_{n}_time"] for r in results if r.get(f"etkdg_{n}_time", 0) > 0]

        print(
            f"{n:<8} {np.mean(oc_rmsds):<10.3f} {np.mean(et_rmsds):<10.3f} "
            f"{np.mean(oc_times):<10.2f} {np.mean(et_times):<10.2f} "
            f"{pct_below(oc_rmsds, 1.0):<8.1f} {pct_below(et_rmsds, 1.0):<8.1f}"
        )

    print("=" * 80)
    print("\nSpeedup:")
    for n in conf_counts:
        oc_t = np.mean([r[f"openconf_{n}_time"] for r in results if r.get(f"openconf_{n}_time", 0) > 0])
        et_t = np.mean([r[f"etkdg_{n}_time"] for r in results if r.get(f"etkdg_{n}_time", 0) > 0])
        print(f"  {n} confs: {et_t / oc_t:.1f}x")


def plot_scaling(results: list[dict], conf_counts: list[int], output_dir: Path):
    """Generate scaling plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute means
    oc_means, et_means, oc_times, et_times, oc_recalls, et_recalls = [], [], [], [], [], []
    for n in conf_counts:
        oc_rmsds = [
            r[f"openconf_{n}_rmsd"] for r in results if r.get(f"openconf_{n}_rmsd", float("inf")) < float("inf")
        ]
        et_rmsds = [r[f"etkdg_{n}_rmsd"] for r in results if r.get(f"etkdg_{n}_rmsd", float("inf")) < float("inf")]
        oc_t = [r[f"openconf_{n}_time"] for r in results if r.get(f"openconf_{n}_time", 0) > 0]
        et_t = [r[f"etkdg_{n}_time"] for r in results if r.get(f"etkdg_{n}_time", 0) > 0]

        oc_means.append(np.mean(oc_rmsds) if oc_rmsds else float("nan"))
        et_means.append(np.mean(et_rmsds) if et_rmsds else float("nan"))
        oc_times.append(np.mean(oc_t) if oc_t else 0)
        et_times.append(np.mean(et_t) if et_t else 0)
        oc_recalls.append(pct_below(oc_rmsds, 1.0))
        et_recalls.append(pct_below(et_rmsds, 1.0))

    colors = {"oc": "#2E86AB", "et": "#A23B72"}

    for fname, x_oc, x_et, y_oc, y_et, xlabel, ylabel, title in [
        (
            "scaling_rmsd.png",
            conf_counts,
            conf_counts,
            oc_means,
            et_means,
            "Max Conformers",
            "Mean RMSD (Å)",
            "Accuracy vs Conformer Count",
        ),
        (
            "scaling_recall.png",
            conf_counts,
            conf_counts,
            oc_recalls,
            et_recalls,
            "Max Conformers",
            "<1.0Å Recall (%)",
            "Recall vs Conformer Count",
        ),
        (
            "scaling_time.png",
            conf_counts,
            conf_counts,
            oc_times,
            et_times,
            "Max Conformers",
            "Mean Time (s)",
            "Runtime vs Conformer Count",
        ),
        (
            "scaling_pareto.png",
            oc_times,
            et_times,
            oc_means,
            et_means,
            "Mean Time (s)",
            "Mean RMSD (Å)",
            "Accuracy vs Runtime",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_oc, y_oc, "o-", label="OpenConf", color=colors["oc"], linewidth=2, markersize=8)
        ax.plot(x_et, y_et, "s-", label="ETKDG", color=colors["et"], linewidth=2, markersize=8)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        if "Conformers" in xlabel:
            ax.set_xscale("log", base=2)
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=300)
        plt.close()
        print(f"Saved: {output_dir / fname}")


def main():
    parser = argparse.ArgumentParser(description="Conformer count scaling benchmark")
    parser.add_argument("--n-molecules", type=int, default=30)
    parser.add_argument("--conf-counts", type=str, default="25,50,100,200,400")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    conf_counts = [int(x) for x in args.conf_counts.split(",")]
    data_dir = Path(__file__).parent.parent / "static" / "iridium"
    sdf_files = sorted(data_dir.glob("*.sdf"))[: args.n_molecules]

    print("=" * 80)
    print(f"SCALING BENCHMARK (n={len(sdf_files)}, counts={conf_counts})")
    print("=" * 80)

    results = []
    for i, sdf_path in enumerate(sdf_files, 1):
        mol_id = sdf_path.stem.split("_")[0]
        print(f"[{i:3d}/{len(sdf_files)}] {mol_id}...", end=" ", flush=True)

        result = benchmark_molecule_scaling(sdf_path, conf_counts)
        if result:
            results.append(result)
            n = conf_counts[-1]
            print(f"OC:{result[f'openconf_{n}_rmsd']:.3f} ET:{result[f'etkdg_{n}_rmsd']:.3f}")
        else:
            print("SKIP")

    output_path = Path(args.output) if args.output else Path(__file__).parent / "scaling_results.csv"
    write_results(results, conf_counts, output_path)
    print(f"\nResults: {output_path}")

    print_summary(results, conf_counts)
    plot_scaling(results, conf_counts, Path(__file__).parent / "figures")


if __name__ == "__main__":
    main()
