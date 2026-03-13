#!/usr/bin/env python
"""Comprehensive benchmark comparing OpenConf, ETKDG, and CREST.

Usage:
    pixi run -e bench python scripts/comprehensive_benchmark.py --dataset iridium --skip-crest
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench_utils import compute_min_rmsd, load_reference, pct_below, run_etkdg, run_openconf


@dataclass
class MethodResult:
    method: str
    n_conformers: int = 0
    time_sec: float = 0.0
    min_rmsd: float = float("inf")
    success: bool = True


@dataclass
class MoleculeResult:
    mol_id: str
    smiles: str
    n_heavy_atoms: int
    n_rotatable: int
    methods: dict[str, MethodResult] = field(default_factory=dict)


def benchmark_molecule(sdf_path: Path, max_confs: int = 200, include_crest: bool = False) -> MoleculeResult | None:
    """Benchmark all methods on a single molecule."""
    ref = load_reference(sdf_path)
    if ref is None:
        return None
    ref_mol, smiles, n_heavy, n_rotatable = ref

    result = MoleculeResult(
        mol_id=sdf_path.stem.split("_")[0],
        smiles=smiles,
        n_heavy_atoms=n_heavy,
        n_rotatable=n_rotatable,
    )

    # OpenConf
    mol, time_sec = run_openconf(smiles, max_confs)
    if mol:
        result.methods["openconf"] = MethodResult(
            "openconf", mol.GetNumConformers(), time_sec, compute_min_rmsd(mol, ref_mol)
        )
    else:
        result.methods["openconf"] = MethodResult("openconf", success=False)

    # ETKDG
    mol, time_sec = run_etkdg(smiles, max_confs)
    if mol:
        result.methods["etkdg"] = MethodResult(
            "etkdg", mol.GetNumConformers(), time_sec, compute_min_rmsd(mol, ref_mol)
        )
    else:
        result.methods["etkdg"] = MethodResult("etkdg", success=False)

    # CREST (optional, known to have issues on macOS ARM)
    if include_crest:
        try:
            import time

            from openconf.crest_wrapper import run_crest

            start = time.perf_counter()
            mol = run_crest(smiles, n_conformers=max_confs, quick=True, timeout_sec=600)
            elapsed = time.perf_counter() - start
            result.methods["crest"] = MethodResult(
                "crest", mol.GetNumConformers(), elapsed, compute_min_rmsd(mol, ref_mol)
            )
        except Exception:
            result.methods["crest"] = MethodResult("crest", success=False)

    return result


def write_results_csv(results: list[MoleculeResult], output_path: Path) -> None:
    """Write benchmark results to CSV."""
    methods = ["openconf", "etkdg", "crest"]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["mol_id", "smiles", "n_heavy_atoms", "n_rotatable"]
        for m in methods:
            header.extend([f"{m}_n_confs", f"{m}_time_sec", f"{m}_min_rmsd", f"{m}_success"])
        writer.writerow(header)

        for r in results:
            row = [r.mol_id, r.smiles, r.n_heavy_atoms, r.n_rotatable]
            for m in methods:
                if m in r.methods:
                    mr = r.methods[m]
                    row.extend(
                        [
                            mr.n_conformers,
                            f"{mr.time_sec:.3f}",
                            f"{mr.min_rmsd:.4f}" if mr.min_rmsd < float("inf") else "inf",
                            mr.success,
                        ]
                    )
                else:
                    row.extend(["", "", "", ""])
            writer.writerow(row)


def write_latex_table(results: list[MoleculeResult], output_path: Path) -> None:
    """Write LaTeX summary table."""

    def stats(method: str) -> tuple[list[float], list[float]]:
        rmsds = [
            r.methods[method].min_rmsd
            for r in results
            if method in r.methods and r.methods[method].success and r.methods[method].min_rmsd < float("inf")
        ]
        times = [r.methods[method].time_sec for r in results if method in r.methods and r.methods[method].success]
        return rmsds, times

    def bootstrap_ci(vals: list[float], n_boot: int = 1000) -> tuple[float, float]:
        if not vals:
            return 0, 0
        boots = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    methods = ["openconf", "etkdg", "crest"]
    available = [m for m in methods if any(m in r.methods for r in results)]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Conformer Generation Benchmark Results}",
        r"\label{tab:benchmark}",
        r"\begin{tabular}{l" + "c" * len(available) + "}",
        r"\toprule",
        "Metric & " + " & ".join(m.upper() for m in available) + r" \\",
        r"\midrule",
    ]

    for label, fn in [
        ("Mean RMSD (Å)", lambda v: (np.mean(v), bootstrap_ci(v))),
        ("Median RMSD (Å)", lambda v: (np.median(v), bootstrap_ci(v, n_boot=1000))),
    ]:
        parts = []
        for m in available:
            rmsds, _ = stats(m)
            if rmsds:
                val, (lo, hi) = fn(rmsds)
                parts.append(f"${val:.3f}$ (${lo:.3f}$--${hi:.3f}$)")
            else:
                parts.append("--")
        lines.append(f"{label} & " + " & ".join(parts) + r" \\")

    for thresh in [0.5, 1.0, 2.0]:
        parts = []
        for m in available:
            rmsds, _ = stats(m)
            parts.append(f"${pct_below(rmsds, thresh):.1f}$" if rmsds else "--")
        lines.append(f"$<{thresh}$ Å (\\%) & " + " & ".join(parts) + r" \\")

    lines.append(r"\midrule")
    parts = []
    for m in available:
        _, times = stats(m)
        parts.append(f"${np.mean(times):.2f}$" if times else "--")
    lines.append("Mean time (s) & " + " & ".join(parts) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_path.write_text("\n".join(lines))


def print_summary(results: list[MoleculeResult]) -> None:
    """Print summary statistics."""

    def get_vals(method: str, attr: str) -> list[float]:
        return [
            getattr(r.methods[method], attr)
            for r in results
            if method in r.methods and r.methods[method].success and getattr(r.methods[method], attr) < float("inf")
        ]

    methods = [m for m in ["openconf", "etkdg", "crest"] if any(m in r.methods for r in results)]
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    print(f"{'Metric':<25}" + "".join(f" {m.upper():>12}" for m in methods))
    print("-" * 80)

    for label, attr, fmt in [
        ("Mean RMSD (Å)", "min_rmsd", ".3f"),
        ("Median RMSD (Å)", "min_rmsd", ".3f"),
        ("Mean time (s)", "time_sec", ".2f"),
        ("Mean n_conformers", "n_conformers", ".1f"),
    ]:
        row = f"{label:<25}"
        for m in methods:
            vals = get_vals(m, attr)
            if vals:
                v = np.median(vals) if "Median" in label else np.mean(vals)
                row += f" {v:>12{fmt}}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    print("-" * 80)
    for thresh in [0.5, 1.0, 2.0]:
        row = f"{'<' + str(thresh) + 'Å (%)':<25}"
        for m in methods:
            vals = get_vals(m, "min_rmsd")
            row += f" {pct_below(vals, thresh):>12.1f}" if vals else f" {'N/A':>12}"
        print(row)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Conformer generation benchmark")
    parser.add_argument("--dataset", default="iridium")
    parser.add_argument("--n-molecules", type=int, default=None)
    parser.add_argument("--skip-crest", action="store_true")
    parser.add_argument("--max-confs", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "static" / args.dataset
    if not data_dir.exists():
        print(f"Error: Dataset not found: {data_dir}")
        sys.exit(1)

    sdf_files = sorted(data_dir.glob("*.sdf"))
    if args.n_molecules:
        sdf_files = sdf_files[: args.n_molecules]

    print("=" * 80)
    print(f"BENCHMARK: {args.dataset} (n={len(sdf_files)})")
    print("=" * 80)

    results = []
    for i, sdf_path in enumerate(sdf_files, 1):
        mol_id = sdf_path.stem.split("_")[0]
        print(f"[{i:3d}/{len(sdf_files)}] {mol_id}...", end=" ", flush=True)

        result = benchmark_molecule(sdf_path, args.max_confs, include_crest=not args.skip_crest)
        if result:
            results.append(result)
            parts = []
            for m in ["openconf", "etkdg", "crest"]:
                if m in result.methods and result.methods[m].success:
                    parts.append(f"{m[:2].upper()}:{result.methods[m].min_rmsd:.3f}")
            print(" ".join(parts))
        else:
            print("SKIP")

    output_path = (
        Path(args.output) if args.output else Path(__file__).parent / f"{args.dataset}_comprehensive_results.csv"
    )
    write_results_csv(results, output_path)
    write_latex_table(results, output_path.with_suffix(".tex"))
    print(f"\nResults: {output_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
