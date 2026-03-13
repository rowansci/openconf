#!/usr/bin/env python
"""Statistical analysis and visualization of benchmark results.

Usage:
    pixi run -e bench python scripts/benchmark_analysis.py results.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from scipy import stats
except ImportError:
    print("Error: requires matplotlib and scipy. Install with: pixi install -e bench")
    sys.exit(1)


def load_results(csv_path: Path) -> dict[str, Any]:
    """Load benchmark results from CSV."""
    results: dict[str, Any] = {"mol_id": [], "smiles": [], "n_heavy_atoms": [], "n_rotatable": []}
    for m in ["openconf", "etkdg", "crest"]:
        results[f"{m}_min_rmsd"] = []
        results[f"{m}_time_sec"] = []
        results[f"{m}_n_confs"] = []

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            results["mol_id"].append(row["mol_id"])
            results["smiles"].append(row["smiles"])
            results["n_heavy_atoms"].append(int(row["n_heavy_atoms"]))
            results["n_rotatable"].append(int(row["n_rotatable"]))

            for m in ["openconf", "etkdg", "crest"]:
                for key, default, conv in [
                    (f"{m}_min_rmsd", float("inf"), lambda x: float(x) if x and x != "inf" else float("inf")),
                    (f"{m}_time_sec", 0.0, lambda x: float(x) if x else 0.0),
                    (f"{m}_n_confs", 0, lambda x: int(x) if x else 0),
                ]:
                    try:
                        results[key].append(conv(row.get(key, "")))
                    except (ValueError, KeyError):
                        results[key].append(default)

    for key in results:
        if key not in ["mol_id", "smiles"]:
            results[key] = np.array(results[key])
    return results


def bootstrap_ci(data: np.ndarray, func=np.mean, n_boot: int = 10000) -> tuple[float, float, float]:
    """Compute bootstrap 95% CI. Returns (point, lower, upper)."""
    data = data[~np.isinf(data)]
    if len(data) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(42)
    boots = [func(rng.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    return func(data), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Paired Wilcoxon test. Returns (statistic, p_value)."""
    valid = (~np.isinf(x)) & (~np.isinf(y))
    if valid.sum() < 10:
        return float("nan"), float("nan")
    try:
        return stats.wilcoxon(y[valid] - x[valid], alternative="greater")
    except Exception:
        return float("nan"), float("nan")


def stratified_analysis(results: dict, n_bins: int = 4) -> dict:
    """Analyze performance by molecular flexibility."""
    n_rot = results["n_rotatable"]
    edges = np.unique(np.percentile(n_rot, np.linspace(0, 100, n_bins + 1)))

    strat = {"bins": [], "n_molecules": []}
    for m in ["openconf", "etkdg", "crest"]:
        strat[f"{m}_mean_rmsd"] = []
        strat[f"{m}_ci_lower"] = []
        strat[f"{m}_ci_upper"] = []

    for i in range(len(edges) - 1):
        mask = (n_rot >= edges[i]) & (n_rot <= edges[i + 1] if i == len(edges) - 2 else n_rot < edges[i + 1])
        strat["bins"].append(f"{int(edges[i])}-{int(edges[i + 1])}")
        strat["n_molecules"].append(mask.sum())

        for m in ["openconf", "etkdg", "crest"]:
            rmsds = results[f"{m}_min_rmsd"][mask]
            rmsds = rmsds[~np.isinf(rmsds)]
            if len(rmsds) > 0:
                mean, lo, hi = bootstrap_ci(rmsds)
                strat[f"{m}_mean_rmsd"].append(mean)
                strat[f"{m}_ci_lower"].append(lo)
                strat[f"{m}_ci_upper"].append(hi)
            else:
                strat[f"{m}_mean_rmsd"].append(float("nan"))
                strat[f"{m}_ci_lower"].append(float("nan"))
                strat[f"{m}_ci_upper"].append(float("nan"))
    return strat


def plot_pareto(results: dict, output: Path):
    """Plot Pareto frontier."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"openconf": "#2E86AB", "etkdg": "#A23B72", "crest": "#F18F01"}
    markers = {"openconf": "o", "etkdg": "s", "crest": "^"}

    for m in ["openconf", "etkdg", "crest"]:
        rmsds, times = results[f"{m}_min_rmsd"], results[f"{m}_time_sec"]
        valid = (~np.isinf(rmsds)) & (times > 0)
        if valid.sum() == 0:
            continue

        mean_rmsd, mean_time = np.mean(rmsds[valid]), np.mean(times[valid])
        ax.scatter(
            mean_time,
            mean_rmsd,
            c=colors[m],
            marker=markers[m],
            s=200,
            label=m.upper() if m != "openconf" else "OpenConf",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )

        _, rmsd_lo, rmsd_hi = bootstrap_ci(rmsds[valid])
        _, time_lo, time_hi = bootstrap_ci(times[valid])
        ax.errorbar(
            mean_time,
            mean_rmsd,
            xerr=[[mean_time - time_lo], [time_hi - mean_time]],
            yerr=[[mean_rmsd - rmsd_lo], [rmsd_hi - mean_rmsd]],
            c=colors[m],
            capsize=5,
            capthick=2,
            linewidth=2,
            zorder=4,
        )

    ax.set_xlabel("Mean Time (s)", fontsize=12)
    ax.set_ylabel("Mean RMSD (Å)", fontsize=12)
    ax.set_title("Pareto Frontier: Accuracy vs Speed", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_rmsd_dist(results: dict, output: Path):
    """Plot RMSD distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"openconf": "#2E86AB", "etkdg": "#A23B72", "crest": "#F18F01"}
    bins = list(np.linspace(0, 3, 31))

    for m in ["openconf", "etkdg", "crest"]:
        rmsds = results[f"{m}_min_rmsd"]
        valid = ~np.isinf(rmsds)
        if valid.sum() > 0:
            ax.hist(
                rmsds[valid],
                bins=bins,
                alpha=0.5,
                label=m.upper() if m != "openconf" else "OpenConf",
                color=colors[m],
                edgecolor="black",
                linewidth=0.5,
            )

    ax.axvline(0.5, color="green", linestyle="--", linewidth=2, label="0.5Å")
    ax.axvline(1.0, color="orange", linestyle="--", linewidth=2, label="1.0Å")
    ax.set_xlabel("Min RMSD (Å)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("RMSD Distribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_stratified(strat: dict, output: Path):
    """Plot performance by flexibility."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"openconf": "#2E86AB", "etkdg": "#A23B72", "crest": "#F18F01"}
    x = np.arange(len(strat["bins"]))
    width = 0.25

    for i, m in enumerate(["openconf", "etkdg", "crest"]):
        means = strat[f"{m}_mean_rmsd"]
        if all(np.isnan(v) for v in means):
            continue
        errors = [
            [max(0, m - l) for m, l in zip(means, strat[f"{m}_ci_lower"])],
            [max(0, h - m) for m, h in zip(means, strat[f"{m}_ci_upper"])],
        ]
        ax.bar(
            x + i * width,
            means,
            width,
            label=m.upper() if m != "openconf" else "OpenConf",
            color=colors[m],
            yerr=errors,
            capsize=4,
        )

    ax.set_xlabel("Rotatable Bonds", fontsize=12)
    ax.set_ylabel("Mean RMSD (Å)", fontsize=12)
    ax.set_title("Performance by Flexibility", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(strat["bins"])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


def print_summary(results: dict):
    """Print statistical summary."""
    methods = ["openconf", "etkdg", "crest"]

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'OpenConf':<20} {'ETKDG':<20} {'CREST':<20}")
    print("-" * 70)

    for label, func in [("Mean RMSD", np.mean), ("Median RMSD", np.median)]:
        row = f"{label:<20}"
        for m in methods:
            rmsds = results[f"{m}_min_rmsd"]
            rmsds = rmsds[~np.isinf(rmsds)]
            if len(rmsds) > 0:
                val, lo, hi = bootstrap_ci(rmsds, func)
                row += f" {val:.3f} [{lo:.3f}-{hi:.3f}]"
            else:
                row += f" {'N/A':<20}"
        print(row)

    print("-" * 70)
    print("\nWilcoxon tests (OpenConf vs others, one-sided):")
    oc = results["openconf_min_rmsd"]
    for other in ["etkdg", "crest"]:
        stat, p = paired_wilcoxon(oc, results[f"{other}_min_rmsd"])
        if not np.isnan(p):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  vs {other.upper()}: p={p:.4f} {sig}")

    print("\nHead-to-head (0.05Å threshold):")
    for other in ["etkdg", "crest"]:
        other_r = results[f"{other}_min_rmsd"]
        valid = (~np.isinf(oc)) & (~np.isinf(other_r))
        if valid.sum() > 0:
            diff = other_r[valid] - oc[valid]
            wins, ties, losses = (diff > 0.05).sum(), (np.abs(diff) <= 0.05).sum(), (diff < -0.05).sum()
            print(
                f"  vs {other.upper()}: {wins} wins, {ties} ties, {losses} losses ({100 * wins / valid.sum():.1f}% win rate)"
            )

    print("=" * 70)


def generate_latex(results: dict) -> str:
    """Generate LaTeX table."""
    methods = ["openconf", "etkdg", "crest"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Benchmark Results}",
        r"\label{tab:benchmark}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        "Metric & OpenConf & ETKDG & CREST" + r" \\",
        r"\midrule",
    ]

    for label, func in [("Mean RMSD (Å)", np.mean), ("Median RMSD (Å)", np.median)]:
        parts = []
        for m in methods:
            rmsds = results[f"{m}_min_rmsd"][~np.isinf(results[f"{m}_min_rmsd"])]
            if len(rmsds) > 0:
                val, lo, hi = bootstrap_ci(rmsds, func)
                parts.append(f"${val:.3f}$ (${lo:.3f}$--${hi:.3f}$)")
            else:
                parts.append("--")
        lines.append(f"{label} & " + " & ".join(parts) + r" \\")

    for thresh in [0.5, 1.0, 2.0]:
        parts = []
        for m in methods:
            rmsds = results[f"{m}_min_rmsd"]
            valid = ~np.isinf(rmsds)
            pct = 100 * (rmsds[valid] < thresh).sum() / valid.sum() if valid.sum() > 0 else 0
            parts.append(f"${pct:.1f}$" if valid.sum() > 0 else "--")
        lines.append(f"$<{thresh}$ Å (\\%) & " + " & ".join(parts) + r" \\")

    lines.append(r"\midrule")
    parts = []
    for m in methods:
        times = results[f"{m}_time_sec"]
        valid = times > 0
        parts.append(f"${np.mean(times[valid]):.2f}$" if valid.sum() > 0 else "--")
    lines.append("Mean time (s) & " + " & ".join(parts) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if not args.csv_file.exists():
        print(f"Error: {args.csv_file} not found")
        sys.exit(1)

    results = load_results(args.csv_file)
    print(f"Loaded {len(results['mol_id'])} molecules from {args.csv_file}")

    print_summary(results)

    latex = generate_latex(results)
    latex_path = args.csv_file.with_suffix(".tex")
    latex_path.write_text(latex)
    print(f"\nLaTeX: {latex_path}")

    if not args.no_plots:
        output_dir = args.output_dir or args.csv_file.parent / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_pareto(results, output_dir / "pareto_frontier.png")
        plot_rmsd_dist(results, output_dir / "rmsd_distribution.png")
        plot_stratified(stratified_analysis(results), output_dir / "stratified_performance.png")


if __name__ == "__main__":
    main()
