#!/usr/bin/env python
"""Generate blog-post figures from the Iridium fair benchmark results.

Usage:
    pixi run -e bench python scripts/make_blog_figures.py
    pixi run -e bench python scripts/make_blog_figures.py --output docs/figures
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_lib import load_iridium

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rc("font", size=11, family="serif")
plt.rc("axes", titlesize=12, labelsize=12)
plt.rc(["xtick", "ytick"], labelsize=11)
plt.rc("legend", fontsize=11)
plt.rc("figure", titlesize=14)

# ── Palette ────────────────────────────────────────────────────────────────────
COLORS = {
    "etkdg_raw": "#9ecae1",  # light blue
    "etkdg_mmff": "#4292c6",  # mid blue
    "openconf": "#d62728",  # red
}
LABELS = {
    "etkdg_raw": "ETKDGv3 (raw)",
    "etkdg_mmff": "ETKDGv3 + MMFF",
    "openconf": "OpenConf",
}
METHOD_ORDER = ["etkdg_raw", "etkdg_mmff", "openconf"]

BUDGETS = [10, 50, 200]

# rb strata used throughout
STRATA = [(0, 3), (4, 6), (7, 9), (10, 99)]
STRATA_LABELS = ["0-3 RB\n(rigid)", "4-6 RB", "7-9 RB\n(flexible)", "10+ RB\n(very flexible)"]


# ── Data loading ───────────────────────────────────────────────────────────────


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["budget"] = int(r["budget"])
        r["seed"] = int(r["seed"])
        r["runtime_s"] = float(r["runtime_s"])
        r["best_rmsd"] = float(r["best_rmsd"]) if r["best_rmsd"] not in ("inf", "") else float("inf")
        r["success_10"] = r["success_10"].strip().lower() == "true"
        r["success_05"] = r["success_05"].strip().lower() == "true"
        r["failure"] = r["failure"].strip()
    return rows


def build_rb_lookup(data_dir: str) -> dict[str, int]:
    """mol_id -> rotatable bond count from Iridium SDF files."""
    records = load_iridium(data_dir)
    return {r.mol_id: r.n_rotatable for r in records}


def mol_rb(mol_id: str, rb_lookup: dict[str, int]) -> int:
    return rb_lookup.get(mol_id, -1)


def per_mol_best_over_seeds(rows, method, budget, key, mol_ids=None):
    """Per molecule, take the best (max) value over seeds.

    For success flags (0/1) this answers: did any seed succeed?
    For RMSD this answers: what was the best RMSD any seed achieved?
    Note: caller should use min() for RMSD and max() for success flags.
    This function uses max(), appropriate for success_10 / success_05.
    """
    recs = [r for r in rows if r["method"] == method and r["budget"] == budget and not r["failure"]]
    if mol_ids is not None:
        recs = [r for r in recs if r["mol_id"] in mol_ids]
    mols = sorted({r["mol_id"] for r in recs})
    vals = []
    for mid in mols:
        mr = [r[key] for r in recs if r["mol_id"] == mid]
        if mr:
            vals.append(max(float(v) for v in mr))
    return vals


def per_mol_mean(rows, method, budget, key, mol_ids=None):
    """Per molecule, average over seeds."""
    recs = [r for r in rows if r["method"] == method and r["budget"] == budget and not r["failure"]]
    if mol_ids is not None:
        recs = [r for r in recs if r["mol_id"] in mol_ids]
    mols = sorted({r["mol_id"] for r in recs})
    vals = []
    for mid in mols:
        mr = [r[key] for r in recs if r["mol_id"] == mid]
        if mr:
            vals.append(float(np.mean(mr)))
    return vals


def recall_rate(rows, method, budget, mol_ids=None):
    """Mean success@1.0A over molecules (best-over-seeds per mol)."""
    vals = per_mol_best_over_seeds(rows, method, budget, "success_10", mol_ids)
    return float(np.mean(vals)) * 100 if vals else float("nan")


def median_runtime(rows, method, budget, mol_ids=None):
    vals = per_mol_mean(rows, method, budget, "runtime_s", mol_ids)
    return float(np.median(vals)) if vals else float("nan")


def stratum_mol_ids(rb_lookup: dict[str, int], rb_lo: int, rb_hi: int) -> set[str]:
    return {mid for mid, rb in rb_lookup.items() if rb_lo <= rb <= rb_hi}


# ── Figure 1: Stratified recall bar chart (THE headline figure) ────────────────


def fig_stratified_recall(rows: list[dict], rb_lookup: dict[str, int], out: Path, budget: int = 50) -> None:
    """Grouped bar chart: success@1.0A by rb stratum and method."""
    x = np.arange(len(STRATA))
    width = 0.22
    offsets = [-width, 0, width]

    # Build per-stratum data; embed molecule count in tick labels
    counts = [len(stratum_mol_ids(rb_lookup, lo, hi)) for lo, hi in STRATA]
    tick_labels = [f"{lbl}\n(n={n})" for lbl, n in zip(STRATA_LABELS, counts)]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, method in enumerate(METHOD_ORDER):
        recalls = [recall_rate(rows, method, budget, stratum_mol_ids(rb_lookup, lo, hi)) for lo, hi in STRATA]
        bars = ax.bar(
            x + offsets[i],
            recalls,
            width,
            label=LABELS[method],
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, recalls):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 1.5,
                    f"{val:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=COLORS[method],
                )

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("Success@1.0 A (%)")
    ax.set_title(
        f"Conformer recovery by molecular flexibility — N={budget} conformers\n"
        f"(fraction of molecules with best RMSD < 1.0 A of crystal structure)"
    )
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out / f"stratified_recall_N{budget}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"stratified_recall_N{budget}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved stratified_recall_N{budget}")


# ── Figure 2: RMSD CDF for flexible molecules only (rb 7-9) ───────────────────


def fig_rmsd_cdf_flexible(rows: list[dict], rb_lookup: dict[str, int], out: Path, budget: int = 50) -> None:
    """RMSD CDF restricted to the rb 7-9 stratum, where methods differ most."""
    mids_flexible = stratum_mol_ids(rb_lookup, 7, 9)
    mids_rigid = stratum_mol_ids(rb_lookup, 0, 3)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, mids, title_suffix in [
        (axes[0], mids_flexible, f"Flexible molecules (7-9 RB, n={len(mids_flexible)})"),
        (axes[1], mids_rigid, f"Rigid molecules (0-3 RB, n={len(mids_rigid)})"),
    ]:
        for method in METHOD_ORDER:
            recs = [
                r
                for r in rows
                if r["method"] == method and r["budget"] == budget and not r["failure"] and r["mol_id"] in mids
            ]
            mol_ids_here = sorted({r["mol_id"] for r in recs})
            mol_bests = []
            for mid in mol_ids_here:
                mr = [r["best_rmsd"] for r in recs if r["mol_id"] == mid]
                if mr:
                    mol_bests.append(min(mr))
            finite = [v for v in mol_bests if v < float("inf")]
            xs = np.sort(finite)
            ys = np.arange(1, len(xs) + 1) / len(xs) * 100
            ax.plot(xs, ys, linewidth=2, color=COLORS[method], label=LABELS[method])

        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="1.0 A threshold")
        ax.set_xlabel("Best RMSD to crystal conformer (A)")
        ax.set_ylabel("Cumulative fraction of molecules (%)")
        ax.set_title(title_suffix, fontsize=11)
        ax.set_xlim(0, 3.5)
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="both", linestyle="--", alpha=0.4)

    fig.suptitle(f"RMSD CDF at N={budget}: where methods diverge vs. where they agree", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"rmsd_cdf_stratified_N{budget}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"rmsd_cdf_stratified_N{budget}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved rmsd_cdf_stratified_N{budget}")


# ── Figure 3: Runtime vs budget (log-log) ─────────────────────────────────────


def fig_runtime_vs_budget(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))

    for method in METHOD_ORDER:
        ys = [median_runtime(rows, method, b) for b in BUDGETS]
        ax.plot(
            BUDGETS,
            ys,
            marker="o",
            markersize=7,
            linewidth=2,
            color=COLORS[method],
            label=LABELS[method],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(BUDGETS)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Conformer budget (N)")
    ax.set_ylabel("Median wall time per molecule (s)")
    ax.set_title("Runtime scaling with conformer budget")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out / "runtime_vs_budget.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "runtime_vs_budget.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved runtime_vs_budget")


# ── Figure 4: Quality-speed Pareto frontier ────────────────────────────────────


def fig_pareto(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for method in METHOD_ORDER:
        xs = [median_runtime(rows, method, b) for b in BUDGETS]
        ys = [recall_rate(rows, method, b) for b in BUDGETS]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=8,
            linewidth=1.5,
            color=COLORS[method],
            label=LABELS[method],
        )
        for b, x, y in zip(BUDGETS, xs, ys):
            ax.annotate(
                f"N={b}",
                (x, y),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=8.5,
                color=COLORS[method],
            )

    ax.set_xlabel("Median wall time per molecule (s)")
    ax.set_ylabel("Recall@1.0 A (%)")
    ax.set_title("Quality-speed trade-off\n(upper-left is better)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out / "pareto_quality_speed.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "pareto_quality_speed.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved pareto_quality_speed")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        default=str(Path(__file__).parent / "fair_benchmark_full_results.csv"),
        help="Path to Iridium benchmark CSV",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).parent.parent / "static" / "iridium"),
        help="Path to Iridium SDF files (for rb counts)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "blog_figures"),
        help="Output directory for figures",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input}...")
    rows = load_results(Path(args.input))
    print(f"  {len(rows)} rows, {len({r['mol_id'] for r in rows})} molecules")

    print(f"Loading rb counts from {args.data_dir}...")
    rb_lookup = build_rb_lookup(args.data_dir)
    print(f"  {len(rb_lookup)} molecules with rb counts\n")

    print("Generating figures:")
    fig_stratified_recall(rows, rb_lookup, out, budget=50)
    fig_stratified_recall(rows, rb_lookup, out, budget=200)
    fig_rmsd_cdf_flexible(rows, rb_lookup, out, budget=50)
    fig_runtime_vs_budget(rows, out)
    fig_pareto(rows, out)

    print(f"\nAll figures saved to: {out}/")


if __name__ == "__main__":
    main()
