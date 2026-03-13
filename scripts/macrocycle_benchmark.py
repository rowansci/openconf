#!/usr/bin/env python
"""MPCONF196GEN macrocycle / peptide conformer benchmark.

Downloads the MPCONF196GEN dataset from GitHub and benchmarks conformer
generation methods against CREST iMTD-GC reference ensembles.

The benchmark measures how well each method covers the low-energy
conformational landscape established by CREST + GFN2-xTB, rather than
recovering a single crystal structure (as in the Iridium benchmark).

Primary metrics
---------------
  coverage@X    fraction of top-K CREST reference conformers that have at
                least one generated conformer within X Angstrom RMSD
  best_ref_rmsd min RMSD from any generated conformer to any reference
  n_unique      unique generated conformers after shared 1.0 A pruning

Usage
-----
    pixi run -e bench python scripts/macrocycle_benchmark.py
    pixi run -e bench python scripts/macrocycle_benchmark.py --budgets 50,200 --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolAlign
from rdkit.Geometry import rdGeometry

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_lib import (
    run_etkdg_mmff,
    run_etkdg_raw,
    run_openconf,
    shared_rmsd_prune,
)

# ──────────────────────────────────────────────────────────────────────────────
# GitHub data fetching
# ──────────────────────────────────────────────────────────────────────────────

REPO = "rowansci/MPCONF196GEN-benchmark"
BRANCH = "master"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"


def fetch_text(path: str) -> str:
    url = f"{RAW_BASE}/{path}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode()


# ──────────────────────────────────────────────────────────────────────────────
# XYZ parsing and conversion
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class XYZFrame:
    elements: list[str]
    coords: list[tuple[float, float, float]]
    energy_eh: float  # GFN2-xTB energy in Eh; inf if not present


def parse_multiframe_xyz(text: str) -> list[XYZFrame]:
    """Parse a multi-frame XYZ file (e.g. CREST output)."""
    frames: list[XYZFrame] = []
    lines = text.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n = int(line)
        except ValueError:
            i += 1
            continue
        energy_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        try:
            energy = float(energy_line)
        except ValueError:
            energy = float("inf")
        elements, coords = [], []
        for j in range(n):
            parts = lines[i + 2 + j].split()
            elements.append(parts[0])
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
        frames.append(XYZFrame(elements, coords, energy))
        i += 2 + n
    return frames


def xyz_frame_to_mol(frame: XYZFrame, charge: int = 0) -> Chem.Mol | None:
    """Convert an XYZ frame to an RDKit mol with bonds determined by distance."""
    n = len(frame.elements)
    edit = Chem.RWMol()
    for sym in frame.elements:
        edit.AddAtom(Chem.Atom(sym))
    conf = Chem.Conformer(n)
    for i, (x, y, z) in enumerate(frame.coords):
        conf.SetAtomPosition(i, rdGeometry.Point3D(x, y, z))
    edit.AddConformer(conf, assignId=True)
    mol = edit.GetMol()
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def best_rmsd_to_reference(
    gen_mol: Chem.Mol,
    gen_conf_ids: list[int],
    ref_mols: list[Chem.Mol],
) -> float:
    """Minimum RMSD from any generated conformer to any reference mol.

    Uses GetBestRMS (symmetry-corrected) with MCS fallback. Both gen_mol and
    ref_mols should be heavy-atom-only mols with one conformer each.
    """
    from rdkit.Chem import rdFMCS

    best = float("inf")
    for cid in gen_conf_ids:
        for ref_mol in ref_mols:
            try:
                r = rdMolAlign.GetBestRMS(gen_mol, ref_mol, prbId=cid, refId=0)
                best = min(best, r)
                continue
            except Exception:
                pass
            # MCS fallback
            try:
                mcs = rdFMCS.FindMCS(
                    [gen_mol, ref_mol],
                    ringMatchesRingOnly=True,
                    completeRingsOnly=True,
                    timeout=2,
                )
                patt = Chem.MolFromSmarts(mcs.smartsString)
                if patt:
                    gm = gen_mol.GetSubstructMatch(patt)
                    rm = ref_mol.GetSubstructMatch(patt)
                    if gm and rm and len(gm) == len(rm):
                        r = float(
                            rdMolAlign.AlignMol(
                                gen_mol,
                                ref_mol,
                                prbCid=cid,
                                refCid=0,
                                atomMap=list(zip(gm, rm, strict=True)),
                            )
                        )
                        best = min(best, r)
            except Exception:
                pass
    return best


def coverage_at_threshold(
    gen_mol: Chem.Mol,
    gen_conf_ids: list[int],
    ref_mols: list[Chem.Mol],
    threshold: float,
) -> float:
    """Fraction of reference conformers covered (nearest generated < threshold A)."""
    if not ref_mols:
        return float("nan")
    covered = 0
    for ref_mol in ref_mols:
        nearest = float("inf")
        for cid in gen_conf_ids:
            try:
                r = rdMolAlign.GetBestRMS(gen_mol, ref_mol, prbId=cid, refId=0)
                nearest = min(nearest, r)
            except Exception:
                pass
        if nearest < threshold:
            covered += 1
    return covered / len(ref_mols)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

# Classify molecules from SMILES: macrocycle if ring size >= 12
_MACROCYCLE_IDS = {"CAMVES_I", "CHPSAR_I", "COHVAW_I", "GS464992_I", "GS557577_I", "POXTRD_I", "SANGLI_I", "YIVNOG_I"}
_LINEAR_PEPTIDE_IDS = {"FGG55", "GFA01", "GGF01", "WG01", "WGG01"}


@dataclass
class MolEntry:
    mol_id: str
    rotatable_bonds: int
    smiles: str
    is_macrocycle: bool
    formal_charge: int
    ref_mols: list[Chem.Mol] = field(default_factory=list)  # top-K CREST reference mols (heavy atoms, 1 conf each)


def load_dataset(max_ref_confs: int = 20) -> list[MolEntry]:
    """Download SMILES + CREST ensembles from GitHub, return benchmark entries."""
    print("Fetching SMILES.txt...")
    smiles_text = fetch_text("SMILES.txt")

    entries: list[MolEntry] = []
    for line in smiles_text.strip().splitlines()[1:]:  # skip header
        parts = line.split("\t")
        mol_id, rb, smiles = parts[0], int(parts[1]), parts[2]

        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            print(f"  {mol_id}: invalid SMILES, skipping")
            continue
        charge = Chem.GetFormalCharge(rdmol)

        print(f"Fetching CREST ensemble for {mol_id}...")
        try:
            xyz_text = fetch_text(f"molecules/{mol_id}/crest_conformers.xyz")
        except Exception as exc:
            print(f"  {mol_id}: download failed ({exc}), skipping")
            continue

        frames = parse_multiframe_xyz(xyz_text)
        if not frames:
            print(f"  {mol_id}: no frames parsed, skipping")
            continue

        # Sort by GFN2-xTB energy, take lowest-energy subset
        frames.sort(key=lambda f: f.energy_eh)
        top_frames = frames[:max_ref_confs]

        # Convert to RDKit mols (heavy atoms only, one conformer each)
        ref_mols: list[Chem.Mol] = []
        for frame in top_frames:
            mol = xyz_frame_to_mol(frame, charge=charge)
            if mol is None:
                continue
            mol_h = Chem.RemoveHs(mol)
            try:
                Chem.SanitizeMol(mol_h)
                ref_mols.append(mol_h)
            except Exception:
                pass

        if not ref_mols:
            print(f"  {mol_id}: no reference mols built, skipping")
            continue

        print(f"  {mol_id}: {len(frames)} CREST conformers, using top {len(ref_mols)} as reference")
        entries.append(
            MolEntry(
                mol_id=mol_id,
                rotatable_bonds=rb,
                smiles=smiles,
                is_macrocycle=mol_id in _MACROCYCLE_IDS,
                formal_charge=charge,
                ref_mols=ref_mols,
            )
        )

    return entries


# ──────────────────────────────────────────────────────────────────────────────
# Run + evaluate
# ──────────────────────────────────────────────────────────────────────────────


METHODS = {
    "etkdg_raw": run_etkdg_raw,
    "etkdg_mmff": run_etkdg_mmff,
    "openconf": run_openconf,
}

COVERAGE_THRESHOLDS = [1.0, 1.5, 2.0, 3.0]
PRUNE_THRESHOLD = 1.0


@dataclass
class RunRecord:
    mol_id: str
    method: str
    budget: int
    seed: int
    runtime_s: float
    n_raw: int
    n_unique: int
    best_ref_rmsd: float
    coverage_10: float
    coverage_15: float
    coverage_20: float
    coverage_30: float
    failure: str | None = None


def run_one(entry: MolEntry, method: str, budget: int, seed: int) -> RunRecord:
    fn = METHODS[method]
    raw = fn(entry.smiles, budget, seed)

    if raw.failure or raw.mol is None or not raw.conf_ids:
        return RunRecord(
            mol_id=entry.mol_id,
            method=method,
            budget=budget,
            seed=seed,
            runtime_s=raw.runtime_s,
            n_raw=raw.n_raw,
            n_unique=0,
            best_ref_rmsd=float("inf"),
            coverage_10=0.0,
            coverage_15=0.0,
            coverage_20=0.0,
            coverage_30=0.0,
            failure=raw.failure or "no_conformers",
        )

    pruned_ids = shared_rmsd_prune(raw.mol, raw.conf_ids, raw.energies, PRUNE_THRESHOLD)
    gen_h = Chem.RemoveHs(raw.mol)

    best_rmsd = best_rmsd_to_reference(gen_h, pruned_ids, entry.ref_mols)
    coverages = [coverage_at_threshold(gen_h, pruned_ids, entry.ref_mols, t) for t in COVERAGE_THRESHOLDS]

    return RunRecord(
        mol_id=entry.mol_id,
        method=method,
        budget=budget,
        seed=seed,
        runtime_s=raw.runtime_s,
        n_raw=raw.n_raw,
        n_unique=len(pruned_ids),
        best_ref_rmsd=best_rmsd,
        coverage_10=coverages[0],
        coverage_15=coverages[1],
        coverage_20=coverages[2],
        coverage_30=coverages[3],
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
    "n_unique",
    "best_ref_rmsd",
    "coverage_10",
    "coverage_15",
    "coverage_20",
    "coverage_30",
    "failure",
]


def write_csv(records: list[RunRecord], path: Path) -> None:
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
                    "n_unique": r.n_unique,
                    "best_ref_rmsd": f"{r.best_ref_rmsd:.4f}" if r.best_ref_rmsd < float("inf") else "inf",
                    "coverage_10": f"{r.coverage_10:.3f}",
                    "coverage_15": f"{r.coverage_15:.3f}",
                    "coverage_20": f"{r.coverage_20:.3f}",
                    "coverage_30": f"{r.coverage_30:.3f}",
                    "failure": r.failure or "",
                }
            )


def agg_by_mol(records: list[RunRecord], method: str, budget: int) -> dict[str, list]:
    recs = [r for r in records if r.method == method and r.budget == budget and not r.failure]
    mols = sorted({r.mol_id for r in recs})
    out: dict[str, list] = {k: [] for k in ["rmsd", "cov10", "cov15", "cov20", "cov30", "time", "uniq"]}
    for mid in mols:
        mr = [r for r in recs if r.mol_id == mid]
        out["rmsd"].append(min(r.best_ref_rmsd for r in mr))
        for k, attr in [
            ("cov10", "coverage_10"),
            ("cov15", "coverage_15"),
            ("cov20", "coverage_20"),
            ("cov30", "coverage_30"),
        ]:
            out[k].append(float(np.mean([getattr(r, attr) for r in mr])))
        out["time"].append(float(np.mean([r.runtime_s for r in mr])))
        out["uniq"].append(float(np.mean([r.n_unique for r in mr])))
    return out


def print_summary(records: list[RunRecord], methods: list[str], budgets: list[int], entries: list[MolEntry]) -> None:
    mol_map = {e.mol_id: e for e in entries}

    print("\n" + "=" * 100)
    print("MPCONF196GEN BENCHMARK RESULTS")
    print("=" * 100)

    for budget in budgets:
        n_mols = len({r.mol_id for r in records if r.budget == budget and not r.failure})
        print(f"\n--- Budget N={budget}  ({n_mols} molecules) ---")
        print(
            f"{'Method':<22} {'N mol':<7} {'Med RMSD':<10} {'Cov@1.0':<9} {'Cov@1.5':<9} "
            f"{'Cov@2.0':<9} {'Cov@3.0':<9} {'Med t':<8} {'Uniq':<6}"
        )
        print("-" * 100)
        for method in methods:
            agg = agg_by_mol(records, method, budget)
            if not agg["rmsd"]:
                print(f"{method:<22} 0")
                continue
            print(
                f"{method:<22} {len(agg['rmsd']):<7} "
                f"{np.median(agg['rmsd']):<10.3f} "
                f"{100 * np.mean(agg['cov10']):<9.1f} "
                f"{100 * np.mean(agg['cov15']):<9.1f} "
                f"{100 * np.mean(agg['cov20']):<9.1f} "
                f"{100 * np.mean(agg['cov30']):<9.1f} "
                f"{np.median(agg['time']):<8.2f} "
                f"{np.median(agg['uniq']):<6.0f}"
            )

    # Stratified: macrocycles vs linear peptides
    for budget in budgets:
        print(f"\n--- Stratified @ N={budget} ---")
        for label, mol_ids in [
            ("Macrocycles", {e.mol_id for e in entries if e.is_macrocycle}),
            ("Linear peptides", {e.mol_id for e in entries if not e.is_macrocycle}),
        ]:
            sub = [r for r in records if r.mol_id in mol_ids and r.budget == budget]
            if not sub:
                continue
            print(f"\n  {label} (n={len(mol_ids)})")
            print(f"  {'Method':<22} {'Med RMSD':<10} {'Cov@1.5':<9} {'Cov@2.0':<9} {'Cov@3.0':<9}")
            for method in methods:
                agg = agg_by_mol(sub, method, budget)
                if not agg["rmsd"]:
                    continue
                print(
                    f"  {method:<22} "
                    f"{np.median(agg['rmsd']):<10.3f} "
                    f"{100 * np.mean(agg['cov15']):<9.1f} "
                    f"{100 * np.mean(agg['cov20']):<9.1f} "
                    f"{100 * np.mean(agg['cov30']):<9.1f}"
                )

    # Per-molecule detail at largest budget
    largest = max(budgets)
    print(f"\n--- Per-molecule detail @ N={largest} ---")
    print(f"{'mol_id':<16} {'rb':<5} {'macro':<7} " + "  ".join(f"{m[:10]:<12}" for m in methods))
    print("-" * 90)
    mol_ids = sorted({r.mol_id for r in records})
    for mid in mol_ids:
        entry = mol_map.get(mid)
        rb = entry.rotatable_bonds if entry else "?"
        is_mac = "Y" if (entry and entry.is_macrocycle) else "N"
        parts = []
        for method in methods:
            mr = [
                r for r in records if r.mol_id == mid and r.method == method and r.budget == largest and not r.failure
            ]
            if not mr:
                parts.append(f"{'FAIL':<12}")
            else:
                best_rmsd = min(r.best_ref_rmsd for r in mr)
                cov20 = np.mean([r.coverage_20 for r in mr])
                parts.append(f"{best_rmsd:.3f}/{100 * cov20:.0f}%".ljust(12))
        print(f"{mid:<16} {rb:<5} {is_mac:<7} {'  '.join(parts)}")
    print("(format: best_rmsd / cov@2.0Å%)")
    print("=" * 100)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budgets", default="50,200", help="Comma-separated conformer budgets (default: 50,200)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds 1..N (default: 3)")
    parser.add_argument(
        "--methods", default=",".join(METHODS), help=f"Comma-separated methods (default: {','.join(METHODS)})"
    )
    parser.add_argument(
        "--max-ref-confs", type=int, default=20, help="Max CREST reference conformers per molecule (default: 20)"
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    seeds = list(range(1, args.seeds + 1))
    methods = [m.strip() for m in args.methods.split(",")]

    entries = load_dataset(max_ref_confs=args.max_ref_confs)
    if not entries:
        print("No molecules loaded — exiting.")
        sys.exit(1)

    total = len(entries) * len(methods) * len(budgets) * len(seeds)
    print(f"\nMolecules: {len(entries)}  Methods: {methods}  Budgets: {budgets}  Seeds: {seeds}")
    print(f"Total runs: {total}\n" + "=" * 80)

    records: list[RunRecord] = []
    run_idx = 0
    for entry in entries:
        for method in methods:
            for budget in budgets:
                for seed in seeds:
                    run_idx += 1
                    label = f"[{run_idx:3d}/{total}] {entry.mol_id:<16} {method:<22} N={budget:<4} s={seed}"
                    print(label, end=" ", flush=True)

                    rec = run_one(entry, method, budget, seed)
                    records.append(rec)

                    if rec.failure:
                        print(f"FAIL({rec.failure})")
                    else:
                        covs = f"cov@1.5={rec.coverage_15 * 100:.0f}% cov@2.0={rec.coverage_20 * 100:.0f}%"
                        print(f"rmsd={rec.best_ref_rmsd:.3f} {covs} uniq={rec.n_unique} t={rec.runtime_s:.2f}s")

    out_path = Path(args.output) if args.output else Path(__file__).parent / "macrocycle_benchmark_results.csv"
    write_csv(records, out_path)
    print(f"\nResults: {out_path}")

    print_summary(records, methods, budgets, entries)


if __name__ == "__main__":
    main()
