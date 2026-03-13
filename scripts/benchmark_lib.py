"""Shared utilities for the fair conformer benchmark.

All postprocessing here is method-agnostic: the same canonical MMFF minimization
and RMSD pruning pipeline is applied to every method's raw output so comparisons
are apples-to-apples.

Design rules:
- No imports from openconf (the runner imports openconf; this lib stays independent)
- All functions are pure / side-effect-free (no file I/O)
- Expensive operations (pairwise RMSD) are capped to avoid quadratic blowup
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolAlign, rdMolTransforms

# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    """Raw output of a single (method, molecule, budget, seed) run."""

    mol: Chem.Mol | None  # molecule with embedded conformers
    conf_ids: list[int]  # conformer IDs (in output order)
    energies: list[float]  # MMFF energies parallel to conf_ids (inf if unavailable)
    runtime_s: float
    n_raw: int  # conformers before any shared postprocessing
    failure: str | None = None


@dataclass
class MolRunRecord:
    """Per-(molecule, method, budget, seed) result after shared postprocessing."""

    mol_id: str
    method: str
    budget: int
    seed: int
    runtime_s: float
    # conformer counts
    n_raw: int
    n_shared_pruned: int
    # Track A: experimental conformer recovery
    best_rmsd: float
    best_tfd: float | None
    # Track B: energy (gap added in post-processing after pooling across methods)
    lowest_energy_kcal: float
    energy_gap_kcal: float = field(default=float("inf"))
    # Track C: diversity
    median_pairwise_rmsd: float | None = None
    failure: str | None = None

    @property
    def success_05(self) -> bool:
        return self.best_rmsd < 0.5

    @property
    def success_10(self) -> bool:
        return self.best_rmsd < 1.0

    @property
    def success_15(self) -> bool:
        return self.best_rmsd < 1.5

    @property
    def success_20(self) -> bool:
        return self.best_rmsd < 2.0


@dataclass
class MolRecord:
    """Molecule from the Iridium dataset with descriptor metadata."""

    mol_id: str
    ref_mol: Chem.Mol
    smiles: str
    n_heavy: int
    n_rotatable: int
    mw: float
    is_macrocycle: bool
    largest_ring: int


# ──────────────────────────────────────────────────────────────────────────────
# Shared postprocessing
# ──────────────────────────────────────────────────────────────────────────────


def shared_mmff_minimize(
    mol: Chem.Mol,
    conf_ids: list[int],
    max_iters: int = 500,
    variant: str = "MMFF94s",
) -> list[tuple[int, float]]:
    """Apply canonical MMFF minimization. Returns (conf_id, energy) for successes only."""
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    results: list[tuple[int, float]] = []
    for cid in conf_ids:
        try:
            ff = (
                AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                if props is not None
                else AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            )
            if ff is None:
                continue
            ff.Minimize(maxIts=max_iters)
            results.append((cid, float(ff.CalcEnergy())))
        except Exception:
            pass
    return results


def shared_rmsd_prune(
    mol: Chem.Mol,
    conf_ids: list[int],
    energies: list[float] | None,
    threshold: float = 1.0,
) -> list[int]:
    """Greedy RMSD pruning on heavy atoms (no symmetry correction, fast).

    Conformers are sorted by energy before pruning so the lowest-energy
    representative of each cluster is retained.
    """
    if len(conf_ids) <= 1:
        return list(conf_ids)

    mol_h = Chem.RemoveHs(mol)

    # Sort lowest-energy first
    if energies and len(energies) == len(conf_ids):
        order = sorted(range(len(conf_ids)), key=lambda i: energies[i])
    else:
        order = list(range(len(conf_ids)))
    sorted_ids = [conf_ids[i] for i in order]

    kept: list[int] = [sorted_ids[0]]
    for cid in sorted_ids[1:]:
        too_close = False
        for kept_cid in kept:
            try:
                # CalcRMS: no symmetry, no modification of coords — fast for pruning
                rmsd = rdMolAlign.CalcRMS(mol_h, mol_h, prbId=cid, refId=kept_cid)
                if rmsd < threshold:
                    too_close = True
                    break
            except Exception:
                pass
        if not too_close:
            kept.append(cid)
    return kept


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


def _heavy_mol(mol: Chem.Mol) -> Chem.Mol:
    """Remove Hs and sanitize. Preserves all conformers."""
    mol_h = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol_h)
    return mol_h


def compute_best_rmsd(
    gen_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    conf_ids: list[int] | None = None,
) -> float:
    """Best heavy-atom RMSD (symmetry-corrected via GetBestRMS) vs reference."""
    if gen_mol is None or gen_mol.GetNumConformers() == 0:
        return float("inf")
    if conf_ids is None:
        conf_ids = [c.GetId() for c in gen_mol.GetConformers()]

    gen_h = _heavy_mol(gen_mol)
    ref_h = _heavy_mol(ref_mol)

    min_rmsd = float("inf")
    for cid in conf_ids:
        try:
            rmsd = rdMolAlign.GetBestRMS(gen_h, ref_h, prbId=cid, refId=0)
            min_rmsd = min(min_rmsd, rmsd)
        except Exception:
            # Fall back to MCS-aligned RMSD for tricky cases
            try:
                mcs = rdFMCS.FindMCS(
                    [gen_h, ref_h],
                    ringMatchesRingOnly=True,
                    completeRingsOnly=True,
                    timeout=2,
                )
                patt = Chem.MolFromSmarts(mcs.smartsString)
                if patt:
                    gm = gen_h.GetSubstructMatch(patt)
                    rm = ref_h.GetSubstructMatch(patt)
                    if gm and rm and len(gm) == len(rm):
                        rmsd = float(
                            rdMolAlign.AlignMol(
                                gen_h,
                                ref_h,
                                prbCid=cid,
                                refCid=0,
                                atomMap=list(zip(gm, rm, strict=True)),
                            )
                        )
                        min_rmsd = min(min_rmsd, rmsd)
            except Exception:
                pass
    return min_rmsd


def compute_best_tfd(
    gen_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    conf_ids: list[int] | None = None,
) -> float | None:
    """Best simplified TFD (torsion fingerprint distance) vs reference.

    Uses sqrt(mean(sin²(Δθ/2))) over all rotatable bonds — the unweighted form
    of the standard TFD formula. Returns None if fewer than one rotatable bond
    or if atom mapping fails.
    """
    if gen_mol is None or gen_mol.GetNumConformers() == 0:
        return None
    if conf_ids is None:
        conf_ids = [c.GetId() for c in gen_mol.GetConformers()]

    try:
        gen_h = _heavy_mol(gen_mol)
        ref_h = _heavy_mol(ref_mol)

        # Map ref atom indices to gen atom indices via substructure match.
        # GetSubstructMatch(ref_h) on gen_h: result[ref_atom_idx] = gen_atom_idx
        match = gen_h.GetSubstructMatch(ref_h)
        if len(match) != ref_h.GetNumAtoms():
            return None
        ref_to_gen = list(match)

        # Build 4-atom dihedral definitions from rotatable bonds in ref_h
        torsions: list[tuple[int, int, int, int]] = []
        for bond in ref_h.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
                continue
            b_idx, c_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            b_atom = ref_h.GetAtomWithIdx(b_idx)
            c_atom = ref_h.GetAtomWithIdx(c_idx)
            if b_atom.GetDegree() < 2 or c_atom.GetDegree() < 2:
                continue
            a_idx = next((n.GetIdx() for n in b_atom.GetNeighbors() if n.GetIdx() != c_idx), None)
            d_idx = next((n.GetIdx() for n in c_atom.GetNeighbors() if n.GetIdx() != b_idx), None)
            if a_idx is not None and d_idx is not None:
                torsions.append((a_idx, b_idx, c_idx, d_idx))

        if not torsions:
            return None

        # Reference torsion angles
        ref_conf = ref_h.GetConformer(0)
        ref_angles = [rdMolTransforms.GetDihedralDeg(ref_conf, a, b, c, d) for a, b, c, d in torsions]

        # Best TFD over generated conformers
        min_tfd = float("inf")
        for cid in conf_ids:
            try:
                gen_conf = gen_h.GetConformer(cid)
                sq_sins = []
                for (a_r, b_r, c_r, d_r), ref_a in zip(torsions, ref_angles):
                    a_g, b_g, c_g, d_g = (
                        ref_to_gen[a_r],
                        ref_to_gen[b_r],
                        ref_to_gen[c_r],
                        ref_to_gen[d_r],
                    )
                    gen_a = rdMolTransforms.GetDihedralDeg(gen_conf, a_g, b_g, c_g, d_g)
                    delta = math.radians(ref_a - gen_a)
                    sq_sins.append(math.sin(delta / 2) ** 2)
                tfd = math.sqrt(sum(sq_sins) / len(sq_sins))
                min_tfd = min(min_tfd, tfd)
            except Exception:
                pass

        return min_tfd if min_tfd < float("inf") else None

    except Exception:
        return None


def pairwise_rmsd_stats(
    mol: Chem.Mol,
    conf_ids: list[int],
    max_pairs: int = 30,
) -> float | None:
    """Median pairwise heavy-atom RMSD. Capped at max_pairs conformers to stay O(1)."""
    if len(conf_ids) < 2:
        return None
    mol_h = _heavy_mol(mol)
    # Sample subset if large
    ids = conf_ids[:max_pairs]
    rmsds: list[float] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            try:
                r = rdMolAlign.CalcRMS(mol_h, mol_h, prbId=ids[i], refId=ids[j])
                rmsds.append(r)
            except Exception:
                pass
    return float(np.median(rmsds)) if rmsds else None


# ──────────────────────────────────────────────────────────────────────────────
# ETKDG baselines
# ──────────────────────────────────────────────────────────────────────────────


def _mol_with_hs(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.AddHs(mol)


def run_etkdg_raw(smiles: str, n: int, seed: int) -> RunResult:
    """ETKDGv3 embedding only — no minimization."""
    mol = _mol_with_hs(smiles)
    if mol is None:
        return RunResult(None, [], [], 0.0, 0, failure="invalid_smiles")

    params = AllChem.ETKDGv3()
    params.randomSeed = seed + 1  # seed=0 produces degenerate conformers in RDKit
    params.pruneRmsThresh = -1.0  # defer to shared pruning

    t0 = time.perf_counter()
    AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params)
    runtime = time.perf_counter() - t0

    conf_ids = [c.GetId() for c in mol.GetConformers()]
    energies = [float("inf")] * len(conf_ids)
    return RunResult(mol, conf_ids, energies, runtime, len(conf_ids))


def run_etkdg_mmff(smiles: str, n: int, seed: int) -> RunResult:
    """ETKDGv3 + shared MMFF minimization."""
    mol = _mol_with_hs(smiles)
    if mol is None:
        return RunResult(None, [], [], 0.0, 0, failure="invalid_smiles")

    params = AllChem.ETKDGv3()
    params.randomSeed = seed + 1  # seed=0 produces degenerate conformers in RDKit
    params.pruneRmsThresh = -1.0

    t0 = time.perf_counter()
    AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params)
    raw_ids = [c.GetId() for c in mol.GetConformers()]
    if not raw_ids:
        return RunResult(mol, [], [], time.perf_counter() - t0, 0, failure="embedding_failed")

    minimized = shared_mmff_minimize(mol, raw_ids)
    runtime = time.perf_counter() - t0

    ok_ids = [cid for cid, _ in minimized]
    ok_energies = [e for _, e in minimized]
    return RunResult(mol, ok_ids, ok_energies, runtime, len(raw_ids))


def run_etkdg_oversample(smiles: str, n: int, seed: int, factor: int = 5) -> RunResult:
    """ETKDGv3 x (n * factor), shared MMFF, energy-ranked to top n."""
    mol = _mol_with_hs(smiles)
    if mol is None:
        return RunResult(None, [], [], 0.0, 0, failure="invalid_smiles")

    params = AllChem.ETKDGv3()
    params.randomSeed = seed + 1  # seed=0 produces degenerate conformers in RDKit
    params.pruneRmsThresh = -1.0

    t0 = time.perf_counter()
    AllChem.EmbedMultipleConfs(mol, numConfs=n * factor, params=params)
    raw_ids = [c.GetId() for c in mol.GetConformers()]
    if not raw_ids:
        return RunResult(mol, [], [], time.perf_counter() - t0, 0, failure="embedding_failed")

    minimized = shared_mmff_minimize(mol, raw_ids)
    runtime = time.perf_counter() - t0

    # Energy-rank and take top n
    top = sorted(minimized, key=lambda x: x[1])[:n]
    ok_ids = [cid for cid, _ in top]
    ok_energies = [e for _, e in top]
    return RunResult(mol, ok_ids, ok_energies, runtime, len(raw_ids))


# ──────────────────────────────────────────────────────────────────────────────
# OpenConf runner (imports openconf at call time to keep this lib independent)
# ──────────────────────────────────────────────────────────────────────────────


def run_openconf(smiles: str, n: int, seed: int) -> RunResult:
    """OpenConf with docking-style config (wide energy window, uniform sampling)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from openconf.api import generate_conformers
    from openconf.config import ConformerConfig

    config = ConformerConfig(
        max_out=n,
        energy_window_kcal=18.0,  # wide window maximizes bioactive recall
        parent_strategy="uniform",  # diversity over exploitation
        final_select="diverse",
        do_final_refine=True,  # want proper MMFF energies for Track B
        random_seed=seed,
    )

    t0 = time.perf_counter()
    try:
        ensemble = generate_conformers(smiles, config=config)
        runtime = time.perf_counter() - t0
        return RunResult(
            mol=ensemble.mol,
            conf_ids=ensemble.conf_ids,
            energies=ensemble.energies,
            runtime_s=runtime,
            n_raw=len(ensemble.conf_ids),
        )
    except Exception as exc:
        return RunResult(None, [], [], time.perf_counter() - t0, 0, failure=str(exc)[:100])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────


def load_iridium(data_dir: str, n_molecules: int | None = None) -> list[MolRecord]:
    """Load Iridium SDF files and extract molecule metadata."""
    from pathlib import Path

    sdf_files = sorted(Path(data_dir).glob("*.sdf"))
    if n_molecules:
        sdf_files = sdf_files[:n_molecules]

    records: list[MolRecord] = []
    for sdf_path in sdf_files:
        try:
            suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
            ref_mol = next(suppl, None)
            if ref_mol is None:
                continue

            mol_nohs = Chem.RemoveHs(ref_mol)
            smiles = Chem.MolToSmiles(mol_nohs)
            if not smiles:
                continue

            ri = mol_nohs.GetRingInfo()
            largest_ring = max((len(r) for r in ri.AtomRings()), default=0)

            records.append(
                MolRecord(
                    mol_id=sdf_path.stem.split("_")[0],
                    ref_mol=ref_mol,
                    smiles=smiles,
                    n_heavy=mol_nohs.GetNumAtoms(),
                    n_rotatable=Descriptors.NumRotatableBonds(mol_nohs),
                    mw=Descriptors.MolWt(mol_nohs),
                    is_macrocycle=largest_ring >= 12,
                    largest_ring=largest_ring,
                )
            )
        except Exception:
            continue

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────


def bootstrap_ci(
    vals: list[float],
    stat: Callable = np.median,
    n_boot: int = 1000,
    ci: float = 95.0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic."""
    if not vals:
        return (float("nan"), float("nan"))
    boots = [stat(np.random.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    lo = (100 - ci) / 2
    return float(np.percentile(boots, lo)), float(np.percentile(boots, 100 - lo))


def success_rate(vals: list[float], threshold: float) -> float:
    return sum(v < threshold for v in vals) / len(vals) if vals else 0.0


def add_energy_gaps(records: list[MolRunRecord]) -> None:
    """Compute energy_gap_kcal relative to best-known across all methods/seeds.

    Best-known = minimum energy for the same (mol_id, budget) combination.
    Mutates records in place.
    """
    best: dict[tuple[str, int], float] = {}
    for r in records:
        key = (r.mol_id, r.budget)
        if r.lowest_energy_kcal < float("inf"):
            best[key] = min(best.get(key, float("inf")), r.lowest_energy_kcal)

    for r in records:
        bk = best.get((r.mol_id, r.budget), float("inf"))
        r.energy_gap_kcal = (r.lowest_energy_kcal - bk) if bk < float("inf") else float("inf")
