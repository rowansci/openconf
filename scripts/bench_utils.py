"""Shared utilities for benchmark scripts."""

import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    rdFMCS,
    rdMolAlign,
)

from openconf.api import generate_conformers
from openconf.config import ConformerConfig
from openconf.perceive import build_rotor_model, prepare_molecule


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Extract largest fragment from molecule."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda x: x.GetNumAtoms()) if frags else mol


def compute_min_rmsd(gen_mol: Chem.Mol, ref_mol: Chem.Mol) -> float:
    """Compute minimum RMSD between generated conformers and reference (heavy atoms, symmetry-aware)."""
    if gen_mol.GetNumConformers() == 0:
        return float("inf")

    gen = largest_fragment(Chem.RemoveHs(gen_mol))
    ref = largest_fragment(Chem.RemoveHs(ref_mol))
    Chem.SanitizeMol(gen)
    Chem.SanitizeMol(ref)

    min_rmsd = float("inf")
    for conf in gen.GetConformers():
        try:
            rmsd = rdMolAlign.GetBestRMS(gen, ref, prbId=conf.GetId(), refId=0)
            min_rmsd = min(min_rmsd, rmsd)
        except Exception:
            # Fallback: MCS-based alignment
            try:
                mcs = rdFMCS.FindMCS([gen, ref], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=2)
                patt = Chem.MolFromSmarts(mcs.smartsString)
                if patt and (gm := gen.GetSubstructMatch(patt)) and (rm := ref.GetSubstructMatch(patt)):
                    if len(gm) == len(rm):
                        rmsd = rdMolAlign.AlignMol(
                            gen,
                            ref,
                            prbCid=conf.GetId(),
                            refCid=0,
                            map=list(zip(gm, rm, strict=True)),  # type:ignore[unknown-argument]
                        )
                        min_rmsd = min(min_rmsd, float(rmsd))
            except Exception:
                pass
    return min_rmsd


def load_reference(sdf_path: Path) -> tuple[Chem.Mol, str, int, int] | None:
    """Load reference molecule from SDF and extract properties.

    Returns: (ref_mol, smiles, n_heavy, n_rotatable) or None if failed.
    """
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    ref_mol = next(suppl, None)
    if ref_mol is None:
        return None

    smiles = Chem.MolToSmiles(Chem.RemoveHs(ref_mol))
    if not smiles:
        return None

    n_heavy = Descriptors.HeavyAtomCount(ref_mol)
    try:
        mol_prep = prepare_molecule(Chem.MolFromSmiles(smiles), add_hs=True)
        n_rotatable = build_rotor_model(mol_prep).n_rotatable
    except Exception:
        n_rotatable = Descriptors.NumRotatableBonds(ref_mol)

    return ref_mol, smiles, n_heavy, n_rotatable


def run_openconf(smiles: str, max_confs: int = 200) -> tuple[Chem.Mol | None, float]:
    """Run OpenConf. Returns (mol, time_sec)."""
    config = ConformerConfig(
        max_out=max_confs,
        n_seeds=min(200, max_confs),
        n_steps=max(100, max_confs * 2),
        pool_max=max(500, max_confs * 5),
        energy_window_kcal=1e9,
        parent_strategy="uniform",
        final_select="diverse",
    )
    start = time.perf_counter()
    try:
        ensemble = generate_conformers(smiles, config=config)
        return ensemble.mol, time.perf_counter() - start
    except Exception:
        return None, 0.0


def run_etkdg(smiles: str, max_confs: int = 200) -> tuple[Chem.Mol | None, float]:
    """Run ETKDGv3 + MMFF. Returns (mol, time_sec)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, 0.0
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = 0.5

    start = time.perf_counter()
    AllChem.EmbedMultipleConfs(mol, numConfs=max_confs, params=params)
    for cid in range(mol.GetNumConformers()):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
        except Exception:
            pass
    return mol, time.perf_counter() - start


def pct_below(vals: list[float], threshold: float) -> float:
    """Percentage of values below threshold."""
    return 100 * sum(1 for v in vals if v < threshold) / len(vals) if vals else 0.0
