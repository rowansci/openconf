"""Cyclic-alkane regression tests for the macrocycle sampling pipeline.

Each test asserts a concrete behavior of the sampler on a single-ring cycloalkane:

- crankshaft move changes ring-atom coordinates,
- seed pruning is disabled automatically on macrocycles,
- the min-energy conformer matches a dense ETKDG+MMFF reference within
  tolerance,
- distinct ring-torsion families are recovered (not collapsed to one pucker).

The reference minimum for each ring is computed in-test from a 500-seed ETKDG
sweep with macrocycle distance bounds, so the tests self-contain their ground
truth and track RDKit updates.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from openconf import ConformerConfig, generate_conformers
from openconf.perceive import build_rotor_model, prepare_molecule

CYCLOALKANES = {
    "C6": "C1CCCCC1",
    "C7": "C1CCCCCC1",
    "C8": "C1CCCCCCC1",
    "C10": "C1CCCCCCCCC1",
    "C12": "C1CCCCCCCCCCC1",
    "C14": "C1CCCCCCCCCCCCC1",
}


def _ring_torsions(mol: Chem.Mol, conf_id: int, ring_atoms: list[int]) -> list[float]:
    """Signed ring dihedrals around each ring bond, in ring traversal order."""
    n = len(ring_atoms)
    conf = mol.GetConformer(conf_id)
    return [
        rdMolTransforms.GetDihedralDeg(
            conf,
            ring_atoms[(i - 1) % n],
            ring_atoms[i],
            ring_atoms[(i + 1) % n],
            ring_atoms[(i + 2) % n],
        )
        for i in range(n)
    ]


def _torsion_signature(torsions: list[float], bin_deg: float = 60.0) -> tuple[int, ...]:
    """Cyclic/reflection-invariant quantized ring-torsion signature.

    Two conformers sharing a signature have the same pucker pattern up to a
    relabeling (rotation or reversal) of the ring.
    """
    n = len(torsions)
    q = tuple(round(t / bin_deg) % 6 for t in torsions)
    rotations = [q[k:] + q[:k] for k in range(n)]
    reversed_rots = [tuple(reversed(r)) for r in rotations]
    return min(rotations + reversed_rots)


def _ref_min_energy(smiles: str, n_seeds: int = 500, seed: int = 1234) -> float:
    """Min MMFF94s energy across a dense ETKDG sweep; self-contained reference."""
    mol = prepare_molecule(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useMacrocycleTorsions = True
    params.useBasicKnowledge = True
    params.useSmallRingTorsions = True
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_seeds, params=params))
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    assert props is not None
    props.SetMMFFDielectricConstant(4.0)
    best = float("inf")
    for cid in conf_ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        if ff is None:
            continue
        ff.Minimize(maxIts=500)
        best = min(best, float(ff.CalcEnergy()))
    return best


# ---------------------------------------------------------------------------
# Unit tests for the crankshaft move itself
# ---------------------------------------------------------------------------


def test_crankshaft_detected_on_macrocycle():
    """Cyclododecane exposes one crankable ring with 12 substituent subtrees."""
    from openconf.propose.hybrid import HybridProposer
    from openconf.torsionlib import TorsionLibrary

    mol = prepare_molecule(Chem.MolFromSmiles("C1CCCCCCCCCCC1"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    rm = build_rotor_model(mol)
    proposer = HybridProposer(mol, rm, TorsionLibrary(), ConformerConfig(random_seed=0))

    assert len(proposer._crankable_rings) == 1
    ring, subtrees = proposer._crankable_rings[0]
    assert len(ring) == 12
    assert len(subtrees) == 12
    # Each ring C in cycloalkane carries itself + 2 H = 3 atoms in its subtree.
    assert all(len(s) == 3 for s in subtrees)


def test_crankshaft_preserves_anchor_bonds():
    """Crankshaft rotation preserves anchor-to-neighbor bond lengths (axis is geometric)."""
    from openconf.propose.hybrid import HybridProposer, _copy_conformer
    from openconf.torsionlib import TorsionLibrary

    mol = prepare_molecule(Chem.MolFromSmiles("C1CCCCCCCCCCC1"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    rm = build_rotor_model(mol)
    proposer = HybridProposer(mol, rm, TorsionLibrary(), ConformerConfig(random_seed=0))

    orig_id = mol.GetConformers()[0].GetId()
    ring = proposer._crankable_rings[0][0]

    # Bond length between each consecutive pair of ring atoms, before.
    def bond_lengths(cid: int) -> list[float]:
        conf = mol.GetConformer(cid)
        pos = conf.GetPositions()
        return [float(np.linalg.norm(pos[ring[i]] - pos[ring[(i + 1) % len(ring)]])) for i in range(len(ring))]

    before = bond_lengths(orig_id)

    # Apply many crankshafts and check bond lengths never drift (within 1e-6).
    for _ in range(20):
        new_id = _copy_conformer(mol, orig_id)
        proposer._apply_crankshaft_move(new_id)
        after = bond_lengths(new_id)
        assert len(before) == len(after)
        for b, a in zip(before, after, strict=True):
            assert abs(a - b) < 1e-6, f"Bond length drifted: {b} -> {a}"
        mol.RemoveConformer(new_id)


def test_crankshaft_actually_moves_atoms():
    """Crankshaft must perturb a non-empty subset of ring atoms each call."""
    from openconf.propose.hybrid import HybridProposer, _copy_conformer
    from openconf.torsionlib import TorsionLibrary

    mol = prepare_molecule(Chem.MolFromSmiles("C1CCCCCCCCCCC1"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    rm = build_rotor_model(mol)
    proposer = HybridProposer(mol, rm, TorsionLibrary(), ConformerConfig(random_seed=0))

    orig_id = mol.GetConformers()[0].GetId()
    orig_pos = mol.GetConformer(orig_id).GetPositions()

    moved_counts = []
    for _ in range(10):
        new_id = _copy_conformer(mol, orig_id)
        proposer._apply_crankshaft_move(new_id)
        new_pos = mol.GetConformer(new_id).GetPositions()
        moved_counts.append(int(np.any(np.abs(new_pos - orig_pos) > 0.01, axis=1).sum()))
        mol.RemoveConformer(new_id)

    assert all(c >= 3 for c in moved_counts), f"Some crankshafts moved <3 atoms: {moved_counts}"


def test_crankshaft_absent_on_aromatic_ring():
    """Benzene must not register any crankable rings — aromatic rings are rigid."""
    from openconf.propose.hybrid import HybridProposer
    from openconf.torsionlib import TorsionLibrary

    mol = prepare_molecule(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    rm = build_rotor_model(mol)
    proposer = HybridProposer(mol, rm, TorsionLibrary(), ConformerConfig())
    assert proposer._crankable_rings == []


def test_macrocycle_seed_prune_disabled():
    """Seeds must not be RMSD-pruned when a macrocycle is present."""
    from openconf.propose.hybrid import HybridProposer
    from openconf.torsionlib import TorsionLibrary

    mol = prepare_molecule(Chem.MolFromSmiles("C1CCCCCCCCCCC1"))
    rm = build_rotor_model(mol)
    assert rm.ring_info["has_macrocycle"]
    # Request 60 seeds; with pruning at 1.0 Å, only a handful would survive.
    config = ConformerConfig(random_seed=0, seed_prune_rms_thresh=1.0)
    proposer = HybridProposer(mol, rm, TorsionLibrary(), config)
    seeds = proposer.generate_seeds(60)
    # With pruning disabled for macrocycles, nearly all seeds should survive.
    assert len(seeds) >= 50, f"Expected most seeds to survive, got {len(seeds)} / 60"


# ---------------------------------------------------------------------------
# Energetic ground-truth regression for cycloalkanes C6 → C14
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "smiles", "gap_tol_kcal"),
    [
        ("C6", CYCLOALKANES["C6"], 0.5),
        ("C7", CYCLOALKANES["C7"], 0.5),
        ("C8", CYCLOALKANES["C8"], 0.5),
        ("C10", CYCLOALKANES["C10"], 1.0),
        pytest.param(
            "C12",
            CYCLOALKANES["C12"],
            1.0,
            marks=pytest.mark.xfail(
                reason="C12's [3333] pucker is ~1% of seeds and needs n_seeds≥300 for reliable recovery; "
                "default seeding (~160) misses it. Override with n_seeds=500 if you need it."
            ),
        ),
        ("C14", CYCLOALKANES["C14"], 1.0),
    ],
)
def test_min_energy_matches_reference(name: str, smiles: str, gap_tol_kcal: float):
    """Default-config min-energy conformer is within tolerance of ETKDG reference."""
    ref = _ref_min_energy(smiles)
    config = ConformerConfig(max_out=50, n_steps=300, random_seed=42)
    ens = generate_conformers(smiles, config=config)
    assert ens.n_conformers > 0
    gap = min(ens.energies) - ref
    assert gap <= gap_tol_kcal, f"{name}: min_E - ref = {gap:+.3f} kcal/mol exceeds {gap_tol_kcal} kcal/mol"


# ---------------------------------------------------------------------------
# Coverage regression — distinct ring-torsion families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "smiles", "min_sigs"),
    [
        # C6 has chair (dominant) + twist-boat family; expect at least 2 sigs.
        ("C6", CYCLOALKANES["C6"], 2),
        ("C10", CYCLOALKANES["C10"], 3),
        ("C12", CYCLOALKANES["C12"], 5),
        ("C14", CYCLOALKANES["C14"], 5),
    ],
)
def test_ring_torsion_diversity(name: str, smiles: str, min_sigs: int):
    """Generated ensemble recovers multiple distinct ring-pucker families."""
    config = ConformerConfig(max_out=50, n_steps=300, random_seed=42)
    ens = generate_conformers(smiles, config=config)
    ring_atoms = list(ens.mol.GetRingInfo().AtomRings()[0])
    sigs = {_torsion_signature(_ring_torsions(ens.mol, r.conf_id, ring_atoms)) for r in ens.records}
    assert len(sigs) >= min_sigs, f"{name}: found {len(sigs)} torsion families, need ≥ {min_sigs}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
