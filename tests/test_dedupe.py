"""Tests for conservative Cartesian RMSD deduplication."""

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from openconf import ConformerConfig, generate_conformers, rmsd_dedupe
from openconf.pool import ConformerPool


def _mol_with_conformers(
    atoms: list[int],
    coordinates: list[np.ndarray],
    bonds: list[tuple[int, int]] | None = None,
) -> Chem.Mol:
    editable = Chem.RWMol()
    for atomic_number in atoms:
        editable.AddAtom(Chem.Atom(atomic_number))
    for left, right in bonds or []:
        editable.AddBond(left, right, Chem.BondType.SINGLE)
    mol = editable.GetMol()

    for positions in coordinates:
        conformer = Chem.Conformer(len(atoms))
        for atom_index, (x, y, z) in enumerate(positions):
            conformer.SetAtomPosition(
                atom_index,
                Point3D(float(x), float(y), float(z)),
            )
        mol.AddConformer(conformer, assignId=True)
    return mol


def _rotation_z(angle_degrees: float) -> np.ndarray:
    angle = np.radians(angle_degrees)
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_rmsd_dedupe_retains_lowest_energy_rigid_representative() -> None:
    """Rigid duplicates retain lowest-energy representative."""
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.1, 1.3, 0.4],
            [0.3, 0.1, 1.7],
        ]
    )
    transformed = reference @ _rotation_z(73.0) + np.array([4.0, -2.0, 7.0])
    mol = _mol_with_conformers([6] * 4, [reference, transformed])

    kept = rmsd_dedupe(mol, [0, 1], energies=[1.0, 0.0])

    assert kept == [1]


def test_rmsd_dedupe_does_not_merge_reflections() -> None:
    """Proper-rotation alignment preserves mirror images."""
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.1, 1.3, 0.4],
            [0.3, 0.1, 1.7],
        ]
    )
    reflected = reference.copy()
    reflected[:, 0] *= -1.0
    mol = _mol_with_conformers([6] * 4, [reference, reflected])

    assert rmsd_dedupe(mol, [0, 1], rmsd_threshold=0.1) == [0, 1]


def test_max_atom_deviation_preserves_local_changes() -> None:
    """Per-atom guard prevents localized movement from averaging away."""
    reference = np.column_stack([np.arange(100, dtype=float), np.zeros((100, 2))])
    changed = reference.copy()
    changed[-1, 1] = 1.0
    mol = _mol_with_conformers([6] * 100, [reference, changed])

    guarded = rmsd_dedupe(
        mol,
        [0, 1],
        rmsd_threshold=0.2,
        max_atom_deviation=0.4,
    )
    loose = rmsd_dedupe(
        mol,
        [0, 1],
        rmsd_threshold=0.2,
        max_atom_deviation=2.0,
    )

    assert guarded == [0, 1]
    assert loose == [0]


def test_all_atom_mode_permutes_only_bonded_methyl_hydrogens() -> None:
    """RDKit bond graph identifies local interchangeable hydrogen group."""
    angles = np.radians([0.0, 120.0, 240.0])
    methyl_hydrogens = np.column_stack([np.cos(angles), np.sin(angles), np.full(3, -0.35)])
    reference = np.vstack(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.4],
            [1.2, 0.0, 2.1],
            methyl_hydrogens,
            [0.7, 0.0, 1.9],
        ]
    )
    permuted = reference.copy()
    permuted[[3, 4, 5]] = permuted[[4, 5, 3]]
    atoms = [6, 8, 6, 1, 1, 1, 1]
    bonds = [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 6)]
    mol = _mol_with_conformers(atoms, [reference, permuted], bonds)

    kept = rmsd_dedupe(
        mol,
        [0, 1],
        use_heavy_atoms_only=False,
        rmsd_threshold=0.05,
    )

    assert kept == [0]


def test_none_rmsd_threshold_disables_pool_deduplication() -> None:
    """None threshold leaves pool conformers untouched."""
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mol = _mol_with_conformers([6, 6], [reference, reference.copy()])
    pool = ConformerPool(
        mol=mol,
        config=ConformerConfig(
            max_out=2,
            pool_max=2,
            dedupe_period=1,
            dedupe_rmsd_threshold=None,
        ),
    )
    assert pool.insert(0, 0.0)
    assert pool.insert(1, 0.0)

    assert not pool.should_dedupe()
    assert pool.dedupe() == 0
    assert pool.conf_ids == [0, 1]


def test_reported_nmr_molecule_retains_post_refinement_states() -> None:
    """Reported chiral NMR regression retains twelve Cartesian representatives."""
    config = ConformerConfig(
        n_steps=200,
        max_out=200,
        random_seed=42,
        num_threads=1,
        final_select="energy",
    )

    ensemble = generate_conformers(
        "COc1ccc(N2C[C@@H](C(C)=O)C2=O)cc1",
        config=config,
    )

    # Six distinct final minima, each retained as a symmetry-equivalent pair;
    # fixed-correspondence dedupe keeps both members of each pair.
    assert ensemble.n_conformers == 12
