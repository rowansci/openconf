"""Regression tests for stereochemistry inferred from 3D inputs."""

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from openconf import generate_conformers, preset_config
from openconf.api import _filter_stereochemistry_consistent_conformers
from openconf.config import ConformerConfig
from openconf.perceive import (
    build_rotor_model,
    conformer_matches_specified_stereochemistry,
    prepare_molecule,
    specified_stereochemistry,
    stereochemistry_from_conformer,
)
from openconf.propose.hybrid import HybridProposer, _copy_conformer
from openconf.torsionlib import TorsionLibrary

_CRANKSHAFT_SMILES = "CN1CC[C@]23c4c5ccc(OC(=O)c6cccnc6)c4O[C@@H]2CC=C[C@H]3[C@H]1C5"
_INVERTED_CRANKSHAFT_SMILES = "CN1CC[C@]23c4c5ccc(OC(=O)c6cccnc6)c4O[C@@H]2CC=C[C@@H]3[C@H]1C5"


@pytest.mark.parametrize(
    ("smiles", "flipped_smiles"),
    [
        ("C/C=C/C", r"C/C=C\C"),
        ("C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"),
    ],
    ids=["2-butene", "lactic-acid"],
)
def test_3d_input_filters_stereochemical_flips(smiles: str, flipped_smiles: str) -> None:
    """Unannotated 3D inputs reject E/Z and tetrahedral flips, even at lower energy."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    flipped = Chem.AddHs(Chem.MolFromSmiles(flipped_smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0
    assert AllChem.EmbedMolecule(flipped, randomSeed=2) == 0
    Chem.RemoveStereochemistry(mol)

    mol = prepare_molecule(mol, add_hs=False)
    reference = specified_stereochemistry(mol)
    mol.AddConformer(Chem.Conformer(flipped.GetConformer()), assignId=True)
    conf_ids, energies = _filter_stereochemistry_consistent_conformers(mol, [0, 1], [0.0, -1.0], reference)

    assert conf_ids == [0]
    assert energies == [0.0]
    assert [conf.GetId() for conf in mol.GetConformers()] == [0]


def test_explicit_stereochemistry_takes_precedence_over_3d() -> None:
    """Explicit trans-2-butene labels survive conflicting cis coordinates."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C/C=C/C"))
    cis = Chem.AddHs(Chem.MolFromSmiles(r"C/C=C\C"))
    assert AllChem.EmbedMolecule(cis, randomSeed=1) == 0
    mol.AddConformer(Chem.Conformer(cis.GetConformer()), assignId=True)

    prepared = prepare_molecule(mol, add_hs=False)

    assert Chem.MolToSmiles(Chem.RemoveHs(prepared)) == "C/C=C/C"


def test_2d_input_keeps_stereochemistry_unspecified() -> None:
    """2D butene coordinates do not specify E/Z stereochemistry."""
    mol = Chem.MolFromSmiles("CC=CC")
    AllChem.Compute2DCoords(mol)

    reference = specified_stereochemistry(prepare_molecule(mol))

    assert reference.tetrahedral == {}
    assert reference.bonds == {}


def _ring_junction_centres(mol: Chem.Mol, reference: dict[int, str]) -> list[int]:
    """Return specified stereocentre indices shared by at least two rings."""
    ring_counts = [0] * mol.GetNumAtoms()
    for ring in mol.GetRingInfo().AtomRings():
        for atom_idx in ring:
            ring_counts[atom_idx] += 1
    return [atom_idx for atom_idx in reference if ring_counts[atom_idx] >= 2]


def test_crankshaft_preserves_specified_ring_junction_stereocentre() -> None:
    """Ensemble generation must not invert the specified ring-junction centre."""
    config = replace(
        preset_config("ensemble"),
        random_seed=42,
        dedupe_rmsd_threshold=1.0,
    )

    ensemble = generate_conformers(_CRANKSHAFT_SMILES, config=config)

    reference = specified_stereochemistry(ensemble.mol)
    junction_centres = _ring_junction_centres(ensemble.mol, reference.tetrahedral)
    assert junction_centres
    assert ensemble.n_conformers > 0
    for conf_id in ensemble.conf_ids:
        observed = stereochemistry_from_conformer(ensemble.mol, conf_id)
        for atom_idx in junction_centres:
            assert observed.tetrahedral.get(atom_idx) == reference.tetrahedral[atom_idx]


def test_inverted_ring_junction_conformer_is_rejected() -> None:
    """A conformer with inverted ring-junction geometry is filtered out."""
    mol = Chem.AddHs(Chem.MolFromSmiles(_CRANKSHAFT_SMILES))
    inverted = Chem.AddHs(Chem.MolFromSmiles(_INVERTED_CRANKSHAFT_SMILES))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    assert AllChem.EmbedMolecule(inverted, randomSeed=7) == 0

    prepared = prepare_molecule(mol, add_hs=False)
    reference = specified_stereochemistry(prepared)
    prepared.AddConformer(Chem.Conformer(inverted.GetConformer()), assignId=True)

    assert not conformer_matches_specified_stereochemistry(prepared, 1, reference)
    kept_ids, kept_energies = _filter_stereochemistry_consistent_conformers(prepared, [0, 1], [0.0, -1.0], reference)

    assert kept_ids == [0]
    assert kept_energies == [0.0]


def test_crankshaft_leaves_unassigned_centres_free_to_change() -> None:
    """Crankshaft still moves rings when no stereocentre is specified."""
    mol = prepare_molecule(Chem.MolFromSmiles("C1CCC2CCCCC2C1"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    rotor_model = build_rotor_model(mol)
    proposer = HybridProposer(mol, rotor_model, TorsionLibrary(), ConformerConfig(random_seed=3))
    orig_id = mol.GetConformers()[0].GetId()
    new_id = _copy_conformer(mol, orig_id)

    before = mol.GetConformer(orig_id).GetPositions().copy()
    proposer._moves.apply_crankshaft_move(new_id)
    after = mol.GetConformer(new_id).GetPositions()

    assert proposer._moves.reference_stereo.tetrahedral == {}
    assert not np.allclose(before, after, atol=0.01)
