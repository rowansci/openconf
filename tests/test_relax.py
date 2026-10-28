"""Tests for MMFF relaxation backends."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from openconf.relax import minimize_confs_mmff


def _max_gradient(mol: Chem.Mol, conf_ids: list[int], dielectric: float) -> float:
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    props.SetMMFFDielectricConstant(dielectric)
    return max(
        max(abs(gradient) for gradient in AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id).CalcGrad())
        for conf_id in conf_ids
    )


def test_minimize_confs_mmff_minimizes_at_requested_dielectric():
    """Geometry converges at dielectric carried by mmff_props, not default eps=1."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NCC(=O)[O-]"))
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=8, randomSeed=0))

    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    props.SetMMFFDielectricConstant(10.0)
    energies = minimize_confs_mmff(mol, props, conf_ids, max_iters=2000)

    assert len(energies) == len(conf_ids)
    assert _max_gradient(mol, conf_ids, 10.0) < 1e-2
    assert _max_gradient(mol, conf_ids, 1.0) > 1.0
    for conf_id, energy in zip(conf_ids, energies, strict=True):
        force_field = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        assert energy == pytest.approx(force_field.CalcEnergy(), abs=1e-6)
