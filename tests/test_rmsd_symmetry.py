"""Regression tests for symmetry-corrected ensemble analysis."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from openconf import ConformerEnsemble
from openconf.pool import ConformerRecord


@pytest.mark.parametrize("heavy_only", [True, False])
@pytest.mark.parametrize("method", ["rmsd_to", "pairwise_rmsd"])
def test_symmetric_methyl_swap_preserves_coordinates(method: str, heavy_only: bool) -> None:
    """Equivalent isopentane methyl groups have zero RMSD without coordinate mutation."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)CC"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    mol.GetConformer().SetId(7)
    permutation = list(range(mol.GetNumAtoms()))
    permutation[0], permutation[2] = 2, 0
    methyl_hs = [[a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetAtomicNum() == 1] for i in (0, 2)]
    for left, right in zip(*methyl_hs, strict=True):
        permutation[left], permutation[right] = right, left
    swapped = Chem.Conformer(mol.GetConformer(7))
    swapped.SetId(23)
    for idx, source in enumerate(permutation):
        swapped.SetAtomPosition(idx, mol.GetConformer(7).GetAtomPosition(source))
    mol.AddConformer(swapped, assignId=False)
    ensemble = ConformerEnsemble(mol, [ConformerRecord(conf_id=i) for i in (23, 7)])
    before = [conf.GetPositions().copy() for conf in mol.GetConformers()]

    if method == "rmsd_to":
        result = ensemble.rmsd_to(heavy_only=heavy_only)
    else:
        result = ensemble.pairwise_rmsd(heavy_only=heavy_only)

    np.testing.assert_allclose(result, 0.0, atol=1e-6)
    for conf, positions in zip(mol.GetConformers(), before, strict=True):
        np.testing.assert_array_equal(conf.GetPositions(), positions)
