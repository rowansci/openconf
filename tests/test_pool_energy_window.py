"""Regression tests for final energy windows after lowering pool minimum."""

from typing import Literal

import pytest
from rdkit import Chem

from openconf import ConformerConfig, generate_conformers
from openconf.pool import ConformerPool


@pytest.mark.parametrize("rmsd_threshold", [None, 1e-6])
@pytest.mark.parametrize("final_select", ["energy", "diverse"])
@pytest.mark.parametrize("max_out", [2, 10])
@pytest.mark.parametrize("protected", [False, True])
def test_final_selection_rechecks_energy_window(
    rmsd_threshold: float | None, final_select: Literal["energy", "diverse"], max_out: int, protected: bool
) -> None:
    """Filter stale high energies while retaining protected and boundary records."""
    mol = Chem.MolFromSmiles("CC")
    for index in range(4):
        conformer = Chem.Conformer(2)
        conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
        conformer.SetAtomPosition(1, (1.0 + index * 0.5, 0.0, 0.0))
        mol.AddConformer(conformer, assignId=True)
    pool = ConformerPool(
        mol,
        ConformerConfig(
            energy_window_kcal=2.0,
            dedupe_rmsd_threshold=rmsd_threshold,
            final_select=final_select,
            max_out=max_out,
            pool_max=10,
        ),
    )
    assert pool.insert(0, 10.0, tags={"protected": protected})
    assert pool.insert(1, 5.0)
    assert pool.insert(2, 0.0)
    assert pool.insert(3, 2.0)
    selected = pool.select_final()
    expected = {0, 2, 3} if protected else {2, 3}
    assert set(selected) <= expected
    assert 2 in selected
    assert len(selected) == min(max_out, len(expected))
    assert (0 in selected) is protected
    if max_out == 10 or not protected:
        assert 3 in selected


def test_generated_conformers_respect_final_energy_window() -> None:
    """Exclude earlier accepted butane seeds outside final minimum's window."""
    window = 0.01
    ensemble = generate_conformers(
        "CCCC",
        config=ConformerConfig(
            energy_window_kcal=window,
            n_seeds=20,
            seed_prune_rms_thresh=-1.0,
            random_seed=42,
            num_threads=1,
            n_steps=0,
            do_final_refine=False,
            dedupe_rmsd_threshold=None,
        ),
    )
    assert ensemble.n_conformers > 0
    energies = ensemble.energies
    assert max(energies) - min(energies) <= window
