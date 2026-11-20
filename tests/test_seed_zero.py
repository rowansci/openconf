"""Regression tests for reproducible generation with seed zero."""

import numpy as np
import pytest

from openconf import ConformerConfig, generate_conformers


@pytest.mark.parametrize("smiles", ["CCCC", "CCCCCCCCO"])
def test_seed_zero_reproduces_coordinates(smiles: str) -> None:
    """Seed zero reproduces coordinates and energies on both generation paths."""
    config = ConformerConfig(
        max_out=5,
        pool_max=30,
        n_seeds=5,
        n_steps=20,
        num_threads=1,
        random_seed=0,
    )
    first = generate_conformers(smiles, config=config)
    second = generate_conformers(smiles, config=config)

    assert first.n_conformers > 0
    assert first.n_conformers == second.n_conformers
    for idx in range(first.n_conformers):
        np.testing.assert_array_equal(first.coords(idx), second.coords(idx))
    np.testing.assert_array_equal(first.energies, second.energies)
