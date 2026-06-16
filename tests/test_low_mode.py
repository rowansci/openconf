"""Unit and integration tests for low mode following conformer seeding."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from openconf import ConformerConfig, generate_conformers, preset_config
from openconf.perceive import prepare_molecule
from openconf.propose.low_mode import (
    _DEFAULT_EIGENVALUE_THRESHOLD,
    _DEFAULT_MAX_MODES,
    _DEFAULT_SCAN_ENERGY_THRESHOLD,
    _DEFAULT_SCAN_MAX_STEPS,
    _DEFAULT_SCAN_STEP_SIZE,
    _compute_hessian,
    _scan_along_mode,
    _select_low_modes,
    generate_low_mode_seeds,
)
from openconf.relax import RDKitMMFFMinimizer


def _make_minimized_mol(smiles: str, seed: int = 1) -> tuple[Chem.Mol, object]:
    """Return (mol, ff_props) for a fully H-added, MMFF-minimized molecule."""
    mol = prepare_molecule(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    assert props is not None, f"MMFF typing failed for {smiles}"
    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
    assert ff is not None
    ff.Minimize(maxIts=500)
    return mol, props


def _make_fast_minimizer(mol: Chem.Mol) -> RDKitMMFFMinimizer:
    """Return an RDKitMMFFMinimizer prepared for mol."""
    m = RDKitMMFFMinimizer(max_iters=50)
    m.prepare(mol)
    return m


# ---------------------------------------------------------------------------
# _compute_hessian
# ---------------------------------------------------------------------------


def test_hessian_shape():
    """Hessian must have shape (3N, 3N) for N-atom molecule."""
    mol, props = _make_minimized_mol("CCC")
    n = mol.GetNumAtoms()
    H = _compute_hessian(mol, props, conf_id=0)
    assert H.shape == (3 * n, 3 * n)


def test_hessian_is_symmetric():
    """Hessian must be symmetric to within numerical noise."""
    mol, props = _make_minimized_mol("CCC")
    H = _compute_hessian(mol, props, conf_id=0)
    assert np.allclose(H, H.T, atol=1e-6), "Hessian is not symmetric"


def test_hessian_restores_positions():
    """Conformer coordinates must be unchanged after Hessian computation."""
    mol, props = _make_minimized_mol("CCC")
    pos_before = mol.GetConformer(0).GetPositions().copy()
    _compute_hessian(mol, props, conf_id=0)
    pos_after = mol.GetConformer(0).GetPositions()
    assert np.allclose(pos_before, pos_after, atol=1e-10), "Positions were not restored"


def test_hessian_positive_eigenvalues_at_minimum():
    """Non-rigid eigenvalues should be positive at a converged minimum."""
    mol, props = _make_minimized_mol("CCC")
    H = _compute_hessian(mol, props, conf_id=0)
    eigenvalues = np.linalg.eigvalsh(H)
    conformational = eigenvalues[6:]
    neg = conformational[conformational < -0.5]
    assert np.all(conformational > -0.5), f"Unexpected large negative eigenvalues: {neg}"


# ---------------------------------------------------------------------------
# _select_low_modes
# ---------------------------------------------------------------------------


def test_select_low_modes_returns_correct_shape():
    """Selected modes must form a (3N, k) matrix with k ≤ max_modes."""
    mol, props = _make_minimized_mol("CCC")
    H = _compute_hessian(mol, props, conf_id=0)
    n_dof = 3 * mol.GetNumAtoms()
    vecs = _select_low_modes(H, mol, 0, _DEFAULT_EIGENVALUE_THRESHOLD, _DEFAULT_MAX_MODES)
    assert vecs.shape[0] == n_dof
    assert vecs.shape[1] <= _DEFAULT_MAX_MODES


def test_select_low_modes_empty_when_threshold_zero():
    """No modes returned when threshold is effectively zero."""
    mol, props = _make_minimized_mol("CCC")
    H = _compute_hessian(mol, props, conf_id=0)
    vecs = _select_low_modes(H, mol, 0, eigenvalue_threshold=0.0, max_modes=10)
    assert vecs.shape[1] == 0


def test_select_low_modes_respects_max_modes():
    """Result must contain at most max_modes columns."""
    mol, props = _make_minimized_mol("CCCCC")
    H = _compute_hessian(mol, props, conf_id=0)
    for cap in [1, 2, 3]:
        vecs = _select_low_modes(H, mol, 0, eigenvalue_threshold=500.0, max_modes=cap)
        assert vecs.shape[1] <= cap, f"Got {vecs.shape[1]} modes with cap={cap}"


def test_select_low_modes_columns_are_unit_vectors():
    """Returned eigenvectors must have unit norm (they come from eigh)."""
    mol, props = _make_minimized_mol("CCC")
    H = _compute_hessian(mol, props, conf_id=0)
    vecs = _select_low_modes(H, mol, 0, _DEFAULT_EIGENVALUE_THRESHOLD, _DEFAULT_MAX_MODES)
    if vecs.shape[1] == 0:
        pytest.skip("No low modes found — threshold may need adjustment")
    norms = np.linalg.norm(vecs, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-10), f"Mode norms not 1.0: {norms}"


# ---------------------------------------------------------------------------
# _scan_along_mode
# ---------------------------------------------------------------------------


def test_scan_along_mode_moves_from_start():
    """Scan must return positions that differ from the starting geometry."""
    mol, props = _make_minimized_mol("CCC")
    H = _compute_hessian(mol, props, conf_id=0)
    vecs = _select_low_modes(H, mol, 0, _DEFAULT_EIGENVALUE_THRESHOLD, 1)
    if vecs.shape[1] == 0:
        pytest.skip("No low modes found")

    n_atoms = mol.GetNumAtoms()
    direction = vecs[:, 0].reshape(n_atoms, 3)
    start_pos = mol.GetConformer(0).GetPositions().copy()

    final_pos = _scan_along_mode(
        mol,
        props,
        0,
        direction,
        _DEFAULT_SCAN_STEP_SIZE,
        _DEFAULT_SCAN_ENERGY_THRESHOLD,
        _DEFAULT_SCAN_MAX_STEPS,
    )
    assert not np.allclose(final_pos, start_pos), "Scan returned starting positions"


def test_scan_along_mode_restores_start_conformer():
    """The start conformer must be unchanged after scanning."""
    mol, props = _make_minimized_mol("CCC")
    H = _compute_hessian(mol, props, conf_id=0)
    vecs = _select_low_modes(H, mol, 0, _DEFAULT_EIGENVALUE_THRESHOLD, 1)
    if vecs.shape[1] == 0:
        pytest.skip("No low modes found")

    n_atoms = mol.GetNumAtoms()
    direction = vecs[:, 0].reshape(n_atoms, 3)
    pos_before = mol.GetConformer(0).GetPositions().copy()
    n_confs_before = mol.GetNumConformers()

    _scan_along_mode(
        mol,
        props,
        0,
        direction,
        _DEFAULT_SCAN_STEP_SIZE,
        _DEFAULT_SCAN_ENERGY_THRESHOLD,
        _DEFAULT_SCAN_MAX_STEPS,
    )

    assert mol.GetNumConformers() == n_confs_before, "Scan left a temporary conformer behind"
    assert np.allclose(mol.GetConformer(0).GetPositions(), pos_before, atol=1e-10)


def test_scan_stops_at_energy_threshold():
    """With a near-zero energy threshold, scan must stop after the first step."""
    mol, props = _make_minimized_mol("CCC")
    H = _compute_hessian(mol, props, conf_id=0)
    vecs = _select_low_modes(H, mol, 0, _DEFAULT_EIGENVALUE_THRESHOLD, 1)
    if vecs.shape[1] == 0:
        pytest.skip("No low modes found")

    n_atoms = mol.GetNumAtoms()
    direction = vecs[:, 0].reshape(n_atoms, 3)
    start_pos = mol.GetConformer(0).GetPositions().copy()

    # Threshold of 0 means any energy increase (even numerical noise) stops the scan.
    # The returned positions should still be the start positions (first step rejected).
    final_pos = _scan_along_mode(
        mol,
        props,
        0,
        direction,
        _DEFAULT_SCAN_STEP_SIZE,
        energy_threshold=0.0,
        max_steps=10,
    )
    # Either no progress or exactly one step taken — at most a tiny displacement
    displacement = float(np.linalg.norm(final_pos - start_pos))
    assert displacement < _DEFAULT_SCAN_STEP_SIZE + 1e-6


# ---------------------------------------------------------------------------
# generate_low_mode_seeds
# ---------------------------------------------------------------------------


@pytest.fixture
def propane_mol_props() -> tuple[Chem.Mol, object]:
    """Propane molecule with MMFF properties, minimized."""
    return _make_minimized_mol("CCC")


def test_generate_low_mode_seeds_returns_finite_energies(
    propane_mol_props: tuple[Chem.Mol, object],
) -> None:
    """All returned seeds must have finite MMFF energies."""
    mol, props = propane_mol_props
    minimizer = _make_fast_minimizer(mol)
    seeds = generate_low_mode_seeds(mol, props, conf_id=0, minimizer=minimizer)
    for _, energy in seeds:
        assert np.isfinite(energy), f"Non-finite energy in low mode seed: {energy}"


def test_generate_low_mode_seeds_adds_conformers_to_mol(
    propane_mol_props: tuple[Chem.Mol, object],
) -> None:
    """Each returned seed conformer must exist in the molecule."""
    mol, props = propane_mol_props
    minimizer = _make_fast_minimizer(mol)
    n_before = mol.GetNumConformers()
    seeds = generate_low_mode_seeds(mol, props, conf_id=0, minimizer=minimizer)
    existing_ids = {c.GetId() for c in mol.GetConformers()}
    for conf_id, _ in seeds:
        assert conf_id in existing_ids, f"Seed conf_id {conf_id} not found in mol"
    assert mol.GetNumConformers() >= n_before


def test_generate_low_mode_seeds_empty_when_threshold_zero(
    propane_mol_props: tuple[Chem.Mol, object],
) -> None:
    """No seeds generated when eigenvalue threshold excludes all conformational modes."""
    mol, props = propane_mol_props
    minimizer = _make_fast_minimizer(mol)
    seeds = generate_low_mode_seeds(mol, props, conf_id=0, minimizer=minimizer, eigenvalue_threshold=0.0)
    assert seeds == []


def test_generate_low_mode_seeds_at_most_two_per_mode(
    propane_mol_props: tuple[Chem.Mol, object],
) -> None:
    """With max_modes=k, at most 2k seeds are returned (two directions per mode)."""
    mol, props = propane_mol_props
    minimizer = _make_fast_minimizer(mol)
    for cap in [1, 2]:
        n_before = mol.GetNumConformers()
        seeds = generate_low_mode_seeds(mol, props, conf_id=0, minimizer=minimizer, max_modes=cap)
        assert len(seeds) <= 2 * cap, f"Got {len(seeds)} seeds with cap={cap}"
        for conf_id, _ in seeds:
            mol.RemoveConformer(conf_id)
        assert mol.GetNumConformers() == n_before


def test_generate_low_mode_seeds_scans_both_directions() -> None:
    """Both the + and − scan directions must be explored for each mode."""
    mol, props = _make_minimized_mol("CCCCCC")
    minimizer = _make_fast_minimizer(mol)

    seeds = generate_low_mode_seeds(mol, props, conf_id=0, minimizer=minimizer, max_modes=1)
    # For one mode with two directions we expect 0 or 2 seeds (not 1),
    # because both directions are always attempted.
    assert len(seeds) in (0, 2), f"Expected 0 or 2 seeds for 1 mode, got {len(seeds)}"
    for conf_id, _ in seeds:
        mol.RemoveConformer(conf_id)


def test_generate_low_mode_seeds_perturbs_geometry() -> None:
    """Seeds must be displaced from the starting geometry after minimization."""
    mol, props = _make_minimized_mol("CCCCCC")
    minimizer = _make_fast_minimizer(mol)
    pos0 = mol.GetConformer(0).GetPositions().copy()

    seeds = generate_low_mode_seeds(mol, props, conf_id=0, minimizer=minimizer)
    if not seeds:
        pytest.skip("No low modes below threshold for this molecule")

    any_moved = False
    for conf_id, _ in seeds:
        pos_new = mol.GetConformer(conf_id).GetPositions()
        rmsd = float(np.sqrt(np.mean(np.sum((pos_new - pos0) ** 2, axis=1))))
        if rmsd > 0.01:
            any_moved = True
        mol.RemoveConformer(conf_id)

    assert any_moved, "All low-mode seeds minimized back to the starting geometry"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_default_low_mode_off():
    """use_low_mode_following must be False by default."""
    assert ConformerConfig().use_low_mode_following is False


def test_config_low_mode_threshold_must_be_positive():
    """Negative or zero eigenvalue threshold must raise ValueError."""
    with pytest.raises(ValueError, match="low_mode_eigenvalue_threshold"):
        ConformerConfig(low_mode_eigenvalue_threshold=0.0)


def test_config_low_mode_max_modes_must_be_at_least_one():
    """max_modes < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="low_mode_max_modes"):
        ConformerConfig(low_mode_max_modes=0)


def test_config_scan_step_size_must_be_positive():
    """Non-positive scan step size must raise ValueError."""
    with pytest.raises(ValueError, match="low_mode_scan_step_size"):
        ConformerConfig(low_mode_scan_step_size=0.0)


def test_config_scan_energy_threshold_must_be_positive():
    """Non-positive energy threshold must raise ValueError."""
    with pytest.raises(ValueError, match="low_mode_scan_energy_threshold"):
        ConformerConfig(low_mode_scan_energy_threshold=0.0)


def test_config_scan_max_steps_must_be_at_least_one():
    """scan_max_steps < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="low_mode_scan_max_steps"):
        ConformerConfig(low_mode_scan_max_steps=0)


def test_config_n_source_seeds_must_be_at_least_one():
    """low_mode_n_source_seeds < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="low_mode_n_source_seeds"):
        ConformerConfig(low_mode_n_source_seeds=0)


def test_n_source_seeds_limits_hessian_evaluations():
    """With n_source_seeds=1, only one seed triggers low mode scanning."""
    config = ConformerConfig(
        use_low_mode_following=True,
        low_mode_n_source_seeds=1,
        n_seeds=5,
        n_steps=10,
        max_out=5,
        random_seed=0,
        collect_stats=True,
        do_final_refine=False,
    )
    ens = generate_conformers("CCCC", config=config)
    stats = ens.generation_stats
    # With n_source_seeds=1 and max_modes=5, at most 2*5=10 low mode seeds
    # can be generated (two directions per mode); this is an upper bound check.
    assert int(stats.get("n_low_mode_seeds", 0)) <= 10


# ---------------------------------------------------------------------------
# Integration: stats populated when enabled
# ---------------------------------------------------------------------------


def test_low_mode_following_stat_populated():
    """With use_low_mode_following=True and collect_stats=True, stat keys appear."""
    config = ConformerConfig(
        use_low_mode_following=True,
        n_steps=20,
        n_seeds=3,
        max_out=5,
        random_seed=42,
        collect_stats=True,
        do_final_refine=False,
    )
    ens = generate_conformers("CCCC", config=config)
    stats = ens.generation_stats
    assert stats
    assert "low_mode_time_s" in stats
    assert "n_low_mode_seeds" in stats
    assert float(stats["low_mode_time_s"]) >= 0.0
    assert int(stats["n_low_mode_seeds"]) >= 0


def test_low_mode_following_survives_pool_overflow():
    """Source seeds must not be evicted before their low modes are scanned.

    With a small pool and several source seeds, the low-mode children can
    overflow the pool partway through. Eviction of a not-yet-processed source
    seed used to leave generate_low_mode_seeds referencing a freed conformer
    (Bad Conformer Id or a hard MMFF abort). Generation must complete cleanly.
    """
    config = ConformerConfig(
        use_low_mode_following=True,
        low_mode_n_source_seeds=5,
        n_seeds=8,
        n_steps=10,
        max_out=5,
        pool_max=5,
        random_seed=0,
        collect_stats=True,
        do_final_refine=False,
    )
    ens = generate_conformers("CCCCCCCC", config=config)
    assert ens.n_conformers > 0
    assert int(ens.generation_stats.get("n_low_mode_seeds", 0)) >= 0


def test_low_mode_following_stat_absent_when_disabled():
    """With use_low_mode_following=False, low_mode_time_s must be 0."""
    config = ConformerConfig(
        use_low_mode_following=False,
        n_steps=10,
        n_seeds=2,
        max_out=3,
        random_seed=0,
        collect_stats=True,
        do_final_refine=False,
    )
    ens = generate_conformers("CCC", config=config)
    stats = ens.generation_stats
    assert float(stats.get("low_mode_time_s", 0.0)) == 0.0
    assert int(stats.get("n_low_mode_seeds", 0)) == 0


# ---------------------------------------------------------------------------
# Macrocycle preset
# ---------------------------------------------------------------------------


def test_macrocycle_preset_config_values():
    """Macrocycle preset must have wide energy window and low-mode following on."""
    config = preset_config("macrocycle")
    assert config.energy_window_kcal == 100.0
    assert config.use_low_mode_following is True
    assert config.parent_strategy == "softmax"
    assert config.max_out == 200
    assert config.final_select == "diverse"


def test_macrocycle_preset_generates_conformers():
    """Macrocycle preset must produce at least one conformer on a simple macrocycle."""
    import dataclasses

    config = dataclasses.replace(
        preset_config("macrocycle"),
        max_out=5,
        n_steps=20,
        n_seeds=4,
        random_seed=7,
        do_final_refine=False,
    )
    ens = generate_conformers("C1CCCCCCCCCCC1", config=config)
    assert ens.n_conformers > 0
