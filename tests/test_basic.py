"""Basic tests for openconf."""

import pytest
from rdkit import Chem


def test_prepare_molecule():
    """Test molecule preparation."""
    from openconf import prepare_molecule

    mol = Chem.MolFromSmiles("CCO")
    mol = prepare_molecule(mol)

    # Should have hydrogens
    assert mol.GetNumAtoms() > 3


def test_build_rotor_model():
    """Test rotor model building."""
    from openconf import build_rotor_model, prepare_molecule

    # Ethane - 1 rotatable bond
    mol = Chem.MolFromSmiles("CC")
    mol = prepare_molecule(mol)
    rotor_model = build_rotor_model(mol)

    # Ethane has 0 rotatable bonds by typical definition (terminal)
    # But let's just check it runs
    assert rotor_model.mol is not None

    # Butane - should have rotatable bonds
    mol = Chem.MolFromSmiles("CCCC")
    mol = prepare_molecule(mol)
    rotor_model = build_rotor_model(mol)
    assert rotor_model.n_rotatable >= 1


def test_torsion_library():
    """Test torsion library."""
    from openconf import TorsionLibrary
    from openconf.torsionlib import get_default_torsion_library
    from openconf.tuning import get_default_move_probs

    lib = TorsionLibrary()
    assert len(lib) > 0

    # Check that patterns compile
    for pattern, _rule, _pos2, _pos3 in lib._compiled:
        assert pattern is not None

    assert get_default_torsion_library() is get_default_torsion_library()
    assert get_default_move_probs() is not get_default_move_probs()


def test_minimizer():
    """Test RDKit minimizer."""
    from rdkit.Chem import AllChem

    from openconf import RDKitMMFFMinimizer, prepare_molecule

    mol = Chem.MolFromSmiles("CCO")
    mol = prepare_molecule(mol)
    AllChem.EmbedMolecule(mol)

    minimizer = RDKitMMFFMinimizer()
    minimizer.prepare(mol)
    energy = minimizer.minimize(mol, 0)

    assert isinstance(energy, float)
    assert energy < 1e6  # Should be a reasonable value


def test_generate_conformers_simple():
    """Test basic conformer generation."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(
        max_out=10,
        n_seeds=5,
        n_steps=20,
        pool_max=50,
    )

    ensemble = generate_conformers("CCO", config=config)

    assert ensemble.n_conformers > 0
    assert ensemble.n_conformers <= 10
    assert len(ensemble.energies) == ensemble.n_conformers


def test_generate_conformers_accepts_custom_torsion_library():
    """Generation accepts a caller-supplied torsion library."""
    from openconf import ConformerConfig, TorsionLibrary, generate_conformers

    config = ConformerConfig(
        max_out=5,
        n_seeds=3,
        n_steps=10,
        pool_max=20,
        random_seed=0,
    )

    ensemble = generate_conformers("CCCC", config=config, torsion_library=TorsionLibrary())

    assert ensemble.n_conformers > 0


def test_low_flex_path_dispatch(monkeypatch):
    """Very low-flexibility molecules use the ETKDG-only fast path."""
    from openconf import ConformerConfig, api

    called = {"low": 0, "hybrid": 0}
    orig_low = api.run_low_flex_generation
    orig_hybrid = api.run_hybrid_generation

    def wrapped_low(*args, **kwargs):
        called["low"] += 1
        return orig_low(*args, **kwargs)

    def wrapped_hybrid(*args, **kwargs):
        called["hybrid"] += 1
        return orig_hybrid(*args, **kwargs)

    monkeypatch.setattr(api, "run_low_flex_generation", wrapped_low)
    monkeypatch.setattr(api, "run_hybrid_generation", wrapped_hybrid)

    config = ConformerConfig(max_out=5, n_seeds=3, n_steps=20, pool_max=20, random_seed=0, collect_stats=True)
    ensemble = api.generate_conformers("CCCC", config=config)

    assert called == {"low": 1, "hybrid": 0}
    assert int(ensemble.generation_stats["n_steps_executed"]) == 0


def test_higher_flex_path_uses_hybrid_generation(monkeypatch):
    """More flexible molecules continue to use the hybrid MC path."""
    from openconf import ConformerConfig, api

    called = {"low": 0, "hybrid": 0}
    orig_low = api.run_low_flex_generation
    orig_hybrid = api.run_hybrid_generation

    def wrapped_low(*args, **kwargs):
        called["low"] += 1
        return orig_low(*args, **kwargs)

    def wrapped_hybrid(*args, **kwargs):
        called["hybrid"] += 1
        return orig_hybrid(*args, **kwargs)

    monkeypatch.setattr(api, "run_low_flex_generation", wrapped_low)
    monkeypatch.setattr(api, "run_hybrid_generation", wrapped_hybrid)

    config = ConformerConfig(max_out=5, n_seeds=3, n_steps=20, pool_max=30, random_seed=0)
    api.generate_conformers("CCCCc1ccccc1", config=config)

    assert called == {"low": 0, "hybrid": 1}


def test_runtime_tuning_policy_helpers():
    """Move scheduling and clash policy helpers preserve current defaults."""
    from openconf.tuning import is_clash_exempt_move, resolve_forced_move, resolve_move_probabilities

    probs = {
        "single_rotor": 0.3,
        "multi_rotor": 0.2,
        "global_shake": 0.1,
        "ring_flip": 0.15,
        "crankshaft": 0.25,
    }

    constrained = resolve_move_probabilities(
        probs,
        constrained=True,
        has_ring_flips=False,
        has_crankshaft=False,
    )
    assert "global_shake" not in constrained
    assert "ring_flip" not in constrained
    assert "crankshaft" not in constrained
    assert constrained["single_rotor"] == pytest.approx(0.8)

    assert resolve_forced_move(20, 20, constrained=False) == "global_shake"
    assert resolve_forced_move(20, 20, constrained=True) is None

    assert is_clash_exempt_move("ring_flip") is True
    assert is_clash_exempt_move("crankshaft") is True
    assert is_clash_exempt_move("single_rotor") is False


def test_generate_candidate_uses_operator_table_and_clash_checker(monkeypatch):
    """Candidate generation dispatches through the operator table and clash helper."""
    from rdkit.Chem import AllChem

    from openconf.config import ConformerConfig
    from openconf.perceive import build_rotor_model, prepare_molecule
    from openconf.pool import ConformerPool
    from openconf.propose.hybrid import HybridProposer
    from openconf.torsionlib import TorsionLibrary

    mol = prepare_molecule(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    rm = build_rotor_model(mol)
    config = ConformerConfig(max_out=5, pool_max=10, random_seed=0)
    proposer = HybridProposer(mol, rm, TorsionLibrary(), config)
    pool = ConformerPool(mol, config)
    pool.insert(mol.GetConformers()[0].GetId(), 0.0, source="seed")

    calls: list[tuple[str, int | str | bool]] = []

    monkeypatch.setattr(proposer, "_select_move_type", lambda step: "single_rotor")
    proposer._move_operators["single_rotor"] = lambda conf_id: calls.append(("operator", conf_id))

    def fake_has_clash(
        conf_id: int,
        move_type: str,
        skip_check: bool = False,
    ) -> bool:
        calls.extend(
            [
                ("move_type", move_type),
                ("skip_check", skip_check),
                ("clash_conf_id", conf_id),
            ]
        )
        return True

    monkeypatch.setattr(proposer._clash_checker, "has_clash", fake_has_clash)

    result = proposer._generate_candidate(pool, 0)

    assert result is None
    assert any(kind == "operator" for kind, _ in calls)
    assert ("move_type", "single_rotor") in calls
    assert ("skip_check", False) in calls


def test_generate_candidate_uses_torsion_multitry(monkeypatch):
    """Torsion candidate generation keeps best clash-scored trial."""
    from rdkit.Chem import AllChem

    from openconf.config import ConformerConfig
    from openconf.perceive import build_rotor_model, prepare_molecule
    from openconf.pool import ConformerPool
    from openconf.propose.hybrid import HybridProposer
    from openconf.propose.stats import new_generation_stats
    from openconf.torsionlib import TorsionLibrary

    mol = prepare_molecule(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    rm = build_rotor_model(mol)
    config = ConformerConfig(max_out=5, pool_max=10, random_seed=0, torsion_multitry_attempts=3)
    stats = new_generation_stats()
    proposer = HybridProposer(mol, rm, TorsionLibrary(), config, stats=stats)
    pool = ConformerPool(mol, config)
    pool.insert(mol.GetConformers()[0].GetId(), 0.0, source="seed")

    scores = iter([5.0, 2.0, 0.0])
    monkeypatch.setattr(proposer, "_select_move_type", lambda step: "single_rotor")
    proposer._move_operators["single_rotor"] = lambda conf_id: None
    monkeypatch.setattr(proposer._clash_checker, "clash_score", lambda conf_id: next(scores))
    monkeypatch.setattr(proposer._clash_checker, "has_clash", lambda conf_id, move_type, skip_check=False: False)

    result = proposer._generate_candidate(pool, 0)

    assert result is not None
    assert stats["n_torsion_multitry_trials"] == 3


def test_generate_conformers_butylbenzene():
    """Test conformer generation for a larger molecule."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(
        max_out=20,
        n_seeds=10,
        n_steps=50,
        pool_max=100,
    )

    # Butylbenzene - a typical druglike fragment
    ensemble = generate_conformers("CCCCc1ccccc1", config=config)

    assert ensemble.n_conformers > 0
    assert ensemble.n_conformers <= 20


def test_sdf_io(tmp_path):
    """Test SDF writing."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(
        max_out=5,
        n_seeds=3,
        n_steps=10,
        pool_max=20,
    )

    ensemble = generate_conformers("CCO", config=config)

    output_file = tmp_path / "test.sdf"
    ensemble.to_sdf(str(output_file))

    assert output_file.exists()

    # Read back
    supplier = Chem.SDMolSupplier(str(output_file))
    count = sum(1 for mol in supplier if mol is not None)

    assert count == ensemble.n_conformers


def test_sdf_roundtrip_preserves_metadata(tmp_path):
    """ConformerEnsemble.from_sdf recovers energies, source, and custom tags."""
    from openconf import ConformerConfig, ConformerEnsemble, generate_conformers

    config = ConformerConfig(max_out=3, n_seeds=3, n_steps=10, pool_max=20, random_seed=1)
    ensemble = generate_conformers("CCCC", config=config)

    # Attach a custom tag so we can verify round-trip fidelity.
    for i, record in enumerate(ensemble.records):
        record.tags["rank"] = str(i)

    out = tmp_path / "rt.sdf"
    ensemble.to_sdf(str(out))
    loaded = ConformerEnsemble.from_sdf(str(out))

    assert loaded.n_conformers == ensemble.n_conformers
    for orig, back in zip(ensemble.records, loaded.records, strict=True):
        orig_energy = orig.energy_kcal if orig.energy_kcal is not None else 0.0
        back_energy = back.energy_kcal if back.energy_kcal is not None else 0.0
        assert abs(orig_energy - back_energy) < 1e-3
        assert back.source == orig.source
        assert back.tags.get("rank") == orig.tags["rank"]


def test_boltzmann_weights_sum_to_one():
    """Boltzmann weights normalize to 1 and favor the lowest-energy conformer."""
    import numpy as np

    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(max_out=5, n_seeds=3, n_steps=20, pool_max=30, random_seed=7)
    ensemble = generate_conformers("CCCCC", config=config)

    weights = ensemble.boltzmann_weights()
    assert weights.shape == (ensemble.n_conformers,)
    assert abs(weights.sum() - 1.0) < 1e-9

    # Lowest-energy conformer should have the largest weight.
    min_idx = int(np.argmin(ensemble.energies))
    assert int(np.argmax(weights)) == min_idx

    # Higher temperature should flatten the distribution (max weight decreases).
    hot = ensemble.boltzmann_weights(temperature=1000.0)
    assert hot.max() <= weights.max() + 1e-9


def test_boltzmann_weights_reject_non_positive_temperature():
    """Boltzmann weights require a positive temperature."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(max_out=5, n_seeds=3, n_steps=20, pool_max=30, random_seed=7)
    ensemble = generate_conformers("CCCCC", config=config)

    with pytest.raises(ValueError, match="temperature"):
        ensemble.boltzmann_weights(temperature=0.0)


def test_rmsd_to_and_pairwise():
    """rmsd_to and pairwise_rmsd are consistent and non-negative."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(max_out=4, n_seeds=3, n_steps=20, pool_max=30, random_seed=3)
    ensemble = generate_conformers("CCCCc1ccccc1", config=config)

    if ensemble.n_conformers < 2:
        pytest.skip("Need at least 2 conformers for RMSD checks")

    rmsds = ensemble.rmsd_to(ref_idx=0)
    assert len(rmsds) == ensemble.n_conformers
    assert rmsds[0] == 0.0
    assert all(r >= 0.0 for r in rmsds)

    matrix = ensemble.pairwise_rmsd()
    n = ensemble.n_conformers
    assert matrix.shape == (n, n)
    # Symmetric, zero diagonal.
    assert (matrix.diagonal() == 0).all()
    assert ((matrix - matrix.T) == 0).all()
    # Row 0 of the matrix should match rmsd_to(ref_idx=0).
    for i, r in enumerate(rmsds):
        assert abs(matrix[0, i] - r) < 1e-6


def test_zero_energy_is_preserved():
    """Zero-valued energies are not treated as missing data."""
    from openconf import ConformerConfig, ConformerEnsemble
    from openconf.pool import ConformerPool, ConformerRecord

    ensemble = ConformerEnsemble(
        mol=Chem.AddHs(Chem.MolFromSmiles("CC")),
        records=[
            ConformerRecord(conf_id=0, energy_kcal=0.0),
            ConformerRecord(conf_id=1, energy_kcal=1.5),
        ],
    )
    assert ensemble.energies == [0.0, 1.5]

    pool = ConformerPool(mol=Chem.AddHs(Chem.MolFromSmiles("CCCC")), config=ConformerConfig(max_out=1, pool_max=2))
    pool.records = {
        0: ConformerRecord(conf_id=0, energy_kcal=0.0),
        1: ConformerRecord(conf_id=1, energy_kcal=1.5),
    }
    assert pool.energies == [0.0, 1.5]
    assert pool.get_parent(strategy="best") == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_out": 0}, "max_out"),
        ({"pool_max": 4, "max_out": 5}, "pool_max"),
        ({"parent_softmax_temperature_kcal": 0.0}, "parent_softmax_temperature_kcal"),
        ({"move_probs": {}}, "move_probs"),
        ({"move_probs": {"unknown": 1.0}}, "unsupported move types"),
        ({"seed_prune_rms_thresh": -0.5}, "seed_prune_rms_thresh"),
        ({"seed_minimization_iters": -1}, "seed_minimization_iters"),
    ],
)
def test_config_validation_rejects_invalid_values(kwargs, match):
    """ConformerConfig rejects invalid runtime values."""
    from openconf import ConformerConfig

    with pytest.raises(ValueError, match=match):
        ConformerConfig(**kwargs)


def test_collect_stats_populates_generation_stats():
    """Enabling stats collection records benchmark timings and counters."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(max_out=5, n_seeds=3, n_steps=20, pool_max=30, random_seed=5, collect_stats=True)
    ensemble = generate_conformers("CCCC", config=config)

    assert ensemble.generation_stats
    assert float(ensemble.generation_stats["total_time_s"]) >= 0.0
    assert float(ensemble.generation_stats["seed_time_s"]) >= 0.0
    assert float(ensemble.generation_stats["parent_selection_time_s"]) >= 0.0
    assert float(ensemble.generation_stats["move_selection_time_s"]) >= 0.0
    assert float(ensemble.generation_stats["move_apply_time_s"]) >= 0.0
    assert float(ensemble.generation_stats["clash_check_time_s"]) >= 0.0
    assert float(ensemble.generation_stats["batch_staging_time_s"]) >= 0.0
    assert float(ensemble.generation_stats["batch_commit_time_s"]) >= 0.0
    assert int(ensemble.generation_stats["n_seed_conformers"]) >= 1
    assert int(ensemble.generation_stats["n_steps_executed"]) >= 0


def test_hybrid_collect_stats_populates_proposal_breakdown():
    """Flexible molecules record nonzero proposal-stage timing components."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(max_out=5, n_seeds=4, n_steps=20, pool_max=30, random_seed=5, collect_stats=True)
    ensemble = generate_conformers("CCCCc1ccccc1", config=config)

    assert float(ensemble.generation_stats["proposal_stage_time_s"]) > 0.0
    assert float(ensemble.generation_stats["parent_selection_time_s"]) > 0.0
    assert float(ensemble.generation_stats["move_selection_time_s"]) > 0.0
    assert float(ensemble.generation_stats["move_apply_time_s"]) > 0.0
    assert float(ensemble.generation_stats["clash_check_time_s"]) > 0.0
    assert float(ensemble.generation_stats["batch_staging_time_s"]) >= 0.0
    assert float(ensemble.generation_stats["batch_commit_time_s"]) >= 0.0
    assert int(ensemble.generation_stats["n_steps_executed"]) > 0


def test_large_flexible_defaults_are_topology_tuned():
    """Large flexible molecules use tuned default-equivalent runtime settings."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(max_out=5, n_steps=20, pool_max=30, random_seed=5, collect_stats=True)
    ensemble = generate_conformers("CCCCCCCCCCCCCCCC", config=config)

    assert int(ensemble.generation_stats["topology_tuned_defaults_applied"]) == 1
    assert int(ensemble.generation_stats["effective_seed_n_per_rotor"]) == 2
    assert float(ensemble.generation_stats["effective_seed_prune_rms_thresh"]) == 1.75
    assert int(ensemble.generation_stats["effective_seed_minimization_iters"]) == 20
    assert float(ensemble.generation_stats["effective_seed_budget_scale"]) == 0.5
    assert int(ensemble.generation_stats["effective_seed_budget_floor"]) == 12
    assert int(ensemble.generation_stats["requested_n_seeds"]) < int(
        ensemble.generation_stats["seed_plan_base_n_seeds"]
    )
    assert int(ensemble.generation_stats["effective_dedupe_period"]) == 100
    assert int(ensemble.generation_stats["effective_minimize_batch_size"]) == 16


def test_macrocycles_do_not_use_large_flexible_tuning():
    """Macrocycles keep the default scheduling knobs."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(max_out=5, n_steps=20, pool_max=30, random_seed=5, collect_stats=True)
    ensemble = generate_conformers("C1CCCCCCCCCCC1", config=config)

    assert int(ensemble.generation_stats["topology_tuned_defaults_applied"]) == 0
    assert int(ensemble.generation_stats["effective_seed_n_per_rotor"]) == 3
    assert float(ensemble.generation_stats["effective_seed_prune_rms_thresh"]) == -1.0
    assert int(ensemble.generation_stats["effective_seed_minimization_iters"]) == 20
    assert float(ensemble.generation_stats["effective_seed_budget_scale"]) == 1.0
    assert int(ensemble.generation_stats["effective_seed_budget_floor"]) == 20
    assert int(ensemble.generation_stats["effective_dedupe_period"]) == 50
    assert int(ensemble.generation_stats["effective_minimize_batch_size"]) == 8


def test_large_flexible_tuning_respects_overrides_and_opt_out():
    """Explicit overrides and opt-out disable the topology tuning path."""
    from openconf import ConformerConfig, generate_conformers

    overridden = ConformerConfig(
        max_out=5,
        n_steps=20,
        pool_max=30,
        random_seed=5,
        collect_stats=True,
        seed_n_per_rotor=5,
        dedupe_period=25,
        minimize_batch_size=4,
        topology_aware_seed_pruning=False,
        topology_aware_seed_budget=False,
    )
    overridden_ensemble = generate_conformers("CCCCCCCCCCCCCCCC", config=overridden)
    assert int(overridden_ensemble.generation_stats["topology_tuned_defaults_applied"]) == 0
    assert int(overridden_ensemble.generation_stats["effective_seed_n_per_rotor"]) == 5
    assert float(overridden_ensemble.generation_stats["effective_seed_prune_rms_thresh"]) == 1.0
    assert int(overridden_ensemble.generation_stats["effective_seed_minimization_iters"]) == 20
    assert float(overridden_ensemble.generation_stats["effective_seed_budget_scale"]) == 1.0
    assert int(overridden_ensemble.generation_stats["effective_seed_budget_floor"]) == 20
    assert int(overridden_ensemble.generation_stats["effective_dedupe_period"]) == 25
    assert int(overridden_ensemble.generation_stats["effective_minimize_batch_size"]) == 4

    opt_out = ConformerConfig(
        max_out=5,
        n_steps=20,
        pool_max=30,
        random_seed=5,
        collect_stats=True,
        auto_tune_large_flexible=False,
    )
    opt_out_ensemble = generate_conformers("CCCCCCCCCCCCCCCC", config=opt_out)
    assert int(opt_out_ensemble.generation_stats["topology_tuned_defaults_applied"]) == 0
    assert int(opt_out_ensemble.generation_stats["effective_seed_n_per_rotor"]) == 3
    assert float(opt_out_ensemble.generation_stats["effective_seed_prune_rms_thresh"]) == 1.0
    assert int(opt_out_ensemble.generation_stats["effective_seed_minimization_iters"]) == 20
    assert float(opt_out_ensemble.generation_stats["effective_seed_budget_scale"]) == 1.0
    assert int(opt_out_ensemble.generation_stats["effective_seed_budget_floor"]) == 20
    assert int(opt_out_ensemble.generation_stats["effective_dedupe_period"]) == 50
    assert int(opt_out_ensemble.generation_stats["effective_minimize_batch_size"]) == 8


def test_seed_experiment_knobs_are_reflected_in_stats():
    """Seed experiment overrides are reflected in the collected stats."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(
        max_out=5,
        n_steps=20,
        pool_max=30,
        random_seed=5,
        collect_stats=True,
        topology_aware_seed_pruning=True,
        topology_aware_seed_budget=True,
        seed_minimization_iters=10,
    )
    ensemble = generate_conformers("CCCCCCCCCCCCCCCC", config=config)

    assert float(ensemble.generation_stats["effective_seed_prune_rms_thresh"]) > 1.0
    assert int(ensemble.generation_stats["effective_seed_minimization_iters"]) == 10
    assert float(ensemble.generation_stats["effective_seed_budget_scale"]) < 1.0


def test_xyz_io(tmp_path):
    """Test XYZ writing."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(
        max_out=5,
        n_seeds=3,
        n_steps=10,
        pool_max=20,
    )

    ensemble = generate_conformers("CCO", config=config)

    output_file = tmp_path / "test.xyz"
    ensemble.to_xyz(str(output_file))

    assert output_file.exists()

    # Check content
    content = output_file.read_text()
    assert "conf_" in content


def test_reproducibility():
    """Test that random seed makes generation reproducible."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(
        max_out=10,
        n_seeds=5,
        n_steps=20,
        pool_max=50,
        random_seed=42,
    )

    ensemble1 = generate_conformers("CCCC", config=config)
    ensemble2 = generate_conformers("CCCC", config=config)

    # Should have same number of conformers
    assert ensemble1.n_conformers == ensemble2.n_conformers

    # Energies should match
    for e1, e2 in zip(ensemble1.energies, ensemble2.energies, strict=True):
        assert abs(e1 - e2) < 0.01


def test_prism_dedupe():
    """Test that PRISM deduplication works."""
    from rdkit.Chem import AllChem

    from openconf.dedupe import prism_dedupe

    mol = Chem.MolFromSmiles("CCCC")
    mol = Chem.AddHs(mol)

    AllChem.EmbedMultipleConfs(mol, numConfs=50)
    for cid in range(mol.GetNumConformers()):
        AllChem.MMFFOptimizeMolecule(mol, confId=cid)

    conf_ids = [c.GetId() for c in mol.GetConformers()]
    keep_ids = prism_dedupe(mol, conf_ids)

    assert len(keep_ids) > 0
    assert len(keep_ids) < len(conf_ids)


# ---------------------------------------------------------------------------
# Torsion library
# ---------------------------------------------------------------------------


def test_torsion_library_loads_from_json():
    """Default library loads 365 CrystalFF rules from the bundled JSON."""
    from openconf import TorsionLibrary

    lib = TorsionLibrary()
    assert len(lib) == 365
    assert len(lib._compiled) == 365


def test_torsion_library_amide_preference():
    """Amide C-N bond should prefer planar conformation (0°), not staggered."""
    from rdkit.Chem import AllChem

    from openconf import TorsionLibrary
    from openconf.perceive import build_rotor_model, prepare_molecule

    lib = TorsionLibrary()
    mol = prepare_molecule(Chem.MolFromSmiles("CC(=O)NC"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    rm = build_rotor_model(mol)

    # Find the C-N rotor (amide bond)
    for rotor in rm.rotors:
        syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in rotor.atom_idxs]
        if set(syms) == {"C", "N"}:
            angles, _ = lib.get_preferred_angles(mol, rotor.dihedral_atoms)
            # Should be planar (0° or 180°), not three staggered conformations
            assert angles != [60.0, 180.0, 300.0], "Amide should not fall back to generic staggered angles"
            assert any(abs(a) < 5 or abs(a - 180) < 5 for a in angles), f"Expected planar preference, got {angles}"
            return

    pytest.skip("No C-N rotor found in test molecule")


def test_torsion_library_roundtrip(tmp_path):
    """to_json / from_json preserves all rules."""
    from openconf import TorsionLibrary

    lib = TorsionLibrary()
    out = tmp_path / "lib.json"
    lib.to_json(out)

    lib2 = TorsionLibrary.from_json(out)
    assert len(lib2) == len(lib)
    assert lib2.rules[0].smarts == lib.rules[0].smarts


def test_torsion_library_custom_rules():
    """A custom single-rule library is respected."""
    from openconf.torsionlib import TorsionLibrary, TorsionRule

    rule = TorsionRule(smarts="[CX4:1][CX4:2][CX4:3][CX4:4]", angles_deg=[60.0, 180.0], weights=[0.3, 0.7])
    lib = TorsionLibrary(rules=[rule])
    assert len(lib) == 1


# ---------------------------------------------------------------------------
# Ring perception
# ---------------------------------------------------------------------------


def test_ring_flips_cyclohexane():
    """Cyclohexane should have exactly one flippable ring."""
    from openconf.perceive import build_rotor_model, prepare_molecule

    mol = prepare_molecule(Chem.MolFromSmiles("C1CCCCC1"))
    rm = build_rotor_model(mol)
    assert len(rm.ring_flips) == 1
    assert rm.ring_flips[0].ring_size == 6


def test_ring_flips_benzene_excluded():
    """Aromatic rings must not appear as ring flips."""
    from openconf.perceive import build_rotor_model, prepare_molecule

    mol = prepare_molecule(Chem.MolFromSmiles("c1ccccc1"))
    rm = build_rotor_model(mol)
    assert len(rm.ring_flips) == 0


def test_ring_flips_mixed():
    """Fused aromatic/saturated ring systems should not receive independent flips."""
    from openconf.perceive import build_rotor_model, prepare_molecule

    # tetralin: benzene fused to cyclohexane
    mol = prepare_molecule(Chem.MolFromSmiles("C1CCc2ccccc2C1"))
    rm = build_rotor_model(mol)
    assert len(rm.ring_flips) == 0


def test_ring_info_macrocycle():
    """Macrocycle flag is set for large rings."""
    from openconf.perceive import build_rotor_model, prepare_molecule

    mol = prepare_molecule(Chem.MolFromSmiles("C1CCCCCCCCCCC1"))  # cyclododecane
    rm = build_rotor_model(mol)
    assert rm.ring_info["has_macrocycle"] is True
    assert rm.ring_info["max_ring_size"] == 12


def test_ring_info_no_macrocycle():
    """Ordinary rings do not trigger the macrocycle flag."""
    from openconf.perceive import build_rotor_model, prepare_molecule

    mol = prepare_molecule(Chem.MolFromSmiles("c1ccccc1"))
    rm = build_rotor_model(mol)
    assert rm.ring_info["has_macrocycle"] is False


# ---------------------------------------------------------------------------
# Adaptive seed count
# ---------------------------------------------------------------------------


def test_adaptive_seeds_scales_with_rotors():
    """More rotatable bonds → more seeds."""
    from openconf.perceive import build_rotor_model, prepare_molecule
    from openconf.propose.hybrid import _compute_n_seeds

    small = prepare_molecule(Chem.MolFromSmiles("CC"))
    large = prepare_molecule(Chem.MolFromSmiles("CCCCCCCCCCCC"))
    seeds_small = _compute_n_seeds(build_rotor_model(small))
    seeds_large = _compute_n_seeds(build_rotor_model(large))
    assert seeds_large > seeds_small


def test_adaptive_seeds_macrocycle_bonus():
    """Macrocycle ring adds substantial extra seeds beyond a linear chain."""
    from openconf.perceive import build_rotor_model, prepare_molecule
    from openconf.propose.hybrid import _compute_n_seeds

    chain = prepare_molecule(Chem.MolFromSmiles("CCCCCCCCCCCC"))  # 12 carbons, no ring
    macro = prepare_molecule(Chem.MolFromSmiles("C1CCCCCCCCCCC1"))  # cyclododecane
    seeds_chain = _compute_n_seeds(build_rotor_model(chain))
    seeds_macro = _compute_n_seeds(build_rotor_model(macro))
    assert seeds_macro > seeds_chain


def test_adaptive_seeds_ring_flip_bonus():
    """Non-aromatic rings add seeds beyond a rigid molecule."""
    from openconf.perceive import build_rotor_model, prepare_molecule
    from openconf.propose.hybrid import _compute_n_seeds

    rigid = prepare_molecule(Chem.MolFromSmiles("c1ccccc1"))  # benzene, 0 flips
    cyclo = prepare_molecule(Chem.MolFromSmiles("C1CCCCC1"))  # cyclohexane, 1 flip
    seeds_rigid = _compute_n_seeds(build_rotor_model(rigid))
    seeds_cyclo = _compute_n_seeds(build_rotor_model(cyclo))
    assert seeds_cyclo > seeds_rigid


def test_n_seeds_none_runs():
    """n_seeds=None (default) triggers auto-computation without error."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(max_out=5, n_steps=20, pool_max=50, collect_stats=True)
    assert config.n_seeds is None
    ens = generate_conformers("CCCC", config=config)
    assert ens.n_conformers > 0
    assert ens.generation_stats["seed_plan_reason"] == "low_flex_acyclic"
    assert int(ens.generation_stats["requested_n_seeds"]) < int(ens.generation_stats["seed_plan_base_n_seeds"])


def test_n_seeds_explicit_overrides_auto():
    """An explicit n_seeds value is respected as-is."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(n_seeds=7, max_out=5, n_steps=10, pool_max=30, random_seed=1, collect_stats=True)
    ens = generate_conformers("CCCC", config=config)
    assert ens.n_conformers > 0
    assert ens.generation_stats["seed_plan_reason"] == "explicit"
    assert int(ens.generation_stats["requested_n_seeds"]) == 7


def test_resolve_seed_plan_preserves_macrocycle_budget():
    """Macrocycles keep dense seed budgets for ring-pucker discovery."""
    from openconf import ConformerConfig
    from openconf.perceive import build_rotor_model, prepare_molecule
    from openconf.propose.hybrid import resolve_seed_plan

    mol = prepare_molecule(Chem.MolFromSmiles("C1CCCCCCCCCCC1"))
    rotor_model = build_rotor_model(mol)
    seed_plan = resolve_seed_plan(mol, rotor_model, ConformerConfig(max_out=5))

    assert seed_plan.reason == "auto"
    assert seed_plan.n_seeds == seed_plan.base_n_seeds
    assert seed_plan.n_seeds > 100


# ---------------------------------------------------------------------------
# Ring flip move
# ---------------------------------------------------------------------------


def test_ring_flip_changes_coords():
    """_apply_ring_flip_move should change ring atom positions."""
    import numpy as np
    from rdkit.Chem import AllChem

    from openconf.config import ConformerConfig
    from openconf.perceive import build_rotor_model, prepare_molecule
    from openconf.propose.hybrid import HybridProposer, _copy_conformer
    from openconf.torsionlib import TorsionLibrary

    mol = prepare_molecule(Chem.MolFromSmiles("C1CCCCC1"))
    AllChem.EmbedMolecule(mol, randomSeed=42)

    rm = build_rotor_model(mol)
    assert rm.ring_flips, "Need at least one ring flip for this test"

    config = ConformerConfig(random_seed=0)
    proposer = HybridProposer(mol, rm, TorsionLibrary(), config)

    orig_id = mol.GetConformers()[0].GetId()
    new_id = _copy_conformer(mol, orig_id)

    proposer._apply_ring_flip_move(new_id)

    orig_pos = np.array([list(mol.GetConformer(orig_id).GetAtomPosition(i)) for i in rm.ring_flips[0].ring_atoms])
    new_pos = np.array([list(mol.GetConformer(new_id).GetAtomPosition(i)) for i in rm.ring_flips[0].ring_atoms])

    # At least some atoms should have moved
    assert not np.allclose(orig_pos, new_pos, atol=0.01), "Ring flip did not change any coordinates"


def test_ring_flip_moves_attached_subtrees():
    """Ring flip should reflect substituents with their host ring atoms."""
    import numpy as np
    from rdkit.Chem import AllChem

    from openconf.config import ConformerConfig
    from openconf.perceive import build_rotor_model, prepare_molecule
    from openconf.propose.hybrid import HybridProposer, _copy_conformer
    from openconf.torsionlib import TorsionLibrary

    mol = prepare_molecule(Chem.MolFromSmiles("CC1CCCCC1"))
    AllChem.EmbedMolecule(mol, randomSeed=42)

    rm = build_rotor_model(mol)
    assert rm.ring_flips, "Need at least one ring flip for this test"

    proposer = HybridProposer(mol, rm, TorsionLibrary(), ConformerConfig(random_seed=0))
    orig_id = mol.GetConformers()[0].GetId()
    new_id = _copy_conformer(mol, orig_id)
    methyl_idx = 0

    before = np.array(mol.GetConformer(orig_id).GetAtomPosition(methyl_idx))
    proposer._apply_ring_flip_move(new_id)
    after = np.array(mol.GetConformer(new_id).GetAtomPosition(methyl_idx))

    assert not np.allclose(before, after, atol=0.01), "Ring flip did not move attached substituent"


def test_fused_rings_do_not_register_ring_flips():
    """Fused ring systems should not receive independent plane-reflection flips."""
    from rdkit.Chem import AllChem

    from openconf.perceive import build_rotor_model, prepare_molecule

    mol = prepare_molecule(Chem.MolFromSmiles("C1CCC2CCCCC2C1"))
    AllChem.EmbedMolecule(mol, randomSeed=42)

    rm = build_rotor_model(mol)

    assert rm.ring_flips == []


def test_ring_flip_in_generation():
    """Generation on a ring-containing molecule completes and returns conformers."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(max_out=10, n_steps=50, pool_max=100, random_seed=42)
    ens = generate_conformers("C1CCCCC1", config=config)
    assert ens.n_conformers > 0


# ---------------------------------------------------------------------------
# Metal-ligand torsions
# ---------------------------------------------------------------------------


def test_metal_ligand_rotors_included():
    """Metal-ligand bonds should appear as metal_ligand rotors, not be skipped."""
    from openconf.perceive import build_rotor_model, prepare_molecule

    # Dimethylzinc: two Zn-C bonds that should be sampled as metal_ligand rotors.
    mol = prepare_molecule(Chem.MolFromSmiles("[Zn](C)C"), add_hs=True)
    rm = build_rotor_model(mol)
    metal_rotors = [r for r in rm.rotors if r.rotor_type == "metal_ligand"]
    assert len(metal_rotors) == 2


def test_metal_ligand_flat_angles():
    """metal_ligand rotors should get 12 equally-spaced angles, not torsion library angles."""
    import numpy as np
    from rdkit.Chem import AllChem

    from openconf.config import ConformerConfig
    from openconf.perceive import build_rotor_model, prepare_molecule
    from openconf.propose.hybrid import HybridProposer
    from openconf.torsionlib import TorsionLibrary

    mol = prepare_molecule(Chem.MolFromSmiles("[Zn](C)C"), add_hs=True)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    rm = build_rotor_model(mol)

    proposer = HybridProposer(mol, rm, TorsionLibrary(), ConformerConfig(random_seed=0))

    metal_indices = [i for i, r in enumerate(rm.rotors) if r.rotor_type == "metal_ligand"]
    for idx in metal_indices:
        angles_arr, weights_arr = proposer._rotor_angles[idx]
        assert len(angles_arr) == 12
        assert np.allclose(weights_arr, 1.0 / 12.0)
        # Equally spaced across [0, 360)
        assert np.allclose(np.diff(angles_arr), 30.0)
