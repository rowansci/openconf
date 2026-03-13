"""Basic tests for openconf."""

import pytest
from rdkit import Chem


def test_import():
    """Test that the package can be imported."""
    import openconf

    assert hasattr(openconf, "generate_conformers")
    assert hasattr(openconf, "ConformerConfig")
    assert hasattr(openconf, "ConformerEnsemble")


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

    lib = TorsionLibrary()
    assert len(lib) > 0

    # Check that patterns compile
    for pattern, _rule, _pos2, _pos3 in lib._compiled:
        assert pattern is not None


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

    from openconf.config import PrismConfig
    from openconf.dedupe import prism_dedupe

    mol = Chem.MolFromSmiles("CCCC")
    mol = Chem.AddHs(mol)

    # Generate some conformers
    AllChem.EmbedMultipleConfs(mol, numConfs=50)

    # Minimize all
    for cid in range(mol.GetNumConformers()):
        AllChem.MMFFOptimizeMolecule(mol, confId=cid)

    conf_ids = [c.GetId() for c in mol.GetConformers()]

    config = PrismConfig()

    keep_ids = prism_dedupe(mol, conf_ids, config)

    # Should significantly reduce the number of conformers
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
    """A molecule with both aromatic and non-aromatic rings: only the non-aromatic one flips."""
    from openconf.perceive import build_rotor_model, prepare_molecule

    # tetralin: benzene fused to cyclohexane
    mol = prepare_molecule(Chem.MolFromSmiles("C1CCc2ccccc2C1"))
    rm = build_rotor_model(mol)
    # Only the saturated 6-membered ring should be flippable
    assert len(rm.ring_flips) == 1
    assert rm.ring_flips[0].ring_size == 6


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

    config = ConformerConfig(max_out=5, n_steps=20, pool_max=50)
    assert config.n_seeds is None
    ens = generate_conformers("CCCC", config=config)
    assert ens.n_conformers > 0


def test_n_seeds_explicit_overrides_auto():
    """An explicit n_seeds value is respected as-is."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(n_seeds=7, max_out=5, n_steps=10, pool_max=30, random_seed=1)
    ens = generate_conformers("CCCC", config=config)
    assert ens.n_conformers > 0


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


def test_ring_flip_in_generation():
    """Generation on a ring-containing molecule completes and returns conformers."""
    from openconf import ConformerConfig, generate_conformers

    config = ConformerConfig(max_out=10, n_steps=50, pool_max=100, random_seed=42)
    ens = generate_conformers("C1CCCCC1", config=config)
    assert ens.n_conformers > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
