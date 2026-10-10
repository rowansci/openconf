"""Tests for excluded volumes during pose-constrained generation."""

from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from openconf import ConformerConfig, ExcludedVolumeSpec, generate_conformers_from_pose
from openconf.exceptions import OpenConfValueError
from openconf.excluded_volume import ExcludedVolumeChecker
from openconf.perceive import build_rotor_model, prepare_molecule
from openconf.pool import ConformerPool
from openconf.propose.hybrid import HybridProposer
from openconf.torsionlib import TorsionLibrary


def _single_carbon(*positions: tuple[float, float, float]) -> Chem.Mol:
    """Return single-carbon molecule with conformers at given positions."""
    editable = Chem.RWMol()
    editable.AddAtom(Chem.Atom(6))
    mol = editable.GetMol()
    Chem.SanitizeMol(mol)
    for position in positions:
        conformer = Chem.Conformer(1)
        conformer.SetAtomPosition(0, position)
        mol.AddConformer(conformer, assignId=True)
    return mol


def _embedded_butane() -> Chem.Mol:
    """Return hydrogenated butane with deterministic 3D coordinates."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return mol


@pytest.mark.parametrize(
    ("distance_fraction", "expected_score"),
    [(0.5, 0.75), (1.0, 0.0)],
)
def test_external_score_matches_internal_overlap_equation(
    distance_fraction: float,
    expected_score: float,
) -> None:
    """External score uses internal normalized squared-overlap equation."""
    carbon_radius = Chem.GetPeriodicTable().GetRcovalent(6)
    mol = _single_carbon((distance_fraction * carbon_radius, 0.0, 0.0))
    checker = ExcludedVolumeChecker(
        mol,
        ExcludedVolumeSpec(positions=((0.0, 0.0, 0.0),), radii=(0.0,), radius_scale=1.0),
    )

    assert checker.clash_score(0) == pytest.approx(expected_score)


def test_excluded_volume_spec_validation() -> None:
    """External coordinates, atom types, and radius settings validate eagerly."""
    default_spec = ExcludedVolumeSpec(positions=((0.0, 0.0, 0.0),))
    assert default_spec.radius_scale == 1.0
    carbon_spec = ExcludedVolumeSpec(positions=((0.0, 0.0, 0.0),), atomic_numbers=(6,))
    assert carbon_spec.resolved_radii() == pytest.approx((Chem.GetPeriodicTable().GetRcovalent(6),))
    with pytest.raises(OpenConfValueError, match="shape"):
        ExcludedVolumeSpec(positions=np.asarray(((0.0, 0.0),)))
    with pytest.raises(OpenConfValueError, match="either"):
        ExcludedVolumeSpec(positions=((0.0, 0.0, 0.0),), atomic_numbers=(6,), radii=(1.7,))
    with pytest.raises(OpenConfValueError, match="integers"):
        ExcludedVolumeSpec(positions=((0.0, 0.0, 0.0),), atomic_numbers=(6.5,))  # type: ignore[arg-type]
    with pytest.raises(OpenConfValueError, match="greater than zero"):
        ExcludedVolumeSpec(positions=((0.0, 0.0, 0.0),), radius_scale=0.0)


def test_torsion_multitry_filters_external_then_scores_internal_overlap() -> None:
    """Multi-try selects lowest internal score among externally clear trials."""
    mol = prepare_molecule(_embedded_butane(), add_hs=False)
    config = ConformerConfig(torsion_multitry_attempts=4, random_seed=3)
    proposer = HybridProposer(mol, build_rotor_model(mol), TorsionLibrary(), config)
    attempted_ids: list[int] = []
    proposer._move_operators["single_rotor"] = attempted_ids.append

    internal_scores = iter((0.10, 0.00, 0.00))
    external_scores = iter((0.00, 0.20, 0.00))
    proposer._clash_checker = SimpleNamespace(clash_score=lambda _conf_id: next(internal_scores))
    proposer._excluded_volume_checker = SimpleNamespace(clash_score=lambda _conf_id: next(external_scores))

    parent_id = mol.GetConformers()[0].GetId()
    selected_id = proposer._generate_multitry_candidate(parent_id, "single_rotor")

    assert selected_id == attempted_ids[2]
    assert len(attempted_ids) == 3
    assert mol.GetNumConformers() == 2


def test_torsion_multitry_returns_none_when_every_trial_clashes_externally() -> None:
    """Multi-try rejects a move when no trial clears external volumes."""
    mol = prepare_molecule(_embedded_butane(), add_hs=False)
    config = ConformerConfig(torsion_multitry_attempts=3, random_seed=3)
    proposer = HybridProposer(mol, build_rotor_model(mol), TorsionLibrary(), config)
    attempted_ids: list[int] = []
    proposer._move_operators["single_rotor"] = attempted_ids.append
    proposer._excluded_volume_checker = SimpleNamespace(clash_score=lambda _conf_id: 1.0)

    parent_id = mol.GetConformers()[0].GetId()
    selected_id = proposer._generate_multitry_candidate(parent_id, "single_rotor")

    assert selected_id is None
    assert len(attempted_ids) == 3
    assert mol.GetNumConformers() == 1


def test_candidate_is_rechecked_after_minimization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidate moving into an external volume during minimization is rejected."""
    mol = prepare_molecule(_embedded_butane(), add_hs=False)
    parent_id = mol.GetConformers()[0].GetId()
    candidate = Chem.Conformer(mol.GetConformer(parent_id))
    candidate_id = mol.AddConformer(candidate, assignId=True)
    config = ConformerConfig(random_seed=3)
    proposer = HybridProposer(mol, build_rotor_model(mol), TorsionLibrary(), config)
    proposer._excluded_volume_checker = SimpleNamespace(clash_score=lambda _conf_id: 1.0)

    monkeypatch.setattr(proposer, "_generate_candidate", lambda _pool, _step: (candidate_id, "single_rotor"))
    monkeypatch.setattr(proposer, "fast_minimizer", SimpleNamespace(minimize=lambda _mol, _conf_id: 0.0))

    assert proposer.propose(ConformerPool(mol, config), 0) is None
    assert candidate_id not in {conformer.GetId() for conformer in mol.GetConformers()}


def test_final_filter_keeps_only_clear_conformers() -> None:
    """Final filtering removes every externally clashing conformer."""
    carbon_radius = Chem.GetPeriodicTable().GetRcovalent(6)
    mol = _single_carbon(
        (0.5 * carbon_radius, 0.0, 0.0),
        (carbon_radius, 0.0, 0.0),
        (0.8 * carbon_radius, 0.0, 0.0),
    )
    config = ConformerConfig(
        max_out=3,
        pool_max=3,
        energy_window_kcal=1000.0,
        dedupe_rmsd_threshold=None,
    )
    excluded_volume = ExcludedVolumeSpec(
        positions=((0.0, 0.0, 0.0),),
        radii=(0.0,),
        radius_scale=1.0,
    )
    proposer = HybridProposer(
        mol,
        build_rotor_model(mol),
        TorsionLibrary(),
        config,
        excluded_volume=excluded_volume,
    )
    conf_ids = [conformer.GetId() for conformer in mol.GetConformers()]
    selected, energies = proposer._filter_external_finalists(conf_ids, [-100.0, 10.0, -50.0])

    assert selected == [conf_ids[1]]
    assert energies == [10.0]


def test_generation_rejects_constrained_atom_clash_upfront() -> None:
    """Generation fails immediately when a fixed atom overlaps the environment."""
    mol = _embedded_butane()
    position = mol.GetConformer().GetAtomPosition(0)
    excluded_volume = ExcludedVolumeSpec(
        positions=((position.x, position.y, position.z),),
        radii=(0.0,),
    )
    config = ConformerConfig(max_out=3, pool_max=20, n_steps=8, random_seed=11)

    with pytest.raises(OpenConfValueError, match="Constrained atoms overlap"):
        generate_conformers_from_pose(mol, [0, 1], config=config, excluded_volume=excluded_volume)


def test_excluded_volume_requires_pose_anchor() -> None:
    """External coordinates require constrained atoms to preserve their frame."""
    mol = _embedded_butane()
    excluded_volume = ExcludedVolumeSpec(positions=((100.0, 100.0, 100.0),))

    with pytest.raises(OpenConfValueError, match="at least one constrained atom"):
        generate_conformers_from_pose(mol, [], excluded_volume=excluded_volume)
