"""Relaxation and minimization backends for conformer generation."""

from dataclasses import dataclass, field
from typing import Protocol

from rdkit import Chem
from rdkit.Chem import AllChem

_METAL_POSITION_FORCE_CONSTANT = 1e4
_METAL_LIGAND_DISTANCE_TOLERANCE = 0.05
_METAL_LIGAND_DISTANCE_FORCE_CONSTANT = 100000.0


def _metal_ligand_reference_distances(
    mol: Chem.Mol,
    metal_atom_indices: frozenset[int],
) -> dict[tuple[int, int], float]:
    """Measure reference metal-ligand distances from first conformer.

    Args:
        mol: molecule with reference conformer
        metal_atom_indices: metal-center atom indices

    Returns:
        Reference distances keyed by metal-neighbor atom index pairs
    """
    if not metal_atom_indices or mol.GetNumConformers() == 0:
        return {}

    conf = mol.GetConformer(mol.GetConformers()[0].GetId())
    distances: dict[tuple[int, int], float] = {}
    for m_idx in metal_atom_indices:
        metal_pos = conf.GetAtomPosition(int(m_idx))
        for nb in mol.GetAtomWithIdx(int(m_idx)).GetNeighbors():
            nb_idx = int(nb.GetIdx())
            nb_pos = conf.GetAtomPosition(nb_idx)
            distance = float(metal_pos.Distance(nb_pos))
            if distance > 0.0:
                distances[(int(m_idx), nb_idx)] = distance
    return distances


class Minimizer(Protocol):
    """Protocol for conformer minimizers."""

    max_iters: int

    def minimize(self, mol: Chem.Mol, conf_id: int) -> float:
        """Minimize a conformer in place.

        Args:
            mol: molecule containing conformer
            conf_id: conformer ID to minimize

        Returns:
            Energy in kcal/mol after minimization
        """
        ...


@dataclass
class RDKitMMFFMinimizer:
    """RDKit MMFF94 force field minimizer.

    Attributes:
        max_iters: maximum iterations for minimization
        force_tol: force convergence tolerance
        energy_tol: energy convergence tolerance
        variant: MMFF variant; `"MMFF94"` or `"MMFF94s"`
        dielectric: dielectric constant for electrostatics. Gas phase is 1.0;
            higher values (4-10) reduce over-strong intramolecular electrostatics
            and are more appropriate for condensed-phase conformer generation.
        metal_atom_indices: metal-center atom indices. When MMFF is
            unavailable these are pinned via UFFAddPositionConstraint so that
            untyped metals (lanthanides, actinides, …) do not drift freely
            during UFF minimization.
        _metal_ref_positions: reference coordinates for metal centers when
            input molecule already has coordinates.
        _metal_ligand_ref_distances: reference distances from metal centers to
            directly connected ligating atoms when input molecule has coordinates.
    """

    max_iters: int = 500
    force_tol: float = 1e-4
    energy_tol: float = 1e-6
    variant: str = "MMFF94s"
    dielectric: float = 4.0
    metal_atom_indices: frozenset[int] = field(default_factory=frozenset)
    _mmff_props: object = field(default=None, init=False, repr=False)
    _metal_ref_positions: dict[int, tuple[float, float, float]] = field(default_factory=dict, init=False, repr=False)
    _metal_ligand_ref_distances: dict[tuple[int, int], float] = field(default_factory=dict, init=False, repr=False)

    def prepare(self, mol: Chem.Mol) -> None:
        """Cache MMFF properties for the molecule.

        Call once per molecule before minimizing.

        Args:
            mol: molecule to prepare
        """
        self._mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=self.variant)
        if self._mmff_props is not None:
            self._mmff_props.SetMMFFDielectricConstant(self.dielectric)
        self._metal_ref_positions = {}
        if self.metal_atom_indices and mol.GetNumConformers() > 0:
            conf = mol.GetConformer(mol.GetConformers()[0].GetId())
            for idx in self.metal_atom_indices:
                pos = conf.GetAtomPosition(int(idx))
                self._metal_ref_positions[int(idx)] = (float(pos.x), float(pos.y), float(pos.z))
        self._metal_ligand_ref_distances = _metal_ligand_reference_distances(mol, self.metal_atom_indices)

    def _add_metal_uff_constraints(self, ff) -> None:
        """Add metal position and reference-shell distance constraints to UFF force field."""
        for m_idx in self.metal_atom_indices:
            ff.UFFAddPositionConstraint(int(m_idx), 0.0, _METAL_POSITION_FORCE_CONSTANT)
        for (m_idx, nb_idx), distance in self._metal_ligand_ref_distances.items():
            ff.UFFAddDistanceConstraint(
                int(m_idx),
                int(nb_idx),
                False,
                max(0.0, distance - _METAL_LIGAND_DISTANCE_TOLERANCE),
                distance + _METAL_LIGAND_DISTANCE_TOLERANCE,
                _METAL_LIGAND_DISTANCE_FORCE_CONSTANT,
            )

    def _reset_metal_positions(self, mol: Chem.Mol, conf_id: int) -> None:
        """Snap metal centers back to reference coordinates when available.

        Args:
            mol: molecule containing conformer
            conf_id: conformer ID to update
        """
        if not self._metal_ref_positions:
            return
        conf = mol.GetConformer(int(conf_id))
        for idx, pos in self._metal_ref_positions.items():
            conf.SetAtomPosition(idx, pos)

    def minimize(self, mol: Chem.Mol, conf_id: int) -> float:
        """Minimize conformer in place and return energy in kcal/mol.

        Args:
            mol: molecule containing conformer
            conf_id: conformer ID to minimize

        Returns:
            Energy in kcal/mol after minimization
        """
        try:
            if self._mmff_props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, self._mmff_props, confId=int(conf_id))
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(conf_id))
            if ff is None:
                return float("inf")
            if self._mmff_props is None and self.metal_atom_indices:
                self._add_metal_uff_constraints(ff)
            ff.Minimize(maxIts=int(self.max_iters))
            self._reset_metal_positions(mol, conf_id)
            return float(ff.CalcEnergy())
        except (ValueError, RuntimeError):
            return float("inf")


def minimize_confs_mmff(
    mol: Chem.Mol,
    mmff_props,
    conf_ids: list[int],
    max_iters: int,
    num_threads: int = 0,
    variant: str = "MMFF94s",
) -> list[float]:
    """Minimize conformers and return energies evaluated with mmff_props.

    Uses MMFFOptimizeMoleculeConfs for parallel C++ geometry minimization, then
    re-evaluates each conformer's energy with the pre-prepared mmff_props (which
    carries the custom dielectric). This recovers the C++ parallelism that would
    be lost by a per-conformer Python loop, while still reporting energies that
    reflect the requested dielectric constant.

    Note: geometries are at the default-dielectric (ε=1) MMFF minimum; the custom
    dielectric is applied only to the energy evaluation. For the coarse fast-
    minimization passes during MCMM sampling this is an acceptable approximation.

    Args:
        mol: molecule containing conformers
        mmff_props: pre-prepared MMFFMoleculeProperties with dielectric already set
        conf_ids: conformer IDs to minimize
        max_iters: maximum minimization iterations
        num_threads: C++ threads for MMFFOptimizeMoleculeConfs; 0 means all available
        variant: MMFF variant string passed to MMFFOptimizeMoleculeConfs

    Returns:
        Energies in kcal/mol aligned to `conf_ids`
    """
    if not conf_ids:
        return []

    AllChem.MMFFOptimizeMoleculeConfs(
        mol, numThreads=int(num_threads or 0), maxIters=int(max_iters), mmffVariant=variant
    )

    return [
        float(ff.CalcEnergy())
        if (ff := AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=cid)) is not None
        else float("inf")
        for cid in conf_ids
    ]


def get_minimizer(name: str = "rdkit_mmff", metal_atom_indices: frozenset[int] = frozenset(), **kwargs) -> Minimizer:
    """Get a minimizer by name.

    Args:
        name: minimizer name; only `"rdkit_mmff"` is currently supported
        metal_atom_indices: metal atoms to exclude from MMFF typing
        **kwargs: additional arguments for minimizer

    Returns:
        Minimizer instance

    Raises:
        ValueError: unknown minimizer name
    """
    if name == "rdkit_mmff":
        return RDKitMMFFMinimizer(metal_atom_indices=metal_atom_indices, **kwargs)
    raise ValueError(f"Unknown minimizer: {name}")
