"""Relaxation and minimization backends for conformer generation."""

from dataclasses import dataclass, field
from typing import Protocol

from rdkit import Chem
from rdkit.Chem import AllChem

from .constraints import ConstraintModel, add_constraints_to_force_field
from .exceptions import OpenConfValueError


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
        constraint_model: geometry constraints to apply during minimization.
    """

    max_iters: int = 500
    force_tol: float = 1e-4
    energy_tol: float = 1e-6
    variant: str = "MMFF94s"
    dielectric: float = 4.0
    metal_atom_indices: frozenset[int] = field(default_factory=frozenset)
    constraint_model: ConstraintModel = field(default_factory=ConstraintModel.empty)
    _mmff_props: object = field(default=None, init=False, repr=False)
    _prepared_constraints: ConstraintModel = field(default_factory=ConstraintModel.empty, init=False, repr=False)

    def prepare(self, mol: Chem.Mol) -> None:
        """Cache MMFF properties for the molecule.

        Call once per molecule before minimizing.

        Args:
            mol: molecule to prepare
        """
        self._mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=self.variant)
        if self._mmff_props is not None:
            self._mmff_props.SetMMFFDielectricConstant(self.dielectric)
        metal_constraints = ConstraintModel.from_metal_shell(mol, self.metal_atom_indices)
        self._prepared_constraints = metal_constraints.combine(self.constraint_model)

    def _reset_constrained_positions(self, mol: Chem.Mol, conf_id: int) -> None:
        """Snap position-constrained atoms back to reference coordinates.

        Args:
            mol: molecule containing conformer
            conf_id: conformer ID to update
        """
        self._prepared_constraints.reset_positions(mol, conf_id)

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
                family = "MMFF"
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(conf_id))
                family = "UFF"
            if ff is None:
                return float("inf")
            add_constraints_to_force_field(ff, self._prepared_constraints, family)
            ff.Minimize(maxIts=int(self.max_iters))
            self._reset_constrained_positions(mol, conf_id)
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


def get_minimizer(
    name: str = "rdkit_mmff",
    metal_atom_indices: frozenset[int] = frozenset(),
    constraint_model: ConstraintModel | None = None,
    **kwargs,
) -> Minimizer:
    """Get a minimizer by name.

    Args:
        name: minimizer name; only `"rdkit_mmff"` is currently supported
        metal_atom_indices: metal atoms to exclude from MMFF typing
        constraint_model: geometry constraints to apply during minimization
        **kwargs: additional arguments for minimizer

    Returns:
        Minimizer instance

    Raises:
        OpenConfValueError: unknown minimizer name
    """
    if name == "rdkit_mmff":
        return RDKitMMFFMinimizer(
            metal_atom_indices=metal_atom_indices,
            constraint_model=constraint_model or ConstraintModel.empty(),
            **kwargs,
        )
    raise OpenConfValueError(f"Unknown minimizer: {name}")
