"""Relaxation and minimization backends for conformer generation."""

from dataclasses import dataclass, field
from typing import Protocol

from rdkit import Chem
from rdkit.Chem import AllChem


class Minimizer(Protocol):
    """Protocol for conformer minimizers."""

    max_iters: int

    def minimize(self, mol: Chem.Mol, conf_id: int) -> float:
        """Minimize a conformer in place.

        Args:
            mol: RDKit molecule containing the conformer.
            conf_id: Conformer ID to minimize.

        Returns:
            Energy in kcal/mol after minimization.
        """
        ...


@dataclass
class RDKitMMFFMinimizer:
    """RDKit MMFF94 force field minimizer.

    Attributes:
        max_iters: Maximum iterations for minimization.
        force_tol: Force convergence tolerance.
        energy_tol: Energy convergence tolerance.
        variant: MMFF variant ("MMFF94" or "MMFF94s").
        dielectric: Dielectric constant for electrostatics. Gas phase is 1.0;
            higher values (4-10) reduce over-strong intramolecular electrostatics
            and are more appropriate for condensed-phase conformer generation.
    """

    max_iters: int = 500
    force_tol: float = 1e-4
    energy_tol: float = 1e-6
    variant: str = "MMFF94s"
    dielectric: float = 4.0
    _mmff_props: object = field(default=None, init=False, repr=False)

    def prepare(self, mol: Chem.Mol) -> None:
        """Cache MMFF properties for the molecule.

        Call once per molecule before minimizing.

        Args:
            mol: RDKit molecule to prepare.
        """
        self._mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=self.variant)
        if self._mmff_props is not None:
            self._mmff_props.SetMMFFDielectricConstant(self.dielectric)

    def minimize(self, mol: Chem.Mol, conf_id: int) -> float:
        """Minimize conformer in place and return energy in kcal/mol.

        Args:
            mol: RDKit molecule containing the conformer.
            conf_id: Conformer ID to minimize.

        Returns:
            Energy in kcal/mol after minimization.
        """
        try:
            if self._mmff_props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, self._mmff_props, confId=int(conf_id))
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(conf_id))
            if ff is None:
                return float("inf")
            ff.Minimize(maxIts=int(self.max_iters))
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
        mol: RDKit molecule containing the conformers.
        mmff_props: Pre-prepared MMFFMoleculeProperties with dielectric already set.
        conf_ids: Conformer IDs to minimize.
        max_iters: Maximum minimization iterations.
        num_threads: C++ threads for MMFFOptimizeMoleculeConfs. 0 = all available.
        variant: MMFF variant string passed to MMFFOptimizeMoleculeConfs.

    Returns:
        Energies in kcal/mol, aligned to conf_ids.
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


def get_minimizer(name: str = "rdkit_mmff", **kwargs) -> Minimizer:
    """Get a minimizer by name.

    Args:
        name: Minimizer name ("rdkit_mmff" or future).
        **kwargs: Additional arguments for the minimizer.

    Returns:
        Minimizer instance.

    Raises:
        ValueError: If unknown minimizer name.
    """
    if name == "rdkit_mmff":
        return RDKitMMFFMinimizer(**kwargs)
    else:
        raise ValueError(f"Unknown minimizer: {name}")
