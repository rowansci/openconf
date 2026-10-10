"""Excluded volumes for pose-constrained conformer generation."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from .exceptions import OpenConfValueError


def sum_clash_scores(
    dist2: NDArray[np.float64],
    threshold2: NDArray[np.float64] | float,
) -> float:
    """Return clash scores summed over all atom pairs closer than allowed.

    Args:
        dist2: squared pair distances
        threshold2: squared minimum distances broadcastable to `dist2`

    Returns:
        Dimensionless overlap score
    """
    threshold2_array = np.asarray(threshold2)
    overlap = np.maximum(threshold2_array - dist2, 0.0)
    normalized = np.divide(
        overlap,
        threshold2_array,
        out=np.zeros_like(overlap),
        where=threshold2_array > 0.0,
    )
    return float(np.sum(normalized))


@dataclass(frozen=True, slots=True)
class ExcludedVolumeSpec:
    """Static atoms or points that ligand conformers cannot overlap.

    Coordinates must use the input ligand pose's Cartesian reference frame.
    Environment radii can be supplied directly, derived from atomic numbers,
    or omitted for zero-radius avoid points.

    Attributes:
        positions: environment coordinates with shape (N, 3) in Å
        atomic_numbers: atomic numbers used to derive covalent radii
        radii: explicit environment radii in Å
        radius_scale: scale applied to summed ligand and environment radii
    """

    positions: Sequence[Sequence[float]] | NDArray[np.float64]
    atomic_numbers: Sequence[int] | NDArray[np.int_] | None = None
    radii: Sequence[float] | NDArray[np.float64] | None = None
    radius_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize excluded-volume inputs."""
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise OpenConfValueError("excluded-volume positions must have shape (n_points, 3).")
        if not np.isfinite(positions).all():
            raise OpenConfValueError("excluded-volume positions must be finite.")

        atomic_numbers = None
        if self.atomic_numbers is not None:
            raw_atomic_numbers = np.asarray(self.atomic_numbers)
            if not np.issubdtype(raw_atomic_numbers.dtype, np.integer):
                raise OpenConfValueError("excluded-volume atomic_numbers must contain integers.")
            atomic_numbers = raw_atomic_numbers.astype(int, copy=False)
        radii = None if self.radii is None else np.asarray(self.radii, dtype=float)
        if atomic_numbers is not None and radii is not None:
            raise OpenConfValueError("Specify either excluded-volume atomic_numbers or radii, not both.")
        if atomic_numbers is not None:
            if atomic_numbers.shape != (len(positions),):
                raise OpenConfValueError("excluded-volume atomic_numbers must match positions.")
            max_atomic_number = Chem.GetPeriodicTable().GetMaxAtomicNumber()
            if np.any(atomic_numbers <= 0) or np.any(atomic_numbers > max_atomic_number):
                raise OpenConfValueError(f"excluded-volume atomic_numbers must be between 1 and {max_atomic_number}.")
        if radii is not None:
            if radii.shape != (len(positions),):
                raise OpenConfValueError("excluded-volume radii must match positions.")
            if not np.isfinite(radii).all() or np.any(radii < 0.0):
                raise OpenConfValueError("excluded-volume radii must be finite and non-negative.")
        if not np.isfinite(self.radius_scale) or self.radius_scale <= 0.0:
            raise OpenConfValueError("excluded-volume radius_scale must be finite and greater than zero.")

        object.__setattr__(self, "positions", tuple(tuple(float(value) for value in row) for row in positions))
        object.__setattr__(
            self,
            "atomic_numbers",
            None if atomic_numbers is None else tuple(int(value) for value in atomic_numbers),
        )
        object.__setattr__(self, "radii", None if radii is None else tuple(float(value) for value in radii))

    def resolved_radii(self) -> NDArray[np.float64]:
        """Return explicit environment radii in Å."""
        if self.radii is not None:
            return np.asarray(self.radii, dtype=float)
        if self.atomic_numbers is None:
            return np.zeros(len(self.positions), dtype=float)
        periodic_table = Chem.GetPeriodicTable()
        return np.asarray([periodic_table.GetRcovalent(int(number)) for number in self.atomic_numbers], dtype=float)


@dataclass(frozen=True, slots=True)
class ExcludedVolumeChecker:
    """Score ligand overlap with static external volumes."""

    mol: Chem.Mol
    spec: ExcludedVolumeSpec
    _environment_positions: NDArray[np.float64] = field(init=False, repr=False)
    _threshold2: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache coordinates and ligand-environment thresholds."""
        object.__setattr__(self, "_environment_positions", np.asarray(self.spec.positions, dtype=float))
        periodic_table = Chem.GetPeriodicTable()
        ligand_radii = np.asarray(
            [periodic_table.GetRcovalent(atom.GetAtomicNum()) for atom in self.mol.GetAtoms()],
            dtype=float,
        )
        environment_radii = self.spec.resolved_radii()
        thresholds = self.spec.radius_scale * (ligand_radii[:, None] + environment_radii[None, :])
        object.__setattr__(self, "_threshold2", thresholds * thresholds)

    def clash_score(self, conf_id: int, atom_indices: Sequence[int] | None = None) -> float:
        """Return normalized ligand-environment overlap score.

        Args:
            conf_id: conformer ID
            atom_indices: ligand atoms to check

        Returns:
            Dimensionless overlap score
        """
        if not len(self._environment_positions):
            return 0.0
        ligand_positions = self.mol.GetConformer(int(conf_id)).GetPositions()
        thresholds = self._threshold2
        if atom_indices is not None:
            indices = np.asarray(atom_indices, dtype=np.intp)
            ligand_positions = ligand_positions[indices]
            thresholds = thresholds[indices]
        differences = ligand_positions[:, None, :] - self._environment_positions[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", differences, differences)
        return sum_clash_scores(dist2, thresholds)
