"""Conformer pool management for generation workflows."""

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors3D, rdFreeSASA

from .config import ConformerConfig
from .dedupe import prism_dedupe


def _energy_or_inf(energy: float | None) -> float:
    """Return a finite sentinel for missing energies."""
    return energy if energy is not None else float("inf")


def _build_3d_descriptors(mol: Chem.Mol, conf_id: int, radii: object) -> list[float]:
    """Build 3D descriptors for a single conformer without copying the molecule.

    Uses SASA, polar SASA, radius of gyration, PBF, NPR1, NPR2.

    Args:
        mol: RDKit molecule (must have conformer conf_id).
        conf_id: RDKit conformer ID.
        radii: Pre-computed SASA radii from rdFreeSASA.classifyAtoms(mol).
            Despite the name, CalcSASA's confIdx parameter accepts a conf_id.

    Returns:
        List of 3D descriptor values [sasa, polar_sasa, rog, pbf, npr1, npr2].
    """
    # SASA — writes per-atom "SASA" props as a side-effect; read immediately.
    # Note: rdFreeSASA.CalcSASA's confIdx parameter accepts a conf_id (RDKit ID),
    # not a 0-based index despite the parameter name.
    sasa = rdFreeSASA.CalcSASA(mol, radii, confIdx=conf_id)
    polar_sasa = sum(float(atom.GetProp("SASA")) for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})

    return [
        sasa,
        polar_sasa,
        Descriptors3D.RadiusOfGyration(mol, conf_id),
        Descriptors3D.PBF(mol, conf_id),
        Descriptors3D.NPR1(mol, conf_id),
        Descriptors3D.NPR2(mol, conf_id),
    ]


def _softmax_parent_weights(energies: np.ndarray, temperature: float) -> np.ndarray:
    """Return numerically stable parent-selection weights."""
    finite = np.isfinite(energies)
    if not finite.any():
        return np.full(len(energies), 1.0 / len(energies))

    shifted = energies[finite] - energies[finite].min()
    weights = np.zeros(len(energies), dtype=float)
    weights[finite] = np.exp(-shifted / temperature)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(len(energies), 1.0 / len(energies))
    weights /= total
    return weights


def _pick_diverse_maxmin(
    mol: Chem.Mol,
    conf_ids: list[int],
    energies: list[float],
    k: int,
) -> list[int]:
    """Select k diverse conformers using greedy MaxMin on shape descriptors.

    Seeds with the lowest-energy conformer, then repeatedly picks the conformer
    farthest (in normalized descriptor space) from all already-selected ones.
    Uses the same six rotation-invariant descriptors as the old clustering approach
    (SASA, polar SASA, Rgyr, PBF, NPR1, NPR2) but directly maximizes the minimum
    pairwise distance rather than clustering.

    Args:
        mol: RDKit molecule with conformers.
        conf_ids: List of conformer IDs to choose from.
        energies: Energies corresponding to each conformer.
        k: Number of conformers to select.

    Returns:
        List of selected conformer IDs (length <= k).
    """
    if k >= len(conf_ids):
        return conf_ids

    radii = rdFreeSASA.classifyAtoms(mol)
    features = np.array([_build_3d_descriptors(mol, cid, radii) for cid in conf_ids])

    # Normalize: zero mean, unit variance (avoid div-by-zero on flat features).
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0.0] = 1.0
    feat = (features - mean) / std

    energies_arr = np.array(energies)

    # Seed: lowest-energy conformer.
    seed = int(np.argmin(energies_arr))
    selected = [seed]

    # min_dist_sq[i] = squared distance from conformer i to nearest selected.
    # Sentinel -inf marks already-selected slots so argmax never revisits them.
    diff = feat - feat[seed]
    min_dist_sq = (diff * diff).sum(axis=1)
    min_dist_sq[seed] = -np.inf

    for _ in range(k - 1):
        next_idx = int(np.argmax(min_dist_sq))
        selected.append(next_idx)
        diff = feat - feat[next_idx]
        new_dists = (diff * diff).sum(axis=1)
        np.minimum(min_dist_sq, new_dists, out=min_dist_sq)
        min_dist_sq[next_idx] = -np.inf

    return [conf_ids[i] for i in selected]


@dataclass
class ConformerRecord:
    """Metadata for a single conformer.

    Attributes:
        conf_id: RDKit conformer ID.
        energy_kcal: Energy in kcal/mol (may be None if not computed).
        source: How this conformer was generated.
        tags: Additional metadata (step index, torsion fingerprint, etc.).
    """

    conf_id: int
    energy_kcal: float | None = None
    source: str = "unknown"
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentSampler:
    """Cached parent selector for a conformer pool."""

    pool: "ConformerPool"
    _dirty: bool = field(default=True, init=False, repr=False)
    _records_ref: dict[int, ConformerRecord] | None = field(default=None, init=False, repr=False)
    _records_version: int = field(default=-1, init=False, repr=False)
    _conf_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _energies: np.ndarray = field(default_factory=lambda: np.array([], dtype=float), init=False, repr=False)
    _best_index: int | None = field(default=None, init=False, repr=False)
    _softmax_temperature: float | None = field(default=None, init=False, repr=False)
    _softmax_weights: np.ndarray | None = field(default=None, init=False, repr=False)

    def mark_dirty(self) -> None:
        """Invalidate cached views after the pool changes."""
        self._dirty = True
        self._softmax_temperature = None
        self._softmax_weights = None

    def _needs_refresh(self) -> bool:
        """Return whether cached pool views are stale."""
        return (
            self._dirty
            or self._records_ref is not self.pool.records
            or self._records_version != self.pool._records_version
        )

    def _refresh(self) -> None:
        """Rebuild cached conformer IDs and energies."""
        self._records_ref = self.pool.records
        self._records_version = self.pool._records_version
        self._conf_ids = list(self.pool.records.keys())
        self._energies = np.array(
            [_energy_or_inf(self.pool.records[cid].energy_kcal) for cid in self._conf_ids],
            dtype=float,
        )
        self._best_index = int(np.argmin(self._energies)) if len(self._conf_ids) > 0 else None
        self._softmax_temperature = None
        self._softmax_weights = None
        self._dirty = False

    def _ensure_fresh(self) -> None:
        """Refresh cached pool views when necessary."""
        if self._needs_refresh():
            self._refresh()

    def conf_ids(self) -> list[int]:
        """Return conformer IDs in cached pool order."""
        self._ensure_fresh()
        return self._conf_ids.copy()

    def energies(self) -> list[float]:
        """Return cached energies aligned to ``conf_ids``."""
        self._ensure_fresh()
        return self._energies.tolist()

    def select(self, strategy: str, temperature: float) -> int | None:
        """Select a parent conformer ID using the requested strategy."""
        self._ensure_fresh()
        if not self._conf_ids:
            return None

        if strategy == "best":
            assert self._best_index is not None
            return self._conf_ids[self._best_index]

        if strategy == "uniform":
            return random.choice(self._conf_ids)

        if self._softmax_weights is None or self._softmax_temperature != temperature:
            self._softmax_weights = _softmax_parent_weights(self._energies, temperature)
            self._softmax_temperature = temperature

        return int(np.random.choice(self._conf_ids, p=self._softmax_weights))


@dataclass
class ConformerPool:
    """Manages a pool of conformers during generation.

    Handles insertion, energy filtering, periodic deduplication,
    and final selection.

    Attributes:
        mol: RDKit molecule (stores conformers).
        config: Generation configuration.
    """

    mol: Chem.Mol
    config: ConformerConfig
    records: dict[int, ConformerRecord] = field(default_factory=dict)
    _best_energy: float = field(default=float("inf"), init=False)
    _steps_since_dedupe: int = field(default=0, init=False)
    # Cached worst conformer — avoids O(N) max() scan on every insert.
    # Set _worst_dirty=True whenever the worst may have been removed so that
    # the next lookup triggers a fresh O(N) scan.
    _worst_energy: float = field(default=float("-inf"), init=False)
    _worst_id: int | None = field(default=None, init=False)
    _worst_dirty: bool = field(default=False, init=False)
    _records_version: int = field(default=0, init=False)
    _parent_sampler: ParentSampler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize cached helpers."""
        self._parent_sampler = ParentSampler(self)

    @property
    def size(self) -> int:
        """Number of conformers in pool."""
        return len(self.records)

    @property
    def best_energy(self) -> float:
        """Best (lowest) energy in pool."""
        return self._best_energy

    @property
    def conf_ids(self) -> list[int]:
        """List of conformer IDs in pool."""
        return self._parent_sampler.conf_ids()

    @property
    def energies(self) -> list[float]:
        """List of energies (in order of conf_ids)."""
        return self._parent_sampler.energies()

    def _mark_records_changed(self) -> None:
        """Invalidate cached selectors after mutating the pool."""
        self._records_version += 1
        self._parent_sampler.mark_dirty()

    def insert(
        self,
        conf_id: int,
        energy: float,
        source: str = "unknown",
        tags: dict[str, Any] | None = None,
    ) -> bool:
        """Insert a conformer into the pool.

        Args:
            conf_id: RDKit conformer ID.
            energy: Energy in kcal/mol.
            source: Source of the conformer.
            tags: Additional metadata.

        Returns:
            True if conformer was accepted, False if rejected.
        """
        # Update best energy (only ever decreases; no full scan needed).
        self._best_energy = min(self._best_energy, energy)

        # Check energy window
        if energy > self._best_energy + self.config.energy_window_kcal:
            return False

        # Check pool size limit
        assert self.config.pool_max is not None
        if self.size >= self.config.pool_max:
            # Refresh cached worst if stale (O(N) scan, amortised rare).
            if self._worst_dirty or self._worst_id is None:
                self._worst_id = max(
                    self.records,
                    key=lambda cid: _energy_or_inf(self.records[cid].energy_kcal),
                )
                self._worst_energy = _energy_or_inf(self.records[self._worst_id].energy_kcal)
                self._worst_dirty = False

            if energy >= self._worst_energy:
                return False

            # Remove worst (O(1) now that we know _worst_id).
            del self.records[self._worst_id]
            self.mol.RemoveConformer(self._worst_id)
            self._worst_dirty = True  # next worst needs a fresh scan

        # Add new conformer.
        self.records[conf_id] = ConformerRecord(
            conf_id=conf_id,
            energy_kcal=energy,
            source=source,
            tags=tags or {},
        )

        # Update cached worst in O(1) if the new conformer is the new worst.
        if not self._worst_dirty and (self._worst_id is None or energy > self._worst_energy):
            self._worst_energy = energy
            self._worst_id = conf_id

        self._mark_records_changed()
        self._steps_since_dedupe += 1
        return True

    def should_dedupe(self) -> bool:
        """Check if deduplication should be run."""
        return self._steps_since_dedupe >= self.config.dedupe_period

    def dedupe(self) -> int:
        """Run deduplication on the pool.

        Returns:
            Number of conformers removed.
        """
        if self.size <= 1:
            return 0

        old_size = self.size

        # Get current conf_ids
        conf_ids = self.conf_ids

        keep_ids = prism_dedupe(
            self.mol,
            conf_ids,
            use_heavy_atoms_only=self.config.use_heavy_atoms_only,
        )

        # Remove duplicates
        keep_set = set(keep_ids)
        to_remove = [cid for cid in conf_ids if cid not in keep_set]

        for cid in to_remove:
            del self.records[cid]
            self.mol.RemoveConformer(cid)

        self._steps_since_dedupe = 0
        self._worst_dirty = True
        # Also refresh _best_energy since dedupe may have removed the best.
        if self.records:
            self._best_energy = min(_energy_or_inf(r.energy_kcal) for r in self.records.values())
        self._mark_records_changed()

        return old_size - self.size

    def select_final(self) -> list[int]:
        """Select final diverse conformers.

        Uses PRISM deduplication, then takes the lowest-energy conformers
        if still over max_out.

        Returns:
            List of selected conformer IDs.
        """
        # Run final dedupe with PRISM
        self.dedupe()

        if self.size <= self.config.max_out:
            return self.conf_ids

        conf_ids = self.conf_ids
        energies = self.energies

        # Take lowest-energy conformers
        if self.config.final_select == "energy":
            sorted_pairs = sorted(zip(energies, conf_ids, strict=True))
            return [cid for _, cid in sorted_pairs[: self.config.max_out]]

        # Diverse final selection (greedy MaxMin on shape descriptors)
        return _pick_diverse_maxmin(
            self.mol,
            conf_ids,
            energies,
            k=self.config.max_out,
        )

    def get_parent(self, strategy: str = "softmax") -> int | None:
        """Select a parent conformer for mutation.

        Args:
            strategy: Selection strategy ("softmax", "uniform", "best").

        Returns:
            Conformer ID or None if pool is empty.
        """
        return self._parent_sampler.select(strategy, self.config.parent_softmax_temperature_kcal)
