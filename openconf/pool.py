"""Conformer pool management for generation workflows."""

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors3D, rdFreeSASA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import ConformerConfig
from .dedupe import prism_dedupe


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


def _pick_diverse_by_clustering(
    mol: Chem.Mol,
    conf_ids: list[int],
    energies: list[float],
    k: int,
) -> list[int]:
    """Select k diverse conformers using k-means clustering on 3D descriptors.

    Clusters conformers by shape descriptors, then picks the lowest-energy
    conformer from each cluster.

    Args:
        mol: RDKit molecule with conformers.
        conf_ids: List of conformer IDs to cluster.
        energies: Energies corresponding to each conformer.
        k: Number of clusters (conformers to select).

    Returns:
        List of selected conformer IDs (one per cluster).
    """
    if k >= len(conf_ids):
        return conf_ids

    # Pre-compute SASA radii once (topology-only; same for all conformers).
    radii = rdFreeSASA.classifyAtoms(mol)

    # Build feature matrix
    features = np.array([_build_3d_descriptors(mol, cid, radii) for cid in conf_ids])

    # Scale features for k-means
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Run k-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)

    # Group conformers by cluster
    clusters: dict[int, list[int]] = {i: [] for i in range(k)}
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(idx)

    # Select lowest-energy conformer from each cluster
    energies_array = np.array(energies)
    selected = []
    for _cluster_id, indices in clusters.items():
        if not indices:
            continue
        cluster_energies = energies_array[indices]
        best_idx = indices[np.argmin(cluster_energies)]
        selected.append(conf_ids[best_idx])

    return selected


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
        return list(self.records.keys())

    @property
    def energies(self) -> list[float]:
        """List of energies (in order of conf_ids)."""
        return [self.records[cid].energy_kcal or float("inf") for cid in self.conf_ids]

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
        if self.size >= self.config.pool_max:
            # Refresh cached worst if stale (O(N) scan, amortised rare).
            if self._worst_dirty or self._worst_id is None:
                self._worst_id = max(
                    self.records,
                    key=lambda cid: self.records[cid].energy_kcal or float("inf"),
                )
                self._worst_energy = self.records[self._worst_id].energy_kcal or float("inf")
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

        # Run PRISM dedupe
        keep_ids = prism_dedupe(
            self.mol,
            conf_ids,
            self.config.prism_config,
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
            self._best_energy = min(r.energy_kcal or float("inf") for r in self.records.values())

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

        # Diverse final selection (k-means clustering on 3D descriptors)
        return _pick_diverse_by_clustering(
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
        if not self.records:
            return None

        conf_ids = self.conf_ids
        energies = self.energies

        if strategy == "best":
            return conf_ids[energies.index(min(energies))]

        if strategy == "uniform":
            return random.choice(conf_ids)

        # Softmax selection (biased toward lower energy)
        energies_array = np.array(energies)
        # Shift to avoid numerical issues
        shifted = energies_array - np.min(energies_array)
        # Temperature controls exploration/exploitation
        temperature = 2.0  # kcal/mol
        weights = np.exp(-shifted / temperature)
        weights /= weights.sum()

        return np.random.choice(conf_ids, p=weights)

    def prune_by_energy(self) -> int:
        """Remove conformers outside energy window.

        Returns:
            Number removed.
        """
        if not self.records:
            return 0

        # Update best energy
        self._best_energy = min(r.energy_kcal or float("inf") for r in self.records.values())

        cutoff = self._best_energy + self.config.energy_window_kcal

        to_remove = [cid for cid, record in self.records.items() if (record.energy_kcal or float("inf")) > cutoff]

        for cid in to_remove:
            del self.records[cid]
            self.mol.RemoveConformer(cid)

        self._worst_dirty = True
        return len(to_remove)

    def get_statistics(self) -> dict[str, Any]:
        """Get pool statistics.

        Returns:
            Dictionary of statistics.
        """
        if not self.records:
            return {
                "size": 0,
                "best_energy": None,
                "worst_energy": None,
                "mean_energy": None,
            }

        energies = [r.energy_kcal for r in self.records.values() if r.energy_kcal is not None]

        return {
            "size": self.size,
            "best_energy": min(energies) if energies else None,
            "worst_energy": max(energies) if energies else None,
            "mean_energy": sum(energies) / len(energies) if energies else None,
            "sources": self._count_sources(),
        }

    def _count_sources(self) -> dict[str, int]:
        """Count conformers by source."""
        counts: dict[str, int] = {}
        for record in self.records.values():
            counts[record.source] = counts.get(record.source, 0) + 1
        return counts
