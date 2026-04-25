"""Candidate staging and clash-check helpers for proposal workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

from ..tuning import is_clash_exempt_move


def build_nonbonded_mask(mol: Chem.Mol) -> np.ndarray:
    """Build a boolean mask of non-bonded, non-1-3 atom pairs."""
    n = mol.GetNumAtoms()
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        mask[i, j] = mask[j, i] = False
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            k = nb.GetIdx()
            if k != j:
                mask[j, k] = mask[k, j] = False
        for nb in mol.GetAtomWithIdx(j).GetNeighbors():
            k = nb.GetIdx()
            if k != i:
                mask[i, k] = mask[k, i] = False
    mask &= np.triu(np.ones((n, n), dtype=bool), k=1)
    return mask


@dataclass
class ClashChecker:
    """Fast clash filter for proposed conformers."""

    mol: Chem.Mol
    nonbonded_mask: np.ndarray
    clash_threshold2: float
    _pair_i: np.ndarray = field(init=False, repr=False)
    _pair_j: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache non-bonded pair indices for repeated clash checks."""
        self._pair_i, self._pair_j = np.nonzero(self.nonbonded_mask)

    def has_clash(
        self,
        conf_id: int,
        move_type: str,
        skip_check: bool = False,
    ) -> bool:
        """Return whether a conformer should be rejected before minimization."""
        if skip_check or is_clash_exempt_move(move_type):
            return False

        pos = self.mol.GetConformer(conf_id).GetPositions()
        diff = pos[self._pair_i] - pos[self._pair_j]
        dist2 = (diff * diff).sum(axis=1)
        return bool((dist2 < self.clash_threshold2).any())


@dataclass(frozen=True)
class StagedCandidate:
    """Candidate conformer staged for batch minimization."""

    main_conf_id: int
    stage_conf_id: int
    move_type: str


@dataclass
class CandidateBatchWorkspace:
    """Batch workspace for staging, minimizing, and committing candidates."""

    main_mol: Chem.Mol
    staging_mol: Chem.RWMol
    staged: list[StagedCandidate]

    @classmethod
    def from_candidates(
        cls,
        main_mol: Chem.Mol,
        staging_mol: Chem.RWMol,
        candidates: list[tuple[int, str]],
    ) -> "CandidateBatchWorkspace":
        """Stage candidate conformers on the minimization workspace."""
        staged: list[StagedCandidate] = []
        for main_conf_id, move_type in candidates:
            conf = main_mol.GetConformer(main_conf_id)
            stage_conf_id = staging_mol.AddConformer(Chem.Conformer(conf), assignId=True)
            staged.append(StagedCandidate(main_conf_id=main_conf_id, stage_conf_id=stage_conf_id, move_type=move_type))
        return cls(main_mol=main_mol, staging_mol=staging_mol, staged=staged)

    @property
    def stage_ids(self) -> list[int]:
        """Return staging conformer IDs in candidate order."""
        return [candidate.stage_conf_id for candidate in self.staged]

    def commit(self, energies: list[float]) -> tuple[list[tuple[int, float, str]], int]:
        """Transfer minimized conformers back to the main molecule."""
        results: list[tuple[int, float, str]] = []
        failures = 0

        for candidate, energy in zip(self.staged, energies, strict=True):
            self.main_mol.RemoveConformer(candidate.main_conf_id)
            if np.isfinite(energy):
                minimized_conf = self.staging_mol.GetConformer(candidate.stage_conf_id)
                final_id = self.main_mol.AddConformer(Chem.Conformer(minimized_conf), assignId=True)
                results.append((final_id, energy, f"hybrid_{candidate.move_type}"))
            else:
                failures += 1

        self.clear()
        return results, failures

    def clear(self) -> None:
        """Reset the staging workspace for the next batch."""
        self.staging_mol.RemoveAllConformers()
        self.staged.clear()
