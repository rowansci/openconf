"""Move execution helpers for hybrid conformer proposals."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
from rdkit.Chem import rdMolTransforms

from ..perceive import RotorModel, _is_metal, _ring_flip_moving_atoms

if TYPE_CHECKING:
    from rdkit import Chem

    from ..config import ConformerConfig
    from ..torsionlib import TorsionLibrary


def _set_dihedral(mol: Chem.Mol, conf_id: int, atoms: tuple[int, int, int, int], angle_deg: float) -> None:
    """Set a dihedral angle in a conformer."""
    conf = mol.GetConformer(conf_id)
    rdMolTransforms.SetDihedralDeg(conf, atoms[0], atoms[1], atoms[2], atoms[3], angle_deg)


class MoveExecutor:
    """Executes move operators for the hybrid proposal engine."""

    def __init__(
        self,
        mol: Chem.Mol,
        rotor_model: RotorModel,
        torsion_lib: TorsionLibrary,
        config: ConformerConfig,
    ) -> None:
        """Initialize move execution state for one molecule."""
        self.mol = mol
        self.rotor_model = rotor_model
        self.config = config

        metal_angles = np.linspace(0.0, 360.0, 12, endpoint=False)
        metal_weights = np.full(12, 1.0 / 12.0)
        self._rotor_angles: list[tuple[np.ndarray, np.ndarray]] = []
        for rotor in rotor_model.rotors:
            if rotor.rotor_type == "metal_ligand":
                self._rotor_angles.append((metal_angles, metal_weights))
            else:
                angles, weights = torsion_lib.get_preferred_angles(mol, rotor.dihedral_atoms)
                angles_arr = np.array(angles, dtype=np.float64)
                weights_arr = np.array(weights, dtype=np.float64)
                weights_arr /= weights_arr.sum()
                self._rotor_angles.append((angles_arr, weights_arr))

        self.correlated_rotor_indices = [i for i, rotor in enumerate(rotor_model.rotors) if rotor.neighbors]
        self.crankable_rings = self._build_crankable_rings(mol, rotor_model)
        self.ring_flip_moving_atoms = [
            tuple(sorted(_ring_flip_moving_atoms(mol, ring_flip.ring_atoms, ring_flip.junction_atoms)))
            for ring_flip in rotor_model.ring_flips
        ]
        self.operators = {
            "single_rotor": self.apply_single_rotor_move,
            "multi_rotor": self.apply_multi_rotor_move,
            "correlated": self.apply_correlated_move,
            "global_shake": self.apply_global_shake,
            "ring_flip": self.apply_ring_flip_move,
            "crankshaft": self.apply_crankshaft_move,
        }

    @staticmethod
    def _compute_substituent_atoms(mol: Chem.Mol, ring_atoms: frozenset[int], ring_atom: int) -> frozenset[int]:
        """Atoms reachable from ring_atom without re-entering the same ring.

        Metal centers are not traversed: dragging a coordination center as a
        substituent of a nearby ring atom produces unphysical geometries.
        """
        visited: set[int] = {ring_atom}
        stack = [ring_atom]
        while stack:
            cur = stack.pop()
            for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
                idx = nb.GetIdx()
                if idx in visited:
                    continue
                if idx in ring_atoms and idx != ring_atom:
                    continue
                if _is_metal(mol.GetAtomWithIdx(idx)):
                    continue
                visited.add(idx)
                stack.append(idx)
        return frozenset(visited)

    def _build_crankable_rings(
        self,
        mol: Chem.Mol,
        rotor_model: RotorModel,
    ) -> list[tuple[tuple[int, ...], list[frozenset[int]]]]:
        """Precompute crankshaft ring metadata."""
        crankable: list[tuple[tuple[int, ...], list[frozenset[int]]]] = []
        atom_rings = rotor_model.ring_info.get("ring_atoms", [])
        for ring in atom_rings:
            if len(ring) < 6:
                continue
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                continue
            if any(_is_metal(mol.GetAtomWithIdx(i)) for i in ring):
                continue
            ring_set = frozenset(ring)
            subtrees = [self._compute_substituent_atoms(mol, ring_set, i) for i in ring]
            crankable.append((tuple(ring), subtrees))
        return crankable

    def sample_angle(self, rotor_idx: int) -> float:
        """Sample a torsion angle for a rotor."""
        angles_arr, weights_arr = self._rotor_angles[rotor_idx]
        idx = np.random.choice(len(angles_arr), p=weights_arr)
        jitter = np.random.normal(0.0, self.config.torsion_jitter_deg)
        return float(angles_arr[idx]) + jitter

    def apply_single_rotor_move(self, conf_id: int) -> None:
        """Apply a single rotor move."""
        if not self.rotor_model.rotors:
            return
        rotor_idx = random.randrange(len(self.rotor_model.rotors))
        rotor = self.rotor_model.rotors[rotor_idx]
        _set_dihedral(self.mol, conf_id, rotor.dihedral_atoms, self.sample_angle(rotor_idx))

    def apply_multi_rotor_move(self, conf_id: int, n_rotors: int = 3) -> None:
        """Apply multiple independent rotor moves."""
        if not self.rotor_model.rotors:
            return
        n = min(n_rotors, len(self.rotor_model.rotors))
        rotor_indices = random.sample(range(len(self.rotor_model.rotors)), n)
        for rotor_idx in rotor_indices:
            rotor = self.rotor_model.rotors[rotor_idx]
            _set_dihedral(self.mol, conf_id, rotor.dihedral_atoms, self.sample_angle(rotor_idx))

    def apply_correlated_move(self, conf_id: int) -> None:
        """Apply a correlated move across a local rotor neighborhood."""
        if not self.rotor_model.rotors:
            return
        if not self.correlated_rotor_indices:
            self.apply_single_rotor_move(conf_id)
            return
        center_idx = random.choice(self.correlated_rotor_indices)
        center_rotor = self.rotor_model.rotors[center_idx]
        for rotor_idx in [center_idx, *center_rotor.neighbors]:
            rotor = self.rotor_model.rotors[rotor_idx]
            _set_dihedral(self.mol, conf_id, rotor.dihedral_atoms, self.sample_angle(rotor_idx))

    def apply_global_shake(self, conf_id: int) -> None:
        """Apply a global shake move."""
        if not self.rotor_model.rotors:
            return
        n_to_change = max(
            1,
            random.randint(len(self.rotor_model.rotors) // 2, int(len(self.rotor_model.rotors) * 0.8) + 1),
        )
        rotor_indices = random.sample(
            range(len(self.rotor_model.rotors)), min(n_to_change, len(self.rotor_model.rotors))
        )
        for rotor_idx in rotor_indices:
            rotor = self.rotor_model.rotors[rotor_idx]
            _set_dihedral(self.mol, conf_id, rotor.dihedral_atoms, self.sample_angle(rotor_idx))

    def apply_crankshaft_move(self, conf_id: int) -> None:
        """Rotate an arc of ring atoms about the axis through two anchor atoms."""
        if not self.crankable_rings:
            return

        ring_atoms, subtrees = random.choice(self.crankable_rings)
        n = len(ring_atoms)
        p = random.randrange(n)
        arc_len = random.randint(1, n - 3)
        interior = [(p + 1 + k) % n for k in range(arc_len)]
        q = (p + arc_len + 1) % n

        anchor_p_idx = ring_atoms[p]
        anchor_q_idx = ring_atoms[q]
        conf = self.mol.GetConformer(conf_id)
        all_pos = conf.GetPositions()
        axis_origin = all_pos[anchor_p_idx]
        axis_vec = all_pos[anchor_q_idx] - axis_origin
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm < 1e-6:
            return
        axis_unit = axis_vec / axis_norm

        base = random.uniform(30.0, 120.0) * random.choice([-1.0, 1.0])
        theta = np.deg2rad(base + np.random.normal(0.0, self.config.torsion_jitter_deg))

        c, s = np.cos(theta), np.sin(theta)
        one_c = 1.0 - c
        ux, uy, uz = axis_unit
        rotation = np.array(
            [
                [c + ux * ux * one_c, ux * uy * one_c - uz * s, ux * uz * one_c + uy * s],
                [uy * ux * one_c + uz * s, c + uy * uy * one_c, uy * uz * one_c - ux * s],
                [uz * ux * one_c - uy * s, uz * uy * one_c + ux * s, c + uz * uz * one_c],
            ]
        )

        moving: set[int] = set()
        for pos_in_ring in interior:
            moving |= subtrees[pos_in_ring]
        if not moving:
            return

        moving_arr = np.fromiter(moving, dtype=np.int64, count=len(moving))
        pts = all_pos[moving_arr] - axis_origin
        rotated = pts @ rotation.T + axis_origin
        for atom_idx, new_xyz in zip(moving_arr.tolist(), rotated, strict=True):
            conf.SetAtomPosition(int(atom_idx), new_xyz.tolist())

    def apply_ring_flip_move(self, conf_id: int) -> None:
        """Reflect a non-aromatic ring through its mean plane."""
        if not self.rotor_model.ring_flips:
            return

        ring_flip_idx = random.randrange(len(self.rotor_model.ring_flips))
        ring_flip = self.rotor_model.ring_flips[ring_flip_idx]
        ring_atoms = ring_flip.ring_atoms
        moving_atoms = self.ring_flip_moving_atoms[ring_flip_idx]
        conf = self.mol.GetConformer(conf_id)
        all_pos = conf.GetPositions()
        ring_idx = list(ring_atoms)
        ring_pos = all_pos[ring_idx]

        centroid = ring_pos.mean(axis=0)
        _, _, vh = np.linalg.svd(ring_pos - centroid)
        normal = vh[-1]
        moving_idx = list(moving_atoms)
        moving_pos = all_pos[moving_idx]
        signed_dists = (moving_pos - centroid) @ normal
        reflected = moving_pos - 2.0 * signed_dists[:, None] * normal

        for atom_idx, new_xyz in zip(moving_idx, reflected, strict=True):
            conf.SetAtomPosition(atom_idx, new_xyz.tolist())
