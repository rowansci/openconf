"""Move execution helpers for hybrid conformer proposals."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np
from rdkit.Chem import rdMolTransforms

from ..perceive import (
    RotorModel,
    _is_metal,
    _ring_flip_moving_atoms,
    conformer_matches_specified_stereochemistry,
    specified_stereochemistry,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from rdkit import Chem

    from ..config import ConformerConfig
    from ..torsionlib import TorsionLibrary


def _set_dihedral(mol: Chem.Mol, conf_id: int, atoms: tuple[int, int, int, int], angle_deg: float) -> None:
    """Set a dihedral angle in a conformer."""
    conf = mol.GetConformer(conf_id)
    rdMolTransforms.SetDihedralDeg(conf, atoms[0], atoms[1], atoms[2], atoms[3], angle_deg)


def _rodrigues_rotate(
    all_pos: np.ndarray,
    arr: np.ndarray,
    origin: np.ndarray,
    ux: float,
    uy: float,
    uz: float,
    theta: float,
) -> None:
    """Rotate atoms at arr indices in-place about axis (ux,uy,uz) through origin by theta radians."""
    if not len(arr):
        return
    c = math.cos(theta)
    s = math.sin(theta)
    one_c = 1.0 - c
    R = np.array(
        [
            [c + ux * ux * one_c, ux * uy * one_c - uz * s, ux * uz * one_c + uy * s],
            [uy * ux * one_c + uz * s, c + uy * uy * one_c, uy * uz * one_c - ux * s],
            [uz * ux * one_c - uy * s, uz * uy * one_c + ux * s, c + uz * uz * one_c],
        ]
    )
    pts = all_pos[arr] - origin
    all_pos[arr] = pts @ R.T + origin


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
        self.reference_stereo = specified_stereochemistry(mol)

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
        self.macro_kic_data = [
            (ring_atoms, subtrees) for ring_atoms, subtrees in self.crankable_rings if len(ring_atoms) >= 10
        ]
        self.ring_flip_moving_atoms = [
            tuple(sorted(_ring_flip_moving_atoms(mol, ring_flip.ring_atoms, ring_flip.junction_atoms)))
            for ring_flip in rotor_model.ring_flips
        ]
        self.ring_flip_subtrees = [
            [
                self._compute_substituent_atoms(mol, frozenset(ring_flip.ring_atoms), ring_atom)
                for ring_atom in ring_flip.ring_atoms
            ]
            for ring_flip in rotor_model.ring_flips
        ]
        self.operators: dict[str, Callable[[int], None]] = {
            "single_rotor": self.apply_single_rotor_move,
            "multi_rotor": self.apply_multi_rotor_move,
            "correlated": self.apply_correlated_move,
            "global_shake": self.apply_global_shake,
            "ring_flip": self.apply_ring_flip_move,
            "crankshaft": self.apply_crankshaft_move,
            "ring_kic": self.apply_ring_kic_move,
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
        all_pos[moving_arr] = rotated
        conf.SetPositions(all_pos)

    def apply_ring_flip_move(self, conf_id: int) -> None:
        """Apply ring flip or stereo-preserving ring pucker move."""
        if not self.rotor_model.ring_flips:
            return

        ring_flip_idx = random.randrange(len(self.rotor_model.ring_flips))
        ring_flip = self.rotor_model.ring_flips[ring_flip_idx]
        if ring_flip.stereo_sensitive:
            self._apply_stereo_preserving_ring_flip_move(conf_id, ring_flip_idx)
            return

        ring_atoms = ring_flip.ring_atoms
        moving_atoms = self.ring_flip_moving_atoms[ring_flip_idx]
        conf = self.mol.GetConformer(conf_id)
        original_pos = conf.GetPositions()
        all_pos = conf.GetPositions()
        ring_idx = list(ring_atoms)
        ring_pos = all_pos[ring_idx]

        centroid = ring_pos.mean(axis=0)
        _, _, vh = np.linalg.svd(ring_pos - centroid)
        normal = vh[-1]
        moving_idx = np.array(list(moving_atoms), dtype=np.int64)
        moving_pos = all_pos[moving_idx]
        signed_dists = (moving_pos - centroid) @ normal
        reflected = moving_pos - 2.0 * signed_dists[:, None] * normal
        all_pos[moving_idx] = reflected
        conf.SetPositions(all_pos)
        if not conformer_matches_specified_stereochemistry(self.mol, conf_id, self.reference_stereo):
            conf.SetPositions(original_pos)

    def _apply_stereo_preserving_ring_flip_move(self, conf_id: int, ring_flip_idx: int) -> None:
        """Apply proper rotational ring-pucker moves and keep only stereo-valid results."""
        ring_flip = self.rotor_model.ring_flips[ring_flip_idx]
        ring_atoms = ring_flip.ring_atoms
        n = len(ring_atoms)
        subtrees = self.ring_flip_subtrees[ring_flip_idx]
        allowed_arcs: list[tuple[int, int, int, list[int]]] = []

        for p in range(n):
            for arc_len in range(1, n - 2):
                q = (p + arc_len + 1) % n
                interior = [(p + 1 + k) % n for k in range(arc_len)]
                if any(ring_atoms[pos] in ring_flip.junction_atoms for pos in interior):
                    continue
                allowed_arcs.append((p, q, arc_len, interior))

        if not allowed_arcs:
            return

        conf = self.mol.GetConformer(conf_id)
        original_pos = conf.GetPositions()
        for _ in range(10):
            all_pos = original_pos.copy()
            p, q, _arc_len, interior = random.choice(allowed_arcs)
            anchor_p_idx = ring_atoms[p]
            anchor_q_idx = ring_atoms[q]
            axis_origin = all_pos[anchor_p_idx]
            axis_vec = all_pos[anchor_q_idx] - axis_origin
            axis_norm = float(np.linalg.norm(axis_vec))
            if axis_norm < 1e-6:
                continue
            axis_unit = axis_vec / axis_norm

            moving: set[int] = set()
            for pos_in_ring in interior:
                moving |= subtrees[pos_in_ring]
            if not moving:
                continue

            moving_arr = np.fromiter(moving, dtype=np.int64, count=len(moving))
            base = random.uniform(60.0, 180.0) * random.choice([-1.0, 1.0])
            theta = np.deg2rad(base + np.random.normal(0.0, self.config.torsion_jitter_deg))
            _rodrigues_rotate(
                all_pos,
                moving_arr,
                axis_origin,
                float(axis_unit[0]),
                float(axis_unit[1]),
                float(axis_unit[2]),
                theta,
            )
            conf.SetPositions(all_pos)
            if conformer_matches_specified_stereochemistry(self.mol, conf_id, self.reference_stereo):
                return

        conf.SetPositions(original_pos)

    def apply_ring_kic_move(self, conf_id: int) -> None:
        """CCD kinematic ring closure move for macrocycle sampling.

        Opens the ring at a random bond, applies 1-2 random driver torsions
        to the first portion of the chain, then uses cyclic coordinate descent
        (CCD) on 3 closure torsions to bring the endpoint back to the anchor.
        Reverts silently if CCD does not converge within 1 Å.
        """
        if not self.macro_kic_data:
            return

        ring_atoms, subtrees = random.choice(self.macro_kic_data)
        n = len(ring_atoms)
        # Break ring after position p: anchor = ring_atoms[p],
        # open chain = ring_atoms[(p+1)%n], ..., ring_atoms[(p+n-1)%n].
        p = random.randrange(n)

        conf = self.mol.GetConformer(conf_id)
        all_pos = conf.GetPositions()

        anchor_idx = ring_atoms[p]
        endpoint_idx = ring_atoms[(p + n - 1) % n]
        # Target is the original endpoint position: CCD brings endpoint back there,
        # preserving the ring-closure bond length (endpoint→anchor = orig_d).
        target = all_pos[endpoint_idx].copy()

        orig_d = float(np.linalg.norm(target - all_pos[anchor_idx]))
        if orig_d < 1e-6:
            return

        # Chain bonds: bond k connects chain[k]=ring_atoms[(p+1+k)%n] to
        # chain[k+1]=ring_atoms[(p+2+k)%n], for k in 0..n-3.
        n_closure = 3
        n_chain_bonds = n - 2
        # Shift closure window left by 1 so endpoint (chain[n-2]) is never the *tip*
        # of a closure bond.  If endpoint = tip it lies on the rotation axis and the
        # CCD step has no leverage on it (v_perp == 0 → θ_opt = 0).
        closure_start = n_chain_bonds - n_closure - 1

        # Precompute downstream atom arrays for the 3 closure bonds (used in CCD loop).
        ccd_bonds: list[tuple[int, int, np.ndarray]] = []
        for k in range(closure_start, closure_start + n_closure):
            origin_atom = ring_atoms[(p + 1 + k) % n]
            tip_atom = ring_atoms[(p + 2 + k) % n]
            moving: set[int] = set()
            for j in range(k + 1, n - 1):
                moving |= subtrees[(p + 1 + j) % n]
            arr = np.fromiter(moving, dtype=np.int64, count=len(moving)) if moving else np.array([], dtype=np.int64)
            ccd_bonds.append((origin_atom, tip_atom, arr))

        _pi_over_180 = math.pi / 180.0

        # Apply 1-2 driver torsions (sd=40°, gentler than 60° -> higher CCD acceptance rate).
        n_driver = random.randint(1, 2)
        driver_ks = random.sample(range(closure_start), min(n_driver, closure_start))
        for k in driver_ks:
            o_idx = ring_atoms[(p + 1 + k) % n]
            t_idx = ring_atoms[(p + 2 + k) % n]
            axis_vec = all_pos[t_idx] - all_pos[o_idx]
            axis_norm = float(np.linalg.norm(axis_vec))
            if axis_norm < 1e-6:
                continue
            u = axis_vec / axis_norm
            ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
            moving_d: set[int] = set()
            for j in range(k + 1, n - 1):
                moving_d |= subtrees[(p + 1 + j) % n]
            arr_d = (
                np.fromiter(moving_d, dtype=np.int64, count=len(moving_d)) if moving_d else np.array([], dtype=np.int64)
            )
            _rodrigues_rotate(all_pos, arr_d, all_pos[o_idx], ux, uy, uz, np.random.normal(0.0, 40.0) * _pi_over_180)

        # CCD: iteratively minimise ||endpoint - target|| via 3 closure torsions.
        # 30 outer iterations covers p95 of converging calls (measured: p95~29-35).
        tol2 = (orig_d * 0.1) ** 2
        for _ in range(30):
            ep = all_pos[endpoint_idx]
            dv = ep - target
            if dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2] < tol2:
                break
            for o_idx, t_idx, arr in ccd_bonds:
                origin = all_pos[o_idx]
                axis = all_pos[t_idx] - origin
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm < 1e-6:
                    continue
                u = axis / axis_norm
                ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
                v = all_pos[endpoint_idx] - origin
                v_para = ux * v[0] + uy * v[1] + uz * v[2]
                # v_perp = v - v_para * u; compute inline to avoid extra np.array
                vpx = v[0] - v_para * ux
                vpy = v[1] - v_para * uy
                vpz = v[2] - v_para * uz
                # u x v_perp (inline cross product)
                vrx = uy * vpz - uz * vpy
                vry = uz * vpx - ux * vpz
                vrz = ux * vpy - uy * vpx
                # d = target - origin - v_para * u; dot with v_perp and v_perp_rot
                dx = target[0] - origin[0] - v_para * ux
                dy = target[1] - origin[1] - v_para * uy
                dz = target[2] - origin[2] - v_para * uz
                cos_t = dx * vpx + dy * vpy + dz * vpz
                sin_t = dx * vrx + dy * vry + dz * vrz
                if abs(cos_t) < 1e-12 and abs(sin_t) < 1e-12:
                    continue
                _rodrigues_rotate(all_pos, arr, origin, ux, uy, uz, math.atan2(sin_t, cos_t))

        # Revert if closure gap exceeds 0.3 Å; MMFF handles small remaining distortions.
        if float(np.linalg.norm(all_pos[endpoint_idx] - target)) > 0.3:
            return

        conf.SetPositions(all_pos)
