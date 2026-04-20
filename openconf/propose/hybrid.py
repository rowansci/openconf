"""Hybrid conformer proposal strategy.

Combines torsion library biasing with MCMM-style exploration.
"""

import random

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from ..config import ConformerConfig, ConstraintSpec
from ..perceive import RotorModel, filter_constrained_rotors
from ..pool import ConformerPool
from ..relax import get_minimizer, minimize_confs_mmff
from ..torsionlib import TorsionLibrary


def _compute_n_seeds(rotor_model: RotorModel, n_per_rotor: int = 3) -> int:
    """Compute an appropriate seed count based on molecular complexity.

    Formula: base of n_per_rotor seeds per rotatable bond (min 20), plus 5 seeds
    per non-aromatic ring that can flip, plus ring_size × 10 seeds per macrocycle
    (the low-energy pucker fraction drops rapidly with ring size, so dense
    seeding is the cheapest way to ensure the global basin is found). Capped at
    500.

    Args:
        rotor_model: Rotor model for the molecule.
        n_per_rotor: Seeds per rotatable bond.

    Returns:
        Recommended number of seed conformers.
    """
    base = max(20, rotor_model.n_rotatable * n_per_rotor)
    ring_bonus = len(rotor_model.ring_flips) * 5
    # Low-energy pucker fraction drops super-linearly with ring size, so the
    # seed budget scales with s**2; the 500-seed cap keeps the cost bounded.
    macro_bonus = sum(s * s for s in rotor_model.ring_info["ring_sizes"] if s >= 10)
    return min(500, base + ring_bonus + macro_bonus)


def _set_dihedral(mol: Chem.Mol, conf_id: int, atoms: tuple[int, int, int, int], angle_deg: float) -> None:
    """Set a dihedral angle in a conformer.

    Args:
        mol: RDKit molecule.
        conf_id: Conformer ID.
        atoms: Four atom indices defining the dihedral.
        angle_deg: Target angle in degrees.
    """
    conf = mol.GetConformer(conf_id)
    rdMolTransforms.SetDihedralDeg(conf, atoms[0], atoms[1], atoms[2], atoms[3], angle_deg)


def _get_dihedral(mol: Chem.Mol, conf_id: int, atoms: tuple[int, int, int, int]) -> float:
    """Get a dihedral angle from a conformer.

    Args:
        mol: RDKit molecule.
        conf_id: Conformer ID.
        atoms: Four atom indices defining the dihedral.

    Returns:
        Angle in degrees.
    """
    conf = mol.GetConformer(conf_id)
    return rdMolTransforms.GetDihedralDeg(conf, atoms[0], atoms[1], atoms[2], atoms[3])


def _copy_conformer(mol: Chem.Mol, source_conf_id: int) -> int:
    """Copy a conformer and return the new ID.

    Args:
        mol: RDKit molecule.
        source_conf_id: Source conformer ID.

    Returns:
        New conformer ID.
    """
    source_conf = mol.GetConformer(int(source_conf_id))
    new_conf = Chem.Conformer(source_conf)
    return mol.AddConformer(new_conf, assignId=True)


def _build_nonbonded_mask(mol: Chem.Mol) -> np.ndarray:
    """Build a boolean mask of non-bonded, non-1-3 atom pairs (upper triangle).

    Args:
        mol: RDKit molecule.

    Returns:
        Boolean array of shape (n_atoms, n_atoms) where True means the pair
        should be checked for clashes (not bonded and not a 1-3 angle pair).
    """
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
    # Keep only upper triangle to avoid double-counting
    mask &= np.triu(np.ones((n, n), dtype=bool), k=1)
    return mask


def _has_clash(mol: Chem.Mol, conf_id: int, threshold: float = 1.5) -> bool:
    """Check for atomic clashes using vectorized numpy (no Python atom loop).

    Prefer calling the precomputed-mask variant via HybridProposer._check_clash
    when inside the hot loop — this standalone function recomputes the mask each
    time and is provided for external / testing use only.

    Args:
        mol: RDKit molecule.
        conf_id: Conformer ID to check.
        threshold: Minimum allowed distance between non-bonded atoms (Angstroms).

    Returns:
        True if any non-bonded atom pair is closer than threshold.
    """
    mask = _build_nonbonded_mask(mol)
    pos = mol.GetConformer(conf_id).GetPositions()
    diff = pos[:, None, :] - pos[None, :, :]
    dist2 = (diff * diff).sum(axis=-1)
    return bool((dist2[mask] < threshold * threshold).any())


class HybridProposer:
    """Hybrid conformer proposal strategy.

    Combines:
    - ETKDG seeding
    - MCMM-style torsion moves with library biasing
    - Periodic global shakes
    - Minimization after each proposal
    """

    def __init__(
        self,
        mol: Chem.Mol,
        rotor_model: RotorModel,
        torsion_lib: TorsionLibrary,
        config: ConformerConfig,
        constraint_spec: ConstraintSpec | None = None,
    ):
        """Initialize the hybrid proposer.

        Args:
            mol: RDKit molecule.
            rotor_model: Rotor model for the molecule.
            torsion_lib: Torsion library for biased sampling.
            config: Generation configuration.
            constraint_spec: Optional positional constraints. When provided,
                ETKDG seeding is replaced by seeding from an existing conformer,
                global shake moves are suppressed, and minimization applies MMFF
                position restraints to keep constrained atoms fixed.
        """
        self.constraint_spec = constraint_spec
        self.mol = mol
        self.rotor_model = rotor_model
        self.torsion_lib = torsion_lib
        self.config = config

        self.fast_minimizer = get_minimizer(
            config.minimizer, max_iters=config.fast_minimization_iters, dielectric=config.fast_dielectric
        )
        self.full_minimizer = get_minimizer(
            config.minimizer, max_iters=config.max_minimization_iters, dielectric=config.final_dielectric
        )
        self.fast_minimizer.prepare(mol)
        self.full_minimizer.prepare(mol)

        # Pre-compute preferred angles for each rotor.
        # Store as (angles_array, normalized_weights_array) so _sample_angle
        # can use np.random.choice without recomputing the sum on every call.
        self._rotor_angles: list[tuple[np.ndarray, np.ndarray]] = []
        for rotor in rotor_model.rotors:
            angles, weights = torsion_lib.get_preferred_angles(mol, rotor.dihedral_atoms)
            angles_arr = np.array(angles, dtype=np.float64)
            weights_arr = np.array(weights, dtype=np.float64)
            weights_arr /= weights_arr.sum()
            self._rotor_angles.append((angles_arr, weights_arr))

        # Precompute non-bonded pair mask for fast numpy clash detection.
        self._nonbonded_mask: np.ndarray = _build_nonbonded_mask(mol)
        self._clash_threshold2: float = config.clash_threshold**2

        # Precompute crankshaft data: for each non-aromatic ring of size >= 6,
        # store the ordered ring atom indices and the substituent-atom subtree
        # for each ring position (atoms that should rotate with that ring atom,
        # excluding other same-ring atoms). Rings containing metals are skipped.
        self._crankable_rings: list[tuple[tuple[int, ...], list[frozenset[int]]]] = []
        atom_rings = rotor_model.ring_info.get("ring_atoms", [])
        for ring in atom_rings:
            if len(ring) < 6:
                continue
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                continue
            from ..perceive import _is_metal

            if any(_is_metal(mol.GetAtomWithIdx(i)) for i in ring):
                continue
            ring_set = frozenset(ring)
            subtrees = [self._compute_substituent_atoms(mol, ring_set, i) for i in ring]
            self._crankable_rings.append((tuple(ring), subtrees))

        # Staging mol for batch minimization: same topology, no conformers.
        # Candidates are isolated here to avoid minimizing pool members.
        self._staging_mol: Chem.RWMol = Chem.RWMol(mol)
        self._staging_mol.RemoveAllConformers()
        self._staging_mmff_props = AllChem.MMFFGetMoleculeProperties(self._staging_mol, mmffVariant="MMFF94s")
        if self._staging_mmff_props is not None:
            self._staging_mmff_props.SetMMFFDielectricConstant(config.fast_dielectric)

        # Store reference positions for constrained atoms so we can snap them
        # back to exactly the starting pose after each minimization. The stiff
        # position restraints keep drift tiny, but this eliminates it entirely.
        self._constrained_ref_pos: dict[int, np.ndarray] = {}
        if constraint_spec is not None and mol.GetNumConformers() > 0:
            ref_conf = mol.GetConformer(mol.GetConformers()[0].GetId())
            for idx in constraint_spec.constrained_atoms:
                self._constrained_ref_pos[idx] = np.array(ref_conf.GetAtomPosition(idx))

        # Set random seed
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        # Adaptive move scheduling: base prior (from config) + empirical reward
        # counters that get blended in periodically. Reward is post-dedupe
        # survival: an accepted conformer is tagged with its producing move;
        # at the next dedupe, conformers still in the pool count +1 reward for
        # their move, pruned ones count as 0. This favors novelty over mere
        # acceptance. When adaptive_moves is disabled, counters accumulate but
        # are never read.
        self._base_move_probs: dict[str, float] = dict(config.move_probs)
        self._current_move_probs: dict[str, float] = dict(config.move_probs)
        self._move_attempts: dict[str, float] = dict.fromkeys(config.move_probs, 0.0)
        self._move_rewards: dict[str, float] = dict.fromkeys(config.move_probs, 0.0)
        self._pending_tags: dict[int, str] = {}

    def generate_seeds(self, n_seeds: int) -> list[tuple[int, float]]:
        """Generate seed conformers using ETKDG, then fast-minimize in batch.

        Args:
            n_seeds: Number of seed conformers to generate.

        Returns:
            List of (conf_id, energy_kcal) tuples for successfully embedded seeds.
        """
        params = AllChem.ETKDGv3()
        params.randomSeed = self.config.random_seed or -1
        params.numThreads = int(self.config.num_threads or 0)
        params.pruneRmsThresh = self.config.seed_prune_rms_thresh

        # Enable ring-aware sampling based on what's in the molecule.
        # useSmallRingTorsions: crystallography-derived preferences for 3-7-membered rings.
        # useMacrocycleTorsions: macrocycle-specific distance bounds (≥8-membered).
        ring_info = self.rotor_model.ring_info
        if ring_info.get("has_small_ring"):
            params.useSmallRingTorsions = True
        if ring_info.get("has_macrocycle"):
            params.useMacrocycleTorsions = True
            params.useBasicKnowledge = True
            # Macrocycles have many low-RMSD-distinct puckers; the default
            # 1.0 Å prune threshold collapses them before minimization. Disable
            # seed pruning so the diversity ETKDG generates actually reaches
            # the MMFF stage.
            params.pruneRmsThresh = -1.0

        conf_ids = list(AllChem.EmbedMultipleConfs(self.mol, numConfs=n_seeds, params=params))
        if not conf_ids:
            # ETKDG failed (common for organometallics where distance-bound tables
            # don't cover the metal). Fall back to random starting coordinates and
            # let UFF minimization produce a reasonable geometry.
            params.useRandomCoords = True
            conf_ids = list(AllChem.EmbedMultipleConfs(self.mol, numConfs=n_seeds, params=params))
        if not conf_ids:
            return []

        # Minimize each seed using pre-prepared MMFF props (includes fast_dielectric).
        mmff_props = getattr(self.fast_minimizer, "_mmff_props", None)
        max_its = int(self.fast_minimizer.max_iters)
        nthreads = int(self.config.num_threads or 0)

        if mmff_props is not None:
            energies = minimize_confs_mmff(self.mol, mmff_props, conf_ids, max_its, nthreads)
            return list(zip(conf_ids, energies, strict=True))

        # Fallback: UFF
        AllChem.UFFOptimizeMoleculeConfs(self.mol, numThreads=nthreads, maxIters=max_its)
        return [
            (
                cid,
                float(ff.CalcEnergy())
                if (ff := AllChem.UFFGetMoleculeForceField(self.mol, confId=cid))
                else float("inf"),
            )
            for cid in conf_ids
        ]

    def _reset_constrained_positions(self, mol: Chem.Mol, conf_id: int) -> None:
        """Snap constrained atoms back to their exact reference coordinates.

        Called after every constrained minimization to eliminate any residual
        drift that the position restraints did not fully suppress.

        Args:
            mol: RDKit molecule containing the conformer.
            conf_id: Conformer ID to update in place.
        """
        if not self._constrained_ref_pos:
            return
        conf = mol.GetConformer(conf_id)
        for idx, pos in self._constrained_ref_pos.items():
            conf.SetAtomPosition(idx, pos.tolist())

    def seed_from_conformer(self, conf_id: int) -> list[tuple[int, float]]:
        """Use an existing conformer as the sole seed (constrained mode).

        Fast-minimizes the conformer with position restraints applied to
        constrained atoms so the core stays at the MCS-aligned pose.

        Args:
            conf_id: ID of the starting conformer already present in self.mol.

        Returns:
            List of one (conf_id, energy_kcal) tuple, or empty list on failure.
        """
        energy = self._minimize_constrained(self.mol, conf_id, use_fast=True)
        if not np.isfinite(energy):
            return []
        return [(conf_id, energy)]

    def _minimize_constrained(self, mol: Chem.Mol, conf_id: int, use_fast: bool = True) -> float:
        """Minimize a conformer with MMFF position restraints on constrained atoms.

        Args:
            mol: RDKit molecule containing the conformer.
            conf_id: Conformer ID to minimize in place.
            use_fast: If True, use fast_minimizer iterations; otherwise full.

        Returns:
            Energy in kcal/mol after minimization, or inf on failure.
        """
        assert self.constraint_spec is not None
        minimizer = self.fast_minimizer if use_fast else self.full_minimizer
        props = getattr(minimizer, "_mmff_props", None)

        try:
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(conf_id))
                if ff is None:
                    return float("inf")
                for idx in self.constraint_spec.constrained_atoms:
                    ff.MMFFAddPositionConstraint(idx, 0.0, self.constraint_spec.position_force_constant)
                ff.Minimize(maxIts=int(minimizer.max_iters))
                energy = float(ff.CalcEnergy())
            else:
                # UFF fallback — no position constraints available, minimize freely
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(conf_id))
                if ff is None:
                    return float("inf")
                ff.Minimize(maxIts=int(minimizer.max_iters))
                energy = float(ff.CalcEnergy())
        except (ValueError, RuntimeError):
            return float("inf")

        # Snap constrained atoms back to exact reference coordinates, eliminating
        # any residual drift that the position restraints did not fully suppress.
        self._reset_constrained_positions(mol, conf_id)
        return energy

    def _propose_constrained(self, pool: "ConformerPool", step: int) -> tuple[int, float, str] | None:
        """Propose a single conformer using constrained minimization.

        Args:
            pool: Conformer pool for parent selection.
            step: Current step number.

        Returns:
            Tuple of (conf_id, energy, source) or None if failed.
        """
        result = self._generate_candidate(pool, step)
        if result is None:
            return None
        new_conf_id, move_type = result

        try:
            energy = self._minimize_constrained(self.mol, new_conf_id, use_fast=True)
        except Exception:
            self.mol.RemoveConformer(new_conf_id)
            return None

        if not np.isfinite(energy):
            self.mol.RemoveConformer(new_conf_id)
            return None

        return (new_conf_id, energy, f"hybrid_{move_type}")

    def _sample_angle(self, rotor_idx: int) -> float:
        """Sample an angle from the torsion library.

        Args:
            rotor_idx: Index of the rotor.

        Returns:
            Sampled angle in degrees (with jitter).
        """
        angles_arr, weights_arr = self._rotor_angles[rotor_idx]
        idx = np.random.choice(len(angles_arr), p=weights_arr)
        jitter = np.random.normal(0.0, self.config.torsion_jitter_deg)
        return float(angles_arr[idx]) + jitter

    def _apply_single_rotor_move(self, conf_id: int) -> None:
        """Apply a single rotor move.

        Args:
            conf_id: Conformer ID to modify.
        """
        if not self.rotor_model.rotors:
            return

        # Pick random rotor
        rotor_idx = random.randrange(len(self.rotor_model.rotors))
        rotor = self.rotor_model.rotors[rotor_idx]

        # Sample new angle
        new_angle = self._sample_angle(rotor_idx)

        # Set angle
        _set_dihedral(self.mol, conf_id, rotor.dihedral_atoms, new_angle)

    def _apply_multi_rotor_move(self, conf_id: int, n_rotors: int = 3) -> None:
        """Apply multiple independent rotor moves.

        Args:
            conf_id: Conformer ID to modify.
            n_rotors: Number of rotors to change.
        """
        if not self.rotor_model.rotors:
            return

        n = min(n_rotors, len(self.rotor_model.rotors))
        rotor_indices = random.sample(range(len(self.rotor_model.rotors)), n)

        for rotor_idx in rotor_indices:
            rotor = self.rotor_model.rotors[rotor_idx]
            new_angle = self._sample_angle(rotor_idx)
            _set_dihedral(self.mol, conf_id, rotor.dihedral_atoms, new_angle)

    def _apply_correlated_move(self, conf_id: int) -> None:
        """Apply a correlated move (change rotor and its neighbors).

        Args:
            conf_id: Conformer ID to modify.
        """
        if not self.rotor_model.rotors:
            return

        # Pick a rotor with neighbors
        candidates = [i for i, r in enumerate(self.rotor_model.rotors) if r.neighbors]

        if not candidates:
            # Fall back to single move
            self._apply_single_rotor_move(conf_id)
            return

        center_idx = random.choice(candidates)
        center_rotor = self.rotor_model.rotors[center_idx]

        # Change center and neighbors
        for rotor_idx in [center_idx, *center_rotor.neighbors]:
            rotor = self.rotor_model.rotors[rotor_idx]
            new_angle = self._sample_angle(rotor_idx)
            _set_dihedral(self.mol, conf_id, rotor.dihedral_atoms, new_angle)

    def _apply_global_shake(self, conf_id: int) -> None:
        """Apply a global shake (change many rotors).

        Args:
            conf_id: Conformer ID to modify.
        """
        if not self.rotor_model.rotors:
            return

        # Change 50-80% of rotors
        n_to_change = max(
            1, random.randint(len(self.rotor_model.rotors) // 2, int(len(self.rotor_model.rotors) * 0.8) + 1)
        )

        rotor_indices = random.sample(
            range(len(self.rotor_model.rotors)), min(n_to_change, len(self.rotor_model.rotors))
        )

        for rotor_idx in rotor_indices:
            rotor = self.rotor_model.rotors[rotor_idx]
            new_angle = self._sample_angle(rotor_idx)
            _set_dihedral(self.mol, conf_id, rotor.dihedral_atoms, new_angle)

    @staticmethod
    def _compute_substituent_atoms(mol: Chem.Mol, ring_atoms: frozenset[int], ring_atom: int) -> frozenset[int]:
        """Atoms reachable from ring_atom without re-entering the same ring.

        Returns ring_atom itself plus any attached substituent subtree
        (hydrogens, pendant groups). Other atoms of the same ring are treated
        as barriers and are not traversed, so the returned set is exactly the
        group that should rigidly follow ring_atom during a crankshaft rotation.

        Args:
            mol: RDKit molecule.
            ring_atoms: Indices of all atoms in the ring currently being cranked.
            ring_atom: Ring atom whose substituent tree is sought.

        Returns:
            frozenset of atom indices that move with ring_atom.
        """
        visited: set[int] = {ring_atom}
        stack = [ring_atom]
        while stack:
            cur = stack.pop()
            for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
                j = nb.GetIdx()
                if j in visited:
                    continue
                if j in ring_atoms and j != ring_atom:
                    continue
                visited.add(j)
                stack.append(j)
        return frozenset(visited)

    def _apply_crankshaft_move(self, conf_id: int) -> None:
        """Rotate an arc of ring atoms about the axis through two anchor atoms.

        Picks a ring uniformly from self._crankable_rings, two non-adjacent
        anchors (a_p, a_q) uniformly from the ring, and rotates every atom of
        the shorter inter-anchor arc (plus its substituent subtree) about the
        a_p–a_q axis by a random angle. Because both anchors lie on the axis,
        bond lengths between each anchor and its ring neighbours are preserved
        exactly; only the anchor bond angles deform, and those relax in the
        subsequent MMFF minimization.

        Has no effect when no crankable rings exist.

        Args:
            conf_id: Conformer ID to modify in place.
        """
        if not self._crankable_rings:
            return

        ring_atoms, subtrees = random.choice(self._crankable_rings)
        n = len(ring_atoms)

        # Pick two non-adjacent anchors. Arc length k = number of interior atoms
        # to rotate, drawn from [1, n-3]. For n=6 this is [1, 3]; for n=12, [1, 9].
        # Shorter arcs give smaller local moves; longer arcs reshape the ring globally.
        p = random.randrange(n)
        arc_len = random.randint(1, n - 3)
        # Interior positions are p+1, p+2, ..., p+arc_len (mod n); anchor q is p+arc_len+1.
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

        # Rotation angle: uniform from [30°, 120°] with random sign, plus gaussian jitter.
        # Smaller range than the ±180° limit keeps post-move geometries within reach of
        # MMFF relaxation; larger rotations on dense macrocycles nearly always clash.
        base = random.uniform(30.0, 120.0) * random.choice([-1.0, 1.0])
        theta = np.deg2rad(base + np.random.normal(0.0, self.config.torsion_jitter_deg))

        # Rodrigues' rotation matrix.
        c, s = np.cos(theta), np.sin(theta)
        one_c = 1.0 - c
        ux, uy, uz = axis_unit
        R = np.array(
            [
                [c + ux * ux * one_c, ux * uy * one_c - uz * s, ux * uz * one_c + uy * s],
                [uy * ux * one_c + uz * s, c + uy * uy * one_c, uy * uz * one_c - ux * s],
                [uz * ux * one_c - uy * s, uz * uy * one_c + ux * s, c + uz * uz * one_c],
            ]
        )

        # Collect every atom that moves with the interior arc: each interior
        # ring atom plus its substituent subtree.
        moving: set[int] = set()
        for pos_in_ring in interior:
            moving |= subtrees[pos_in_ring]

        if not moving:
            return

        moving_arr = np.fromiter(moving, dtype=np.int64, count=len(moving))
        pts = all_pos[moving_arr] - axis_origin
        rotated = pts @ R.T + axis_origin

        for atom_idx, new_xyz in zip(moving_arr.tolist(), rotated, strict=True):
            conf.SetAtomPosition(int(atom_idx), new_xyz.tolist())

    def _apply_ring_flip_move(self, conf_id: int) -> None:
        """Apply a ring flip to a randomly chosen non-aromatic ring.

        Reflects each ring atom through the ring's mean plane (computed via
        SVD), effectively inverting chair/envelope conformations. Attached
        non-ring atoms are left in place and relaxed by subsequent minimization.

        Args:
            conf_id: Conformer ID to modify in-place.
        """
        if not self.rotor_model.ring_flips:
            return

        ring_flip = random.choice(self.rotor_model.ring_flips)
        ring_atoms = ring_flip.ring_atoms

        conf = self.mol.GetConformer(conf_id)
        # Fetch all positions once (no per-atom Python calls on the way in).
        all_pos = conf.GetPositions()  # shape (n_atoms, 3), C++ copy
        ring_idx = list(ring_atoms)
        ring_pos = all_pos[ring_idx]  # shape (n_ring, 3)

        # Compute ring centroid and plane normal via SVD.
        # The normal is the right-singular vector with the smallest singular value
        # (direction of least variance = perpendicular to the ring plane).
        centroid = ring_pos.mean(axis=0)
        _, _, vh = np.linalg.svd(ring_pos - centroid)
        normal = vh[-1]  # unit normal to the best-fit plane

        # Vectorized reflection: new_pos = pos - 2 * (dot(pos-centroid, n)) * n
        signed_dists = (ring_pos - centroid) @ normal  # shape (n_ring,)
        reflected = ring_pos - 2.0 * signed_dists[:, None] * normal

        # Write back only the modified ring atoms.
        for atom_idx, new_xyz in zip(ring_idx, reflected, strict=True):
            conf.SetAtomPosition(atom_idx, new_xyz.tolist())

    def _select_move_type(self, step: int) -> str:
        """Select move type based on probabilities and step count.

        Args:
            step: Current step number.

        Returns:
            Move type string.
        """
        # Periodic global shake — suppressed in constrained mode (would thrash
        # the carefully placed starting pose).
        if self.constraint_spec is None and step > 0 and step % self.config.shake_period == 0:
            return "global_shake"

        # Weighted random selection, suppressing unavailable move types.
        probs = dict(self._current_move_probs)

        # Remove global_shake from weighted pool in constrained mode.
        if self.constraint_spec is not None:
            extra = probs.pop("global_shake", 0.0)
            probs["single_rotor"] = probs.get("single_rotor", 0.0) + extra
            # Crankshaft moves the ring backbone, which would drift constrained atoms.
            extra = probs.pop("crankshaft", 0.0)
            probs["single_rotor"] = probs.get("single_rotor", 0.0) + extra

        if not self.rotor_model.ring_flips and "ring_flip" in probs:
            extra = probs.pop("ring_flip")
            # redistribute to single_rotor
            probs["single_rotor"] = probs.get("single_rotor", 0.0) + extra

        if not self._crankable_rings and "crankshaft" in probs:
            extra = probs.pop("crankshaft")
            probs["single_rotor"] = probs.get("single_rotor", 0.0) + extra

        total = sum(probs.values())
        r = random.random() * total
        cumsum = 0.0
        for move_type, prob in probs.items():
            cumsum += prob
            if r <= cumsum:
                return move_type

        return "single_rotor"  # fallback

    def record_accepted(self, conf_id: int, move_type: str) -> None:
        """Tag a newly accepted conformer with the move that produced it.

        The conformer is only scored later, when a dedupe pass decides whether
        it was geometrically novel. Called from the main loop right after a
        successful ``pool.insert`` for a candidate produced by this proposer.

        Args:
            conf_id: RDKit conformer ID of the accepted candidate.
            move_type: Move type that produced it.
        """
        self._pending_tags[conf_id] = move_type

    def record_dedupe_outcome(self, surviving_ids: set[int]) -> None:
        """Score pending tags against which conformers survived the last dedupe.

        For every pending (conf_id, move_type): if conf_id is still in the pool
        after dedupe, credit the move with +1 reward; otherwise credit 0 (but
        still count as an attempt). Pool-overflow eviction and PRISM pruning
        are treated identically — both mean "this pose didn't add value" —
        which is the intended signal for acceptance-vs-novelty.

        Triggers an adaptation step at the end when ``adaptive_moves`` is on,
        so the move probabilities shift in lockstep with the dedupe cycle.

        Args:
            surviving_ids: Conformer IDs still in the pool after dedupe.
        """
        for cid, move_type in self._pending_tags.items():
            self._move_attempts[move_type] = self._move_attempts.get(move_type, 0.0) + 1.0
            if cid in surviving_ids:
                self._move_rewards[move_type] = self._move_rewards.get(move_type, 0.0) + 1.0
        self._pending_tags.clear()

        if self.config.adaptive_moves:
            self._adapt_move_probs()

    def _adapt_move_probs(self) -> None:
        """Recompute ``_current_move_probs`` as a prior+empirical blend.

        For each move type: empirical = rewards / attempts (fallback to base
        prior when unsampled). The learned distribution is the normalized
        empirical rates across the move types present in the base prior. The
        current prob is then a convex combination of base prior and learned
        rates, subject to a per-move floor so no move ever dies and exploration
        continues.
        """
        base = self._base_move_probs
        sampled_rates: list[float] = []
        for m in base:
            att = self._move_attempts.get(m, 0.0)
            if att > 0:
                sampled_rates.append(self._move_rewards.get(m, 0.0) / att)
        neutral = sum(sampled_rates) / len(sampled_rates) if sampled_rates else 1.0

        empirical: dict[str, float] = {}
        for m in base:
            att = self._move_attempts.get(m, 0.0)
            if att > 0:
                empirical[m] = self._move_rewards.get(m, 0.0) / att
            else:
                empirical[m] = neutral

        total = sum(empirical.values())
        if total <= 0:
            learned = dict(base)
        else:
            learned = {m: v / total for m, v in empirical.items()}

        blend = self.config.adapt_blend
        floor = self.config.adapt_floor
        blended = {m: blend * base[m] + (1.0 - blend) * learned[m] for m in base}
        blended = {m: max(floor, v) for m, v in blended.items()}
        total = sum(blended.values())
        self._current_move_probs = {m: v / total for m, v in blended.items()}

        decay = self.config.adapt_decay
        for m in self._move_attempts:
            self._move_attempts[m] *= decay
            self._move_rewards[m] *= decay

    def _generate_candidate(self, pool: ConformerPool, step: int) -> tuple[int, str] | None:
        """Copy a parent, apply a move, optionally clash-check.

        Adds the candidate conformer to self.mol and returns its conf_id.
        Returns None (and cleans up) if parent unavailable or clash detected.

        Args:
            pool: Conformer pool for parent selection.
            step: Current step number (used for move-type selection).

        Returns:
            Tuple of (conf_id, move_type), or None if candidate was rejected.
        """
        parent_id = pool.get_parent(strategy=self.config.parent_strategy)
        if parent_id is None:
            return None

        new_conf_id = _copy_conformer(self.mol, parent_id)
        move_type = self._select_move_type(step)

        if move_type == "single_rotor":
            self._apply_single_rotor_move(new_conf_id)
        elif move_type == "multi_rotor":
            self._apply_multi_rotor_move(new_conf_id)
        elif move_type == "correlated":
            self._apply_correlated_move(new_conf_id)
        elif move_type == "global_shake":
            self._apply_global_shake(new_conf_id)
        elif move_type == "ring_flip":
            self._apply_ring_flip_move(new_conf_id)
        elif move_type == "crankshaft":
            self._apply_crankshaft_move(new_conf_id)

        # Crankshaft and ring_flip legitimately produce strained intermediate
        # geometries (tight H..H in particular) that MMFF relaxes fine. The
        # static clash threshold rejects most of these, so skip it for those
        # move types and rely on the post-minimization energy filter instead.
        clash_exempt = move_type in ("crankshaft", "ring_flip")
        if not self.config.skip_clash_check and not clash_exempt:
            pos = self.mol.GetConformer(new_conf_id).GetPositions()
            diff = pos[:, None, :] - pos[None, :, :]
            dist2 = (diff * diff).sum(axis=-1)
            if bool((dist2[self._nonbonded_mask] < self._clash_threshold2).any()):
                self.mol.RemoveConformer(new_conf_id)
                return None

        return (new_conf_id, move_type)

    def propose(self, pool: ConformerPool, step: int) -> tuple[int, float, str] | None:
        """Propose a single conformer (sequential minimization).

        Args:
            pool: Conformer pool for parent selection.
            step: Current step number.

        Returns:
            Tuple of (conf_id, energy, source) or None if failed.
        """
        result = self._generate_candidate(pool, step)
        if result is None:
            return None
        new_conf_id, move_type = result

        try:
            energy = self.fast_minimizer.minimize(self.mol, new_conf_id)
        except Exception:
            self.mol.RemoveConformer(new_conf_id)
            return None

        if not np.isfinite(energy):
            self.mol.RemoveConformer(new_conf_id)
            return None

        return (new_conf_id, energy, f"hybrid_{move_type}")

    def propose_batch(self, pool: ConformerPool, step: int) -> list[tuple[int, float, str]]:
        """Propose a batch of conformers and minimize them in parallel.

        Generates config.minimize_batch_size candidates on self.mol, copies them
        to an isolated staging mol, minimizes each with MMFF using the pre-prepared
        staging props (fast_dielectric applied), then transfers accepted
        (finite-energy) conformers back to self.mol.

        When constraint_spec is set, falls back to sequential per-conformer
        minimization with MMFF position restraints (MMFFOptimizeMoleculeConfs
        does not support custom force field terms).

        Args:
            pool: Conformer pool for parent selection.
            step: Current step number (used for move-type selection of first item).

        Returns:
            List of (conf_id, energy, source) tuples for accepted conformers.
        """
        # Constrained mode: per-conformer MMFF with position restraints.
        if self.constraint_spec is not None:
            results: list[tuple[int, float, str]] = []
            for i in range(self.config.minimize_batch_size):
                result = self._propose_constrained(pool, step + i)
                if result is not None:
                    results.append(result)
            return results

        batch_size = self.config.minimize_batch_size

        # 1. Generate candidates on self.mol (move applied, clash-checked).
        candidates: list[tuple[int, str]] = []
        for i in range(batch_size):
            result = self._generate_candidate(pool, step + i)
            if result is not None:
                candidates.append(result)

        if not candidates:
            return []

        # 2. Copy each candidate into the staging mol, track main→stage mapping.
        stage_ids: list[int] = []
        for main_id, _ in candidates:
            conf = self.mol.GetConformer(main_id)
            stage_id = self._staging_mol.AddConformer(Chem.Conformer(conf), assignId=True)
            stage_ids.append(stage_id)

        # 3. Minimize staging conformers using pre-prepared props (includes fast_dielectric).
        nthreads = int(self.config.num_threads or 0)
        max_its = int(self.fast_minimizer.max_iters)
        if self._staging_mmff_props is not None:
            energies = minimize_confs_mmff(self._staging_mol, self._staging_mmff_props, stage_ids, max_its, nthreads)
        else:
            AllChem.UFFOptimizeMoleculeConfs(self._staging_mol, numThreads=nthreads, maxIters=max_its)
            energies = [
                float(ff.CalcEnergy())
                if (ff := AllChem.UFFGetMoleculeForceField(self._staging_mol, confId=sid))
                else float("inf")
                for sid in stage_ids
            ]
        stage_energies = dict(zip(stage_ids, energies, strict=True))

        # 4. Transfer accepted conformers back; discard rejected ones.
        results: list[tuple[int, float, str]] = []
        for (main_id, move_type), stage_id in zip(candidates, stage_ids, strict=True):
            # Always remove the unminimized candidate from self.mol.
            self.mol.RemoveConformer(main_id)

            energy = stage_energies[stage_id]

            if np.isfinite(energy):
                # Add the minimized staging conformer to self.mol.
                minimized_conf = self._staging_mol.GetConformer(stage_id)
                final_id = self.mol.AddConformer(Chem.Conformer(minimized_conf), assignId=True)
                results.append((final_id, energy, f"hybrid_{move_type}"))

        # 5. Clear staging mol for next batch.
        self._staging_mol.RemoveAllConformers()

        return results

    def full_refine_final(
        self,
        mol: Chem.Mol,
        final_ids: list[int],
        num_threads: int = 0,
        max_iters: int = 200,
        variant: str = "MMFF94s",
        dielectric: float = 4.0,
    ) -> list[float]:
        """Run full MMFF minimization on the final selected conformers.

        Args:
            mol: RDKit molecule containing the conformers to refine.
            final_ids: Conformer IDs to refine (others are removed).
            num_threads: Number of threads for parallel minimization (UFF fallback only).
            max_iters: Maximum MMFF iterations.
            variant: MMFF variant ("MMFF94" or "MMFF94s").
            dielectric: Dielectric constant for electrostatics.

        Returns:
            List of refined energies in kcal/mol, aligned to final_ids.
        """
        # Keep only finals to avoid optimizing thousands of confs
        final_set = set(final_ids)
        for conf in list(mol.GetConformers()):
            if conf.GetId() not in final_set:
                mol.RemoveConformer(conf.GetId())

        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
        if mmff_props is not None:
            mmff_props.SetMMFFDielectricConstant(dielectric)
            return minimize_confs_mmff(mol, mmff_props, final_ids, max_iters, num_threads)

        # Fallback UFF
        AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=int(num_threads or 0), maxIters=int(max_iters))
        energies = []
        for cid in final_ids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(cid))
            energies.append(ff.CalcEnergy() if ff else float("inf"))
        return energies

    def full_refine_final_constrained(
        self,
        mol: Chem.Mol,
        final_ids: list[int],
        max_iters: int = 200,
        dielectric: float = 4.0,
    ) -> list[float]:
        """Run full MMFF minimization on final conformers with position restraints.

        Used in constrained mode so the core remains pinned to the MCS pose
        during final refinement.

        Args:
            mol: RDKit molecule containing the conformers to refine.
            final_ids: Conformer IDs to refine.
            max_iters: Maximum MMFF iterations.
            dielectric: Dielectric constant for the refinement pass.

        Returns:
            List of refined energies in kcal/mol, aligned to final_ids.
        """
        assert self.constraint_spec is not None

        # Keep only finals before refining
        final_set = set(final_ids)
        for conf in list(mol.GetConformers()):
            if conf.GetId() not in final_set:
                mol.RemoveConformer(conf.GetId())

        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        energies: list[float] = []

        for cid in final_ids:
            try:
                if props is not None:
                    props.SetMMFFDielectricConstant(dielectric)
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(cid))
                    if ff is None:
                        energies.append(float("inf"))
                        continue
                    for idx in self.constraint_spec.constrained_atoms:
                        ff.MMFFAddPositionConstraint(idx, 0.0, self.constraint_spec.position_force_constant)
                    ff.Minimize(maxIts=int(max_iters))
                    energies.append(float(ff.CalcEnergy()))
                else:
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(cid))
                    if ff is None:
                        energies.append(float("inf"))
                        continue
                    ff.Minimize(maxIts=int(max_iters))
                    energies.append(float(ff.CalcEnergy()))
            except (ValueError, RuntimeError):
                energies.append(float("inf"))
                continue
            self._reset_constrained_positions(mol, cid)

        return energies


def run_hybrid_generation(
    mol: Chem.Mol,
    rotor_model: RotorModel,
    config: ConformerConfig,
) -> tuple[Chem.Mol, list[int], list[float]]:
    """Run hybrid conformer generation.

    Args:
        mol: RDKit molecule (will be modified).
        rotor_model: Rotor model.
        config: Generation configuration.

    Returns:
        Tuple of (mol, conf_ids, energies).
    """
    constraint_spec = config.constraint_spec

    # Filter rotors before building the proposer so _rotor_angles is computed
    # only for free rotors.
    if constraint_spec is not None:
        rotor_model = filter_constrained_rotors(rotor_model, constraint_spec.constrained_atoms)

    torsion_lib = TorsionLibrary()
    proposer = HybridProposer(mol, rotor_model, torsion_lib, config, constraint_spec=constraint_spec)
    pool = ConformerPool(mol, config)

    if constraint_spec is not None:
        # Constrained mode: seed from the single starting conformer already in mol.
        existing_ids = [c.GetId() for c in mol.GetConformers()]
        if not existing_ids:
            raise ValueError(
                "Constrained conformer generation requires a starting conformer. "
                "Use generate_conformers_from_pose to supply one."
            )
        seeds = proposer.seed_from_conformer(existing_ids[0])
        seed_source = "seed_pose"
    else:
        # Standard mode: ETKDG seeding.
        n_seeds = (
            config.n_seeds if config.n_seeds is not None else _compute_n_seeds(rotor_model, config.seed_n_per_rotor)
        )
        seeds = proposer.generate_seeds(n_seeds)
        seed_source = "seed_etkdg"

    for conf_id, energy in seeds:
        pool.insert(conf_id, energy, source=seed_source)

    # Run exploration.
    # Batch mode: accumulate minimize_batch_size proposals, minimize in parallel.
    # Sequential mode (batch_size=1): original one-at-a-time behaviour.
    batch_size = config.minimize_batch_size
    step = 0
    while step < config.n_steps:
        if batch_size > 1:
            results = proposer.propose_batch(pool, step)
            step += batch_size
        else:
            result = proposer.propose(pool, step)
            results = [result] if result is not None else []
            step += 1

        for conf_id, energy, source in results:
            accepted = pool.insert(conf_id, energy, source=source)
            if accepted:
                move_type = source.removeprefix("hybrid_") if source.startswith("hybrid_") else source
                proposer.record_accepted(conf_id, move_type)
            else:
                mol.RemoveConformer(conf_id)

        # Periodic dedupe — also the tick where the adaptive scheduler
        # collects survival-based rewards and updates move probabilities.
        if pool.should_dedupe():
            pool.dedupe()
            proposer.record_dedupe_outcome(set(pool.conf_ids))

    # Final selection
    final_ids = pool.select_final()

    # Full refinement on the final set (optional — skip for docking-prep workflows).
    if config.do_final_refine:
        if constraint_spec is not None:
            final_energies = proposer.full_refine_final_constrained(
                mol, final_ids, config.max_minimization_iters, dielectric=config.final_dielectric
            )
        else:
            final_energies = proposer.full_refine_final(
                mol, final_ids, config.num_threads, config.max_minimization_iters, dielectric=config.final_dielectric
            )
    else:
        # Return the fast-minimized energies already stored in the pool.
        energy_map = {cid: (rec.energy_kcal or float("inf")) for cid, rec in pool.records.items()}
        final_energies = [energy_map.get(cid, float("inf")) for cid in final_ids]

    # Clean up: remove non-selected conformers
    all_ids = set(pool.conf_ids)
    final_set = set(final_ids)
    for cid in all_ids - final_set:
        mol.RemoveConformer(cid)

    return mol, final_ids, final_energies
