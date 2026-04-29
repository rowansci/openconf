"""Hybrid conformer proposal strategy.

Combines torsion library biasing with MCMM-style exploration.
"""

import dataclasses
import random
import time
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ..config import ConformerConfig, ConstraintSpec
from ..perceive import RotorModel, filter_constrained_rotors
from ..pool import ConformerPool
from ..relax import get_minimizer, minimize_confs_mmff
from ..torsionlib import TorsionLibrary, get_default_torsion_library
from ..tuning import (
    get_runtime_tuning,
    resolve_forced_move,
    resolve_move_probabilities,
)
from .candidates import CandidateBatchWorkspace, ClashChecker, build_nonbonded_mask
from .moves import MoveExecutor
from .seeding import (
    SeedPlan,
    _compute_n_seeds,
    _is_large_flexible_non_macrocyclic,
    _resolve_seed_prune_rms_thresh,
    populate_seed_plan_stats,
    resolve_seed_plan,
)
from .stats import GenerationStat, new_generation_stats, populate_effective_config_stats

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "HybridProposer",
    "SeedPlan",
    "_compute_n_seeds",
    "resolve_seed_plan",
    "run_hybrid_generation",
    "run_low_flex_generation",
]

_TORSION_MOVE_TYPES = frozenset({"single_rotor", "multi_rotor", "correlated", "global_shake"})


def _resolve_runtime_tuned_config(config: ConformerConfig, rotor_model: RotorModel) -> tuple[ConformerConfig, bool]:
    """Adjust default-equivalent runtime knobs for large flexible molecules."""
    tuning = get_runtime_tuning()
    defaults = tuning.large_flexible.defaults
    tuned = tuning.large_flexible.tuned
    if not config.auto_tune_large_flexible:
        return config, False
    if not _is_large_flexible_non_macrocyclic(config, rotor_model):
        return config, False

    overrides: dict[str, int | bool | None] = {}
    if config.n_seeds is None and config.seed_n_per_rotor == defaults.seed_n_per_rotor:
        overrides["seed_n_per_rotor"] = tuned.seed_n_per_rotor
    if config.dedupe_period == defaults.dedupe_period:
        overrides["dedupe_period"] = tuned.dedupe_period
    if config.minimize_batch_size == defaults.minimize_batch_size and config.num_threads != 1:
        overrides["minimize_batch_size"] = tuned.minimize_batch_size
    if config.topology_aware_seed_pruning is defaults.topology_aware_seed_pruning:
        overrides["topology_aware_seed_pruning"] = tuned.topology_aware_seed_pruning
    if config.topology_aware_seed_budget is defaults.topology_aware_seed_budget:
        overrides["topology_aware_seed_budget"] = tuned.topology_aware_seed_budget

    if not overrides:
        return config, False
    return dataclasses.replace(config, **overrides), True


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
        stats: dict[str, GenerationStat] | None = None,
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
            stats: Optional benchmark timing/counter mapping updated in place.
        """
        self.constraint_spec = constraint_spec
        self.mol = mol
        self.rotor_model = rotor_model
        self.torsion_lib = torsion_lib
        self.config = config
        self.stats = stats

        from ..perceive import _is_metal

        self._metal_atom_indices: frozenset[int] = frozenset(a.GetIdx() for a in mol.GetAtoms() if _is_metal(a))

        self.fast_minimizer = get_minimizer(
            config.minimizer,
            max_iters=config.fast_minimization_iters,
            dielectric=config.fast_dielectric,
            metal_atom_indices=self._metal_atom_indices,
        )
        self.full_minimizer = get_minimizer(
            config.minimizer,
            max_iters=config.max_minimization_iters,
            dielectric=config.final_dielectric,
            metal_atom_indices=self._metal_atom_indices,
        )
        self.fast_minimizer.prepare(mol)
        self.full_minimizer.prepare(mol)

        self._moves = MoveExecutor(mol, rotor_model, torsion_lib, config)
        self._rotor_angles = self._moves._rotor_angles
        self._correlated_rotor_indices = self._moves.correlated_rotor_indices
        self._crankable_rings = self._moves.crankable_rings

        # Precompute non-bonded pair mask for fast numpy clash detection.
        self._nonbonded_mask: np.ndarray = build_nonbonded_mask(mol)
        self._clash_threshold2: float = config.clash_threshold**2
        self._clash_checker = ClashChecker(mol, self._nonbonded_mask, self._clash_threshold2)

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
        self._move_operators: dict[str, Callable[[int], None]] = {
            "single_rotor": self._apply_single_rotor_move,
            "multi_rotor": self._apply_multi_rotor_move,
            "correlated": self._apply_correlated_move,
            "global_shake": self._apply_global_shake,
            "ring_flip": self._apply_ring_flip_move,
            "crankshaft": self._apply_crankshaft_move,
        }

    def _increment_stat(self, key: str, amount: int = 1) -> None:
        """Increment an integer stat when instrumentation is enabled."""
        if self.stats is not None:
            self.stats[key] = int(self.stats.get(key, 0)) + amount

    def _add_time_stat(self, key: str, elapsed_s: float) -> None:
        """Accumulate a timing stat when instrumentation is enabled."""
        if self.stats is not None:
            self.stats[key] = float(self.stats.get(key, 0.0)) + elapsed_s

    def generate_seeds(self, n_seeds: int, prune_rms_thresh: float | None = None) -> list[tuple[int, float]]:
        """Generate seed conformers using ETKDG, then fast-minimize in batch.

        Args:
            n_seeds: Number of seed conformers to generate.
            prune_rms_thresh: Optional pre-resolved ETKDG prune threshold.

        Returns:
            List of (conf_id, energy_kcal) tuples for successfully embedded seeds.
        """
        params = AllChem.ETKDGv3()
        params.randomSeed = self.config.random_seed or -1
        params.numThreads = int(self.config.num_threads or 0)
        params.pruneRmsThresh = (
            prune_rms_thresh
            if prune_rms_thresh is not None
            else _resolve_seed_prune_rms_thresh(self.mol, self.rotor_model, self.config)
        )

        # Enable ring-aware sampling based on what's in the molecule.
        # useSmallRingTorsions: crystallography-derived preferences for 3-7-membered rings.
        # useMacrocycleTorsions: macrocycle-specific distance bounds (≥8-membered).
        ring_info = self.rotor_model.ring_info
        macrocycle_tuning = get_runtime_tuning().macrocycle_seeding
        if ring_info.get("has_small_ring"):
            params.useSmallRingTorsions = True
        if ring_info.get("has_macrocycle"):
            if macrocycle_tuning.use_macrocycle_torsions:
                params.useMacrocycleTorsions = True
            if macrocycle_tuning.use_basic_knowledge:
                params.useBasicKnowledge = True
            # Macrocycles have many low-RMSD-distinct puckers; the default
            # 1.0 Å prune threshold collapses them before minimization. Disable
            # seed pruning so the diversity ETKDG generates actually reaches
            # the MMFF stage.
            if macrocycle_tuning.disable_prune_rms:
                params.pruneRmsThresh = -1.0

        # Pin metal centers to their reference geometry during ETKDG so that
        # atoms without RDKit distance-bound tables (lanthanides, actinides, …)
        # do not land in random positions after embedding.
        if self._metal_atom_indices and self.mol.GetNumConformers() > 0:
            from rdkit.Geometry import rdGeometry

            ref_conf = self.mol.GetConformers()[0]
            params.SetCoordMap(
                {int(idx): rdGeometry.Point3D(*ref_conf.GetAtomPosition(idx)) for idx in self._metal_atom_indices}
            )

        # EmbedMultipleConfs clears all existing conformers even on failure, so
        # snapshot the reference geometry now for the final fallback below.
        _ref_positions: np.ndarray | None = (
            self.mol.GetConformers()[0].GetPositions().copy() if self.mol.GetNumConformers() > 0 else None
        )

        embed_start = time.perf_counter()
        try:
            conf_ids = list(AllChem.EmbedMultipleConfs(self.mol, numConfs=n_seeds, params=params))
        except RuntimeError:
            conf_ids = []
        if not conf_ids:
            # ETKDG failed (common for organometallics where distance-bound tables
            # don't cover the metal). Fall back to random starting coordinates and
            # let UFF minimization produce a reasonable geometry.
            params.useRandomCoords = True
            try:
                conf_ids = list(AllChem.EmbedMultipleConfs(self.mol, numConfs=n_seeds, params=params))
            except RuntimeError:
                conf_ids = []
        self._add_time_stat("seed_embedding_time_s", time.perf_counter() - embed_start)
        if not conf_ids:
            # Both ETKDG attempts failed (e.g. sandwich complexes like ferrocene whose
            # η5/η6 bonds cause bad distance-bound triangle inequalities). Restore the
            # saved reference geometry as the single seed for MCMM exploration.
            if _ref_positions is not None:
                new_conf = Chem.Conformer(self.mol.GetNumAtoms())
                for i, pos in enumerate(_ref_positions):
                    new_conf.SetAtomPosition(i, pos.tolist())
                conf_ids = [self.mol.AddConformer(new_conf, assignId=True)]
            else:
                return []

        # Minimize each seed using pre-prepared MMFF props (includes fast_dielectric).
        mmff_props = getattr(self.fast_minimizer, "_mmff_props", None)
        max_its = int(
            self.config.seed_minimization_iters
            if self.config.seed_minimization_iters is not None
            else self.fast_minimizer.max_iters
        )
        nthreads = int(self.config.num_threads or 0)

        if mmff_props is not None:
            minimize_start = time.perf_counter()
            energies = minimize_confs_mmff(self.mol, mmff_props, conf_ids, max_its, nthreads)
            self._add_time_stat("seed_minimization_time_s", time.perf_counter() - minimize_start)
            return list(zip(conf_ids, energies, strict=True))

        # Fallback: UFF — per-conformer loop so we can add metal position constraints.
        minimize_start = time.perf_counter()
        results: list[tuple[int, float]] = []
        for cid in conf_ids:
            ff = AllChem.UFFGetMoleculeForceField(self.mol, confId=cid)
            if ff is None:
                results.append((cid, float("inf")))
                continue
            for m_idx in self._metal_atom_indices:
                ff.UFFAddPositionConstraint(int(m_idx), 0.0, 1e4)
                for nb in self.mol.GetAtomWithIdx(m_idx).GetNeighbors():
                    ff.UFFAddDistanceConstraint(int(m_idx), int(nb.GetIdx()), False, 2.0, 3.5, 500.0)
            try:
                ff.Minimize(maxIts=max_its)
                results.append((cid, float(ff.CalcEnergy())))
            except (ValueError, RuntimeError):
                results.append((cid, float("inf")))
        self._add_time_stat("seed_minimization_time_s", time.perf_counter() - minimize_start)
        return results

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

        energy = self._minimize_constrained(self.mol, new_conf_id, use_fast=True)
        if not np.isfinite(energy):
            self.mol.RemoveConformer(new_conf_id)
            return None

        return (new_conf_id, energy, f"hybrid_{move_type}")

    def _sample_angle(self, rotor_idx: int) -> float:
        """Sample an angle from the torsion library."""
        return self._moves.sample_angle(rotor_idx)

    def _apply_single_rotor_move(self, conf_id: int) -> None:
        """Apply a single rotor move."""
        self._moves.apply_single_rotor_move(conf_id)

    def _apply_multi_rotor_move(self, conf_id: int, n_rotors: int = 3) -> None:
        """Apply multiple independent rotor moves."""
        self._moves.apply_multi_rotor_move(conf_id, n_rotors=n_rotors)

    def _apply_correlated_move(self, conf_id: int) -> None:
        """Apply a correlated move (change rotor and its neighbors)."""
        self._moves.apply_correlated_move(conf_id)

    def _apply_global_shake(self, conf_id: int) -> None:
        """Apply a global shake (change many rotors)."""
        self._moves.apply_global_shake(conf_id)

    def _apply_crankshaft_move(self, conf_id: int) -> None:
        """Rotate an arc of ring atoms about the axis through two anchor atoms."""
        self._moves.apply_crankshaft_move(conf_id)

    def _apply_ring_flip_move(self, conf_id: int) -> None:
        """Apply a ring flip to a randomly chosen non-aromatic ring."""
        self._moves.apply_ring_flip_move(conf_id)

    def _select_move_type(self, step: int) -> str:
        """Select move type based on probabilities and step count.

        Args:
            step: Current step number.

        Returns:
            Move type string.
        """
        forced = resolve_forced_move(
            step,
            self.config.shake_period,
            constrained=self.constraint_spec is not None,
        )
        if forced is not None:
            return forced

        probs = resolve_move_probabilities(
            self._current_move_probs,
            constrained=self.constraint_spec is not None,
            has_ring_flips=bool(self.rotor_model.ring_flips),
            has_crankshaft=bool(self._crankable_rings),
        )

        total = sum(probs.values())
        r = random.random() * total
        cumsum = 0.0
        for move_type, prob in probs.items():
            cumsum += prob
            if r <= cumsum:
                return move_type

        return "single_rotor"  # fallback

    def _use_torsion_multitry(self, move_type: str) -> bool:
        """Return whether move should use clash-aware multi-try staging."""
        return (
            move_type in _TORSION_MOVE_TYPES
            and self.config.torsion_multitry_attempts > 1
            and not self.config.skip_clash_check
        )

    def _generate_multitry_candidate(self, parent_id: int, move_type: str) -> int:
        """Generate several torsion candidates and keep lowest clash score.

        Args:
            parent_id: Parent conformer ID.
            move_type: Torsion move type to apply.

        Returns:
            Conformer ID for best trial.
        """
        best_conf_id: int | None = None
        best_score = float("inf")
        n_trials = 0

        for _ in range(self.config.torsion_multitry_attempts):
            n_trials += 1
            trial_conf_id = _copy_conformer(self.mol, parent_id)

            move_apply_start = time.perf_counter()
            self._move_operators[move_type](trial_conf_id)
            self._add_time_stat("move_apply_time_s", time.perf_counter() - move_apply_start)

            clash_start = time.perf_counter()
            score = self._clash_checker.clash_score(trial_conf_id)
            self._add_time_stat("clash_check_time_s", time.perf_counter() - clash_start)

            if score < best_score:
                if best_conf_id is not None:
                    self.mol.RemoveConformer(best_conf_id)
                best_conf_id = trial_conf_id
                best_score = score
            else:
                self.mol.RemoveConformer(trial_conf_id)

            if score <= 0.0:
                break

        self._increment_stat("n_torsion_multitry_trials", n_trials)
        assert best_conf_id is not None
        return best_conf_id

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
        parent_start = time.perf_counter()
        parent_id = pool.get_parent(strategy=self.config.parent_strategy)
        self._add_time_stat("parent_selection_time_s", time.perf_counter() - parent_start)
        if parent_id is None:
            return None

        self._increment_stat("n_candidate_attempts")
        move_select_start = time.perf_counter()
        move_type = self._select_move_type(step)
        self._add_time_stat("move_selection_time_s", time.perf_counter() - move_select_start)

        if self._use_torsion_multitry(move_type):
            new_conf_id = self._generate_multitry_candidate(parent_id, move_type)
        else:
            new_conf_id = _copy_conformer(self.mol, parent_id)
            move_apply_start = time.perf_counter()
            self._move_operators[move_type](new_conf_id)
            self._add_time_stat("move_apply_time_s", time.perf_counter() - move_apply_start)

        # Some ring-centric moves produce strained intermediates that MMFF
        # relaxes successfully; the clash filter would reject many of them
        # prematurely, so the policy may exempt those move types.
        clash_start = time.perf_counter()
        if self._clash_checker.has_clash(
            new_conf_id,
            move_type,
            skip_check=self.config.skip_clash_check,
        ):
            self._add_time_stat("clash_check_time_s", time.perf_counter() - clash_start)
            self._increment_stat("n_clash_rejections")
            self.mol.RemoveConformer(new_conf_id)
            return None
        self._add_time_stat("clash_check_time_s", time.perf_counter() - clash_start)

        self._increment_stat("n_candidates_passed_clash")
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

        self._increment_stat("n_minimization_calls")
        minimize_start = time.perf_counter()
        energy = self.fast_minimizer.minimize(self.mol, new_conf_id)
        self._add_time_stat("minimization_time_s", time.perf_counter() - minimize_start)
        if not np.isfinite(energy):
            self._increment_stat("n_minimization_failures")
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

        staging_start = time.perf_counter()
        batch = CandidateBatchWorkspace.from_candidates(self.mol, self._staging_mol, candidates)
        self._add_time_stat("batch_staging_time_s", time.perf_counter() - staging_start)
        stage_ids = batch.stage_ids

        # 2. Minimize staging conformers using pre-prepared props (includes fast_dielectric).
        nthreads = int(self.config.num_threads or 0)
        max_its = int(self.fast_minimizer.max_iters)
        self._increment_stat("n_minimization_calls", len(stage_ids))
        minimize_start = time.perf_counter()
        if self._staging_mmff_props is not None:
            energies = minimize_confs_mmff(
                self._staging_mol,
                self._staging_mmff_props,
                stage_ids,
                max_its,
                nthreads,
            )
        else:
            energies = []
            for stage_id in stage_ids:
                ff = AllChem.UFFGetMoleculeForceField(self._staging_mol, confId=stage_id)
                if ff is None:
                    energies.append(float("inf"))
                    continue
                for m_idx in self._metal_atom_indices:
                    ff.UFFAddPositionConstraint(int(m_idx), 0.0, 1e4)
                    for nb in self._staging_mol.GetAtomWithIdx(m_idx).GetNeighbors():
                        ff.UFFAddDistanceConstraint(int(m_idx), int(nb.GetIdx()), False, 2.0, 3.5, 500.0)
                try:
                    ff.Minimize(maxIts=max_its)
                    energies.append(float(ff.CalcEnergy()))
                except (ValueError, RuntimeError):
                    energies.append(float("inf"))
        self._add_time_stat("minimization_time_s", time.perf_counter() - minimize_start)
        commit_start = time.perf_counter()
        results, failures = batch.commit(energies)
        self._add_time_stat("batch_commit_time_s", time.perf_counter() - commit_start)
        if failures:
            self._increment_stat("n_minimization_failures", failures)
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

        # Fallback UFF — per-conformer loop with metal position and M-L distance constraints.
        energies = []
        for cid in final_ids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(cid))
            if ff is None:
                energies.append(float("inf"))
                continue
            for m_idx in self._metal_atom_indices:
                ff.UFFAddPositionConstraint(int(m_idx), 0.0, 1e4)
                for nb in mol.GetAtomWithIdx(m_idx).GetNeighbors():
                    ff.UFFAddDistanceConstraint(int(m_idx), int(nb.GetIdx()), False, 2.0, 3.5, 500.0)
            try:
                ff.Minimize(maxIts=int(max_iters))
                energies.append(float(ff.CalcEnergy()))
            except (ValueError, RuntimeError):
                energies.append(float("inf"))
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
    torsion_library: TorsionLibrary | None = None,
) -> tuple[Chem.Mol, list[int], list[float], dict[str, GenerationStat]]:
    """Run hybrid conformer generation.

    Args:
        mol: RDKit molecule (will be modified).
        rotor_model: Rotor model.
        config: Generation configuration.
        torsion_library: Optional torsion library override.

    Returns:
        Tuple of (mol, conf_ids, energies, generation_stats).
    """
    total_start = time.perf_counter()
    stats = new_generation_stats() if config.collect_stats else {}
    constraint_spec = config.constraint_spec

    # Filter rotors before building the proposer so _rotor_angles is computed
    # only for free rotors.
    if constraint_spec is not None:
        rotor_model = filter_constrained_rotors(rotor_model, constraint_spec.constrained_atoms)

    effective_config, tuned_defaults_applied = _resolve_runtime_tuned_config(config, rotor_model)
    seed_plan = (
        SeedPlan(
            n_seeds=1,
            base_n_seeds=1,
            budget_scale=1.0,
            budget_floor=1,
            prune_rms_thresh=_resolve_seed_prune_rms_thresh(mol, rotor_model, effective_config),
            reason="seed_pose",
        )
        if constraint_spec is not None
        else resolve_seed_plan(mol, rotor_model, effective_config)
    )
    populate_effective_config_stats(
        stats,
        config=effective_config,
        tuned_defaults_applied=tuned_defaults_applied,
        seed_prune_rms_thresh=seed_plan.prune_rms_thresh,
    )
    populate_seed_plan_stats(stats, seed_plan)

    torsion_lib = torsion_library if torsion_library is not None else get_default_torsion_library()
    proposer = HybridProposer(
        mol, rotor_model, torsion_lib, effective_config, constraint_spec=constraint_spec, stats=stats or None
    )
    pool = ConformerPool(mol, effective_config)

    seed_start = time.perf_counter()
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
        seeds = proposer.generate_seeds(seed_plan.n_seeds, prune_rms_thresh=seed_plan.prune_rms_thresh)
        seed_source = "seed_etkdg"
    if stats:
        stats["seed_time_s"] = time.perf_counter() - seed_start
        stats["n_seed_conformers"] = len(seeds)

    for conf_id, energy in seeds:
        pool.insert(conf_id, energy, source=seed_source)

    # Run exploration.
    # Batch mode: accumulate minimize_batch_size proposals, minimize in parallel.
    # Sequential mode (batch_size=1): original one-at-a-time behaviour.
    batch_size = effective_config.minimize_batch_size
    step = 0
    stagnation = 0
    while step < effective_config.n_steps:
        proposal_start = time.perf_counter()
        if batch_size > 1:
            results = proposer.propose_batch(pool, step)
            step_inc = batch_size
        else:
            result = proposer.propose(pool, step)
            results = [result] if result is not None else []
            step_inc = 1
        if stats:
            stats["proposal_stage_time_s"] = float(stats["proposal_stage_time_s"]) + (
                time.perf_counter() - proposal_start
            )
            stats["n_batches"] = int(stats["n_batches"]) + 1
        step += step_inc

        any_accepted = False
        for conf_id, energy, source in results:
            accepted = pool.insert(conf_id, energy, source=source)
            if accepted:
                any_accepted = True
                if stats:
                    stats["n_pool_accepts"] = int(stats["n_pool_accepts"]) + 1
                move_type = source.removeprefix("hybrid_") if source.startswith("hybrid_") else source
                proposer.record_accepted(conf_id, move_type)
            else:
                if stats:
                    stats["n_pool_rejections"] = int(stats["n_pool_rejections"]) + 1
                mol.RemoveConformer(conf_id)

        if any_accepted:
            stagnation = 0
        else:
            stagnation += step_inc
        if effective_config.patience > 0 and stagnation >= effective_config.patience:
            break

        # Periodic dedupe — also the tick where the adaptive scheduler
        # collects survival-based rewards and updates move probabilities.
        if pool.should_dedupe():
            dedupe_start = time.perf_counter()
            removed = pool.dedupe()
            if stats:
                stats["dedupe_time_s"] = float(stats["dedupe_time_s"]) + (time.perf_counter() - dedupe_start)
                stats["n_dedupe_calls"] = int(stats["n_dedupe_calls"]) + 1
                stats["n_dedupe_removed"] = int(stats["n_dedupe_removed"]) + removed
            proposer.record_dedupe_outcome(set(pool.conf_ids))

    # Final selection
    final_selection_start = time.perf_counter()
    final_ids = pool.select_final()
    if stats:
        stats["final_selection_time_s"] = time.perf_counter() - final_selection_start
        stats["n_final_selected"] = len(final_ids)
        stats["n_steps_executed"] = step

    # Full refinement on the final set (optional — skip for docking-prep workflows).
    if effective_config.do_final_refine:
        final_refine_start = time.perf_counter()
        if constraint_spec is not None:
            final_energies = proposer.full_refine_final_constrained(
                mol, final_ids, effective_config.max_minimization_iters, dielectric=effective_config.final_dielectric
            )
        else:
            final_energies = proposer.full_refine_final(
                mol,
                final_ids,
                effective_config.num_threads,
                effective_config.max_minimization_iters,
                dielectric=effective_config.final_dielectric,
            )
        if stats:
            stats["final_refine_time_s"] = time.perf_counter() - final_refine_start
    else:
        # Return the fast-minimized energies already stored in the pool.
        energy_map = {
            cid: (rec.energy_kcal if rec.energy_kcal is not None else float("inf")) for cid, rec in pool.records.items()
        }
        final_energies = [energy_map.get(cid, float("inf")) for cid in final_ids]

    # Clean up: remove non-selected conformers
    all_ids = set(pool.conf_ids)
    final_set = set(final_ids)
    for cid in all_ids - final_set:
        mol.RemoveConformer(cid)

    if stats:
        stats["total_time_s"] = time.perf_counter() - total_start

    return mol, final_ids, final_energies, stats


def run_low_flex_generation(
    mol: Chem.Mol,
    rotor_model: RotorModel,
    config: ConformerConfig,
    torsion_library: TorsionLibrary | None = None,
) -> tuple[Chem.Mol, list[int], list[float], dict[str, GenerationStat]]:
    """Run the ETKDG-only low-flexibility fast path.

    This path is intended for simple unconstrained molecules where torsional
    MC exploration adds little value over dense ETKDG seeding. It reuses the
    existing seed generation, pool selection, and final refinement logic while
    skipping the proposal loop entirely.

    Args:
        mol: RDKit molecule (will be modified).
        rotor_model: Rotor model.
        config: Generation configuration.
        torsion_library: Optional torsion library override.

    Returns:
        Tuple of (mol, conf_ids, energies, generation_stats).
    """
    total_start = time.perf_counter()
    stats = new_generation_stats() if config.collect_stats else {}
    seed_plan = resolve_seed_plan(mol, rotor_model, config)

    populate_effective_config_stats(
        stats,
        config=config,
        tuned_defaults_applied=False,
        seed_prune_rms_thresh=seed_plan.prune_rms_thresh,
    )
    populate_seed_plan_stats(stats, seed_plan)

    torsion_lib = torsion_library if torsion_library is not None else get_default_torsion_library()
    proposer = HybridProposer(mol, rotor_model, torsion_lib, config, stats=stats or None)
    pool = ConformerPool(mol, config)

    seed_start = time.perf_counter()
    seeds = proposer.generate_seeds(seed_plan.n_seeds, prune_rms_thresh=seed_plan.prune_rms_thresh)
    if stats:
        stats["seed_time_s"] = time.perf_counter() - seed_start
        stats["n_seed_conformers"] = len(seeds)

    for conf_id, energy in seeds:
        accepted = pool.insert(conf_id, energy, source="seed_etkdg")
        if accepted:
            if stats:
                stats["n_pool_accepts"] = int(stats["n_pool_accepts"]) + 1
        else:
            if stats:
                stats["n_pool_rejections"] = int(stats["n_pool_rejections"]) + 1
            mol.RemoveConformer(conf_id)

    final_selection_start = time.perf_counter()
    final_ids = pool.select_final()
    if stats:
        stats["final_selection_time_s"] = time.perf_counter() - final_selection_start
        stats["n_final_selected"] = len(final_ids)
        stats["n_steps_executed"] = 0

    if config.do_final_refine:
        final_refine_start = time.perf_counter()
        final_energies = proposer.full_refine_final(
            mol,
            final_ids,
            config.num_threads,
            config.max_minimization_iters,
            dielectric=config.final_dielectric,
        )
        if stats:
            stats["final_refine_time_s"] = time.perf_counter() - final_refine_start
    else:
        energy_map = {
            cid: (rec.energy_kcal if rec.energy_kcal is not None else float("inf")) for cid, rec in pool.records.items()
        }
        final_energies = [energy_map.get(cid, float("inf")) for cid in final_ids]

    all_ids = set(pool.conf_ids)
    final_set = set(final_ids)
    for cid in all_ids - final_set:
        mol.RemoveConformer(cid)

    if stats:
        stats["total_time_s"] = time.perf_counter() - total_start

    return mol, final_ids, final_energies, stats
