"""Generation-stats helpers for proposal workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ConformerConfig


def new_generation_stats() -> dict[str, float | int]:
    """Create an empty generation-stats mapping."""
    return {
        "topology_tuned_defaults_applied": 0,
        "requested_n_seeds": 0,
        "effective_seed_n_per_rotor": 0,
        "effective_seed_prune_rms_thresh": 0.0,
        "effective_seed_minimization_iters": 0,
        "effective_seed_budget_scale": 1.0,
        "effective_dedupe_period": 0,
        "effective_minimize_batch_size": 0,
        "n_seed_conformers": 0,
        "n_steps_executed": 0,
        "n_batches": 0,
        "n_candidate_attempts": 0,
        "n_candidates_passed_clash": 0,
        "n_clash_rejections": 0,
        "n_minimization_calls": 0,
        "n_minimization_failures": 0,
        "n_pool_accepts": 0,
        "n_pool_rejections": 0,
        "n_dedupe_calls": 0,
        "n_dedupe_removed": 0,
        "n_final_selected": 0,
        "seed_time_s": 0.0,
        "seed_embedding_time_s": 0.0,
        "seed_minimization_time_s": 0.0,
        "proposal_stage_time_s": 0.0,
        "parent_selection_time_s": 0.0,
        "move_selection_time_s": 0.0,
        "move_apply_time_s": 0.0,
        "clash_check_time_s": 0.0,
        "batch_staging_time_s": 0.0,
        "batch_commit_time_s": 0.0,
        "minimization_time_s": 0.0,
        "dedupe_time_s": 0.0,
        "final_selection_time_s": 0.0,
        "final_refine_time_s": 0.0,
        "total_time_s": 0.0,
    }


def populate_effective_config_stats(
    stats: dict[str, float | int],
    *,
    config: ConformerConfig,
    tuned_defaults_applied: bool,
    seed_prune_rms_thresh: float,
) -> None:
    """Populate common effective-config stats for a generation run."""
    if not stats:
        return

    stats["topology_tuned_defaults_applied"] = int(tuned_defaults_applied)
    stats["effective_seed_n_per_rotor"] = config.seed_n_per_rotor
    stats["effective_seed_prune_rms_thresh"] = seed_prune_rms_thresh
    stats["effective_seed_minimization_iters"] = (
        config.seed_minimization_iters if config.seed_minimization_iters is not None else config.fast_minimization_iters
    )
    stats["effective_dedupe_period"] = config.dedupe_period
    stats["effective_minimize_batch_size"] = config.minimize_batch_size
