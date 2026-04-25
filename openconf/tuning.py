"""Bundled runtime-tuning data for conformer generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib.resources import files


@dataclass(frozen=True)
class RuntimeTuningDefaults:
    """Default-valued knobs that can be auto-tuned at runtime."""

    seed_n_per_rotor: int
    dedupe_period: int
    minimize_batch_size: int
    topology_aware_seed_pruning: bool | None
    topology_aware_seed_budget: bool | None


@dataclass(frozen=True)
class RuntimeTuningThresholds:
    """Topology-aware scalar adjustments for large flexible molecules."""

    flexible: float
    acyclic: float
    hydrocarbon: float


@dataclass(frozen=True)
class LargeFlexibleRuntimeTuning:
    """Runtime tuning profile for large non-macrocyclic flexible molecules."""

    rotatable_threshold: int
    defaults: RuntimeTuningDefaults
    tuned: RuntimeTuningDefaults
    prune_thresholds: RuntimeTuningThresholds
    seed_scales: RuntimeTuningThresholds


@dataclass(frozen=True)
class MacrocycleSeedingTuning:
    """ETKDG seeding tweaks for macrocyclic molecules."""

    disable_prune_rms: bool
    use_macrocycle_torsions: bool
    use_basic_knowledge: bool


@dataclass(frozen=True)
class MoveSchedulingTuning:
    """Move-scheduling defaults and fallback rules."""

    default_move_probs: dict[str, float]
    forced_periodic_move: str
    constrained_fallback_move: str
    constrained_disabled_moves: frozenset[str]
    availability_fallbacks: dict[str, str]


@dataclass(frozen=True)
class ClashCheckTuning:
    """Clash-check exemptions for specific move types."""

    exempt_moves: frozenset[str]


@dataclass(frozen=True)
class LowFlexPathTuning:
    """Settings for the ETKDG-only low-flexibility fast path."""

    max_rotatable: int
    allow_macrocycles: bool
    allow_rings: bool


@dataclass(frozen=True)
class RuntimeTuning:
    """Complete bundled runtime-tuning settings."""

    large_flexible: LargeFlexibleRuntimeTuning
    macrocycle_seeding: MacrocycleSeedingTuning
    move_scheduling: MoveSchedulingTuning
    clash_check: ClashCheckTuning
    low_flex_path: LowFlexPathTuning


def _load_runtime_tuning() -> RuntimeTuning:
    """Load bundled runtime-tuning settings from JSON."""
    data_path = files("openconf.data").joinpath("runtime_tuning.json")
    with data_path.open() as f:
        raw = json.load(f)

    large_flexible = raw["large_flexible"]
    defaults = large_flexible["defaults"]
    tuned = large_flexible["tuned"]
    topology_aware = large_flexible["topology_aware"]

    return RuntimeTuning(
        large_flexible=LargeFlexibleRuntimeTuning(
            rotatable_threshold=int(large_flexible["rotatable_threshold"]),
            defaults=RuntimeTuningDefaults(
                seed_n_per_rotor=int(defaults["seed_n_per_rotor"]),
                dedupe_period=int(defaults["dedupe_period"]),
                minimize_batch_size=int(defaults["minimize_batch_size"]),
                topology_aware_seed_pruning=defaults["topology_aware_seed_pruning"],
                topology_aware_seed_budget=defaults["topology_aware_seed_budget"],
            ),
            tuned=RuntimeTuningDefaults(
                seed_n_per_rotor=int(tuned["seed_n_per_rotor"]),
                dedupe_period=int(tuned["dedupe_period"]),
                minimize_batch_size=int(tuned["minimize_batch_size"]),
                topology_aware_seed_pruning=bool(tuned["topology_aware_seed_pruning"]),
                topology_aware_seed_budget=bool(tuned["topology_aware_seed_budget"]),
            ),
            prune_thresholds=RuntimeTuningThresholds(
                flexible=float(topology_aware["prune_thresholds"]["flexible"]),
                acyclic=float(topology_aware["prune_thresholds"]["acyclic"]),
                hydrocarbon=float(topology_aware["prune_thresholds"]["hydrocarbon"]),
            ),
            seed_scales=RuntimeTuningThresholds(
                flexible=float(topology_aware["seed_scales"]["flexible"]),
                acyclic=float(topology_aware["seed_scales"]["acyclic"]),
                hydrocarbon=float(topology_aware["seed_scales"]["hydrocarbon"]),
            ),
        ),
        macrocycle_seeding=MacrocycleSeedingTuning(
            disable_prune_rms=bool(raw["macrocycle_seeding"]["disable_prune_rms"]),
            use_macrocycle_torsions=bool(raw["macrocycle_seeding"]["use_macrocycle_torsions"]),
            use_basic_knowledge=bool(raw["macrocycle_seeding"]["use_basic_knowledge"]),
        ),
        move_scheduling=MoveSchedulingTuning(
            default_move_probs={k: float(v) for k, v in raw["move_scheduling"]["default_move_probs"].items()},
            forced_periodic_move=str(raw["move_scheduling"]["forced_periodic_move"]),
            constrained_fallback_move=str(raw["move_scheduling"]["constrained_fallback_move"]),
            constrained_disabled_moves=frozenset(raw["move_scheduling"]["constrained_disabled_moves"]),
            availability_fallbacks={
                str(k): str(v) for k, v in raw["move_scheduling"]["availability_fallbacks"].items()
            },
        ),
        clash_check=ClashCheckTuning(exempt_moves=frozenset(raw["clash_check"]["exempt_moves"])),
        low_flex_path=LowFlexPathTuning(
            max_rotatable=int(raw["low_flex_path"]["max_rotatable"]),
            allow_macrocycles=bool(raw["low_flex_path"]["allow_macrocycles"]),
            allow_rings=bool(raw["low_flex_path"]["allow_rings"]),
        ),
    )


@cache
def get_runtime_tuning() -> RuntimeTuning:
    """Return the cached bundled runtime-tuning settings."""
    return _load_runtime_tuning()


def get_default_move_probs() -> dict[str, float]:
    """Return a fresh copy of the default move probabilities."""
    return dict(get_runtime_tuning().move_scheduling.default_move_probs)


def is_clash_exempt_move(move_type: str) -> bool:
    """Return whether a move type should bypass the clash filter."""
    return move_type in get_runtime_tuning().clash_check.exempt_moves


def resolve_move_probabilities(
    current_probs: dict[str, float],
    *,
    constrained: bool,
    has_ring_flips: bool,
    has_crankshaft: bool,
) -> dict[str, float]:
    """Return move probabilities adjusted for current availability constraints."""
    tuning = get_runtime_tuning().move_scheduling
    probs = dict(current_probs)

    if constrained:
        fallback = tuning.constrained_fallback_move
        for move_type in tuning.constrained_disabled_moves:
            extra = probs.pop(move_type, 0.0)
            probs[fallback] = probs.get(fallback, 0.0) + extra

    availability = {
        "ring_flip": has_ring_flips,
        "crankshaft": has_crankshaft,
    }
    for move_type, is_available in availability.items():
        if is_available or move_type not in probs:
            continue
        fallback = tuning.availability_fallbacks[move_type]
        extra = probs.pop(move_type)
        probs[fallback] = probs.get(fallback, 0.0) + extra

    return probs


def resolve_forced_move(step: int, shake_period: int, *, constrained: bool) -> str | None:
    """Return a forced scheduled move for this step, if any."""
    tuning = get_runtime_tuning().move_scheduling
    if constrained:
        return None
    if step > 0 and step % shake_period == 0:
        return tuning.forced_periodic_move
    return None
