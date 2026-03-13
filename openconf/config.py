"""Configuration dataclasses for openconf."""

from dataclasses import dataclass, field
from typing import Literal

ConformerPreset = Literal["rapid", "ensemble", "spectroscopic", "docking"]


@dataclass
class PrismConfig:
    """Configuration for PRISM Pruner deduplication.

    Note: PRISM uses automatic thresholds for RMSD, MOI, and max deviation.

    Attributes:
        energy_window_kcal: Energy window for comparing conformers (kcal/mol).
    """

    energy_window_kcal: float = 15.0


@dataclass
class ConformerConfig:
    """Configuration for conformer generation.

    Attributes:
        max_out: Maximum number of conformers to return.
        pool_max: Maximum pool size during generation.
        n_seeds: Number of initial ETKDG seed conformers. If None (default),
            computed automatically as ``max(20, n_rotatable * seed_n_per_rotor)`` with additional
            seeds for non-aromatic rings and macrocycles.
        n_steps: Number of exploration steps for the walker.
        energy_window_kcal: Energy window for keeping conformers (kcal/mol).
        dedupe_period: Run deduplication every N steps.
        shake_period: Apply global shake every N steps.
        move_probs: Probabilities for each move type.
        torsion_jitter_deg: Random jitter to add to torsion angles (degrees).
        minimizer: Which minimizer to use ("rdkit_mmff" or "openmm").
        prism_config: Configuration for PRISM deduplication.
        random_seed: Random seed for reproducibility.
        num_threads: Number of threads for parallel operations.
        use_heavy_atoms_only: Use only heavy atoms for RMSD calculations.
        clash_threshold: Distance threshold for clash detection (Angstroms).
        fast_minimization_iters: Iterations for quick minimization.
        max_minimization_iters: Maximum iterations for minimization.
        parent_strategy: How to choose the parent conformer from the current pool when proposing a new move.
            This controls the exploration/exploitation balance of the MCMM-style search:

            - "softmax": Energy-biased sampling. Parents are drawn with probability
              ~ exp(-(E - Emin)/T), favoring low-energy conformers while still allowing
              occasional high-energy picks. Good for converging into low-energy basins,
              but can reduce conformational diversity and hurt "best-of-N" recall metrics.

            - "uniform": Parents are sampled uniformly at random from the pool.
              Encourages exploration and diversity (often better for docking-style
              benchmarks where the goal is minimum RMSD to a bioactive conformation).

            - "best": Always mutate the current lowest-energy conformer. Fast convergence
              but most prone to mode collapse; generally not recommended when diversity
              is important.
        final_select: How the final conformer set is chosen.
        skip_clash_check: If True, skip the pre-minimization clash check entirely.
            If False (default), use a fast numpy-based clash filter that avoids
            expensive minimization of heavily clashed structures. For large or
            flexible molecules this is a net speedup; for tiny molecules (<15 heavy
            atoms) the check overhead may slightly exceed the savings.
        seed_n_per_rotor: Seeds generated per rotatable bond (formula:
            max(20, n_rotors * seed_n_per_rotor) + ring bonuses). Lower values
            reduce seeding cost for flexible molecules (default 3, was historically 5).
        seed_prune_rms_thresh: ETKDG pruning RMSD threshold during seed generation.
            Seeds within this distance of each other are discarded during embedding.
            Higher values (e.g. 1.0 Å) give fewer, more diverse seeds and reduce
            the batch-MMFF cost. Lower values (e.g. 0.5 Å) give more seeds.
        minimize_batch_size: Number of proposals to accumulate before running a
            single parallel MMFFOptimizeMoleculeConfs call. Values > 1 use
            numThreads cores simultaneously and give substantial speedup on
            multi-core machines. 1 = sequential (original behaviour).
        do_final_refine: If True (default), run a full MMFF minimization on the
            final conformer set using max_minimization_iters. Set to False to skip
            this step and return the fast-minimized geometries directly — useful
            for docking-prep or virtual screening workflows where accurate MMFF
            energies are not required. Saves ~0.9s per 100 conformers on drug-like
            molecules.
        fast_dielectric: Dielectric constant used during MCMM sampling and seed
            generation. Higher values (default 10.0) soften intramolecular
            electrostatics, reducing artificial gas-phase minima and improving
            conformational space coverage.
        final_dielectric: Dielectric constant used for the optional final MMFF
            refinement pass. Lower than fast_dielectric (default 4.0) to give
            more physically meaningful energies representative of a condensed-phase
            environment (protein interior / organic solvent).
    """

    max_out: int = 200
    pool_max: int = 2000
    n_seeds: int | None = None
    n_steps: int = 500
    energy_window_kcal: float = 12.0
    dedupe_period: int = 50
    shake_period: int = 20
    move_probs: dict[str, float] = field(
        default_factory=lambda: {
            "single_rotor": 0.35,
            "multi_rotor": 0.28,
            "correlated": 0.18,
            "global_shake": 0.09,
            "ring_flip": 0.10,
        }
    )
    torsion_jitter_deg: float = 10.0
    minimizer: Literal["rdkit_mmff", "openmm"] = "rdkit_mmff"
    prism_config: PrismConfig = field(default_factory=PrismConfig)
    random_seed: int | None = None
    num_threads: int = 0
    use_heavy_atoms_only: bool = True
    clash_threshold: float = 1.5
    fast_minimization_iters: int = 20
    max_minimization_iters: int = 200
    do_final_refine: bool = True
    parent_strategy: Literal["softmax", "uniform", "best"] = "softmax"
    final_select: Literal["energy", "diverse"] = "diverse"
    skip_clash_check: bool = False
    seed_n_per_rotor: int = 3
    seed_prune_rms_thresh: float = 1.0
    minimize_batch_size: int = 8
    fast_dielectric: float = 10.0
    final_dielectric: float = 4.0


def preset_config(preset: ConformerPreset) -> "ConformerConfig":
    """Return a ConformerConfig tuned for a common use case.

    Presets:

    - ``"rapid"`` — FastROCS-style virtual screening. Generates 5 diverse
      conformers per molecule as fast as possible (~45 ms for drug-like
      molecules on a single core). Skips final MMFF refinement.

    - ``"ensemble"`` — Balanced conformer ensemble for property prediction
      (logP, pKa, ML descriptors). 50 conformers, full refinement.

    - ``"spectroscopic"`` — Exhaustive Boltzmann ensemble for NMR/IR/VCD.
      Tight 5 kcal energy window, energy-ranked output, dense seeding. Caller
      should weight conformers by ``exp(-E/RT)``.

    - ``"docking"`` — Maximize bioactive conformation recall for docking
      workflows. Wide energy window, uniform parent sampling, no final
      refinement (docking programs minimize in-situ).

    Args:
        preset: One of ``"rapid"``, ``"ensemble"``, ``"spectroscopic"``,
            ``"docking"``.

    Returns:
        ConformerConfig configured for the requested use case.

    Raises:
        ValueError: If preset is not recognized.

    Examples:
        >>> config = preset_config("docking")
        >>> config.max_out
        250
        >>> config = preset_config("rapid")
        >>> config.do_final_refine
        False
    """
    match preset:
        case "rapid":
            return ConformerConfig(
                max_out=5,
                pool_max=100,
                n_steps=30,
                energy_window_kcal=20.0,
                seed_n_per_rotor=2,
                seed_prune_rms_thresh=1.5,
                do_final_refine=False,
                minimize_batch_size=16,
                dedupe_period=15,
                shake_period=10,
                final_select="diverse",
            )
        case "ensemble":
            return ConformerConfig(
                max_out=50,
                pool_max=500,
                n_steps=200,
                energy_window_kcal=10.0,
                seed_n_per_rotor=3,
                seed_prune_rms_thresh=1.0,
                do_final_refine=True,
                minimize_batch_size=8,
                final_select="diverse",
            )
        case "spectroscopic":
            return ConformerConfig(
                max_out=100,
                pool_max=1000,
                n_steps=400,
                energy_window_kcal=5.0,
                seed_n_per_rotor=5,
                seed_prune_rms_thresh=0.5,
                do_final_refine=True,
                minimize_batch_size=8,
                parent_strategy="softmax",
                final_select="energy",
            )
        case "docking":
            return ConformerConfig(
                max_out=250,
                pool_max=2500,
                n_steps=500,
                energy_window_kcal=18.0,
                seed_n_per_rotor=4,
                seed_prune_rms_thresh=0.8,
                do_final_refine=False,
                minimize_batch_size=8,
                parent_strategy="uniform",
                final_select="diverse",
                prism_config=PrismConfig(energy_window_kcal=18.0),
            )
        case _:
            raise ValueError(
                f"Unknown preset {preset!r}. Choose from: 'rapid', 'ensemble', 'spectroscopic', 'docking'."
            )
