# How openconf Works

## The Science

Conformer generation is fundamentally a search problem: given a molecular graph, find the set of 3D geometries that represent distinct low-energy arrangements of the atoms. The challenge is that conformational space grows exponentially with molecular flexibility — a molecule with *n* rotatable bonds and *k* preferred angles per bond has roughly *kⁿ* possible conformations, most of which are either duplicates (same geometry reached via different paths) or high-energy strained structures.

openconf uses a hybrid strategy combining two complementary ideas. First, a torsion library: a collection of SMARTS patterns encoding preferred dihedral angles for common functional groups. Rather than handwritten rules, openconf ships 365 crystallography-derived rules via the RDKit CrystalFF torsion preferences (Riniker & Landrum, *J. Chem. Inf. Model.* 56, 2016). These rules are stored as Fourier-series coefficients; openconf converts them to discrete preferred angles by numerically minimizing each potential, then weights each minimum by a Boltzmann factor proportional to its depth. The result is a library that knows, for example, that secondary amide bonds strongly prefer planarity (0°), that aryl ethers prefer 0°/180°, and that sp³ C–C bonds prefer staggered 60°/180°/300°, with occupancy weights derived from crystal structure statistics rather than uniform guesses. Second, MCMM-style iterative exploration: starting from seed conformers, it randomly perturbs torsion angles guided by the library, minimizes the resulting geometry with MMFF94s, and keeps the result if it is energetically reasonable and structurally distinct.

Ring conformations require special treatment beyond simple torsion moves. For non-aromatic rings of size 5–7 (cyclohexane chairs, cyclopentane envelopes, piperidine flips, etc.), openconf adds a dedicated ring flip move: it computes the best-fit plane of the ring via SVD, then reflects each ring atom through that plane, generating the opposite face conformation. The reflected geometry is immediately minimized, which corrects any strain in the attached atoms. This move is selected with 10% probability per step when flippable rings exist, and probability is redistributed to other moves for fully rigid or fully aromatic molecules. For macrocycles (rings ≥ 10 atoms), the ETKDGv3 seeding step automatically enables `useMacrocycleTorsions`, which applies crystallography-derived distance bounds specific to large rings. For smaller non-aromatic rings (3–7 atoms), `useSmallRingTorsions` is likewise enabled, providing better starting geometries before exploration begins.

Seed count is computed automatically from the molecular topology rather than set to a fixed value. The formula is `max(20, n_rotatable × 3)` as a base (controlled by `seed_n_per_rotor`, default 3), plus 5 seeds per flippable ring and `ring_size × 3` seeds per macrocycle ring, capped at 500. A simple drug-like molecule with 8 rotatable bonds gets ~24 seeds; a steroid with three non-aromatic rings gets ~35; a 12-membered macrocycle gets ~56. You can override this by setting `n_seeds` explicitly in `ConformerConfig`.

The key to making this efficient is aggressive deduplication. Without it, the search quickly fills with near-identical structures that differ only in insignificant ways. openconf uses PRISM Pruner, which implements a cached divide-and-conquer algorithm for comparing conformers by RMSD and moment of inertia. Rather than doing O(N²) all-to-all comparisons, PRISM sorts conformers by energy and recursively partitions them, exploiting the fact that similar structures tend to cluster. This lets openconf maintain large internal pools (thousands of conformers) while keeping only the truly unique ones. The final selection step returns the lowest-energy conformers after PRISM deduplication, ensuring the output ensemble contains distinct, low-strain geometries.

## Pose-Constrained Generation (FEP / Analogue Mode)

A common need in lead optimization is generating conformers for an analogue where the core scaffold is already placed — for example, the result of an MCS alignment from a co-crystal structure. Standard ETKDG seeding is not appropriate here: it randomizes the entire molecule and cannot be guaranteed to recover the aligned pose. Instead, openconf supports constrained generation via `generate_conformers_from_pose`.

The idea is to identify a set of *constrained atoms* (the MCS core) whose Cartesian positions must remain fixed at the input geometry, and to explore only the remaining degrees of freedom — the *free rotors*, i.e., bonds whose entire moving fragment lies outside the constrained set. The algorithm adapts in three ways:

**Seeding.** ETKDG is skipped entirely. The single input conformer is used directly as the seed, immediately fast-minimized with position restraints to relax any bond-length or angle strain in the free fragment before exploration begins.

**Move set.** Only free rotors (and ring flips of entirely free rings) are sampled. The global shake move is suppressed: it would randomize 50–80% of all rotors, some of which pass through the core, defeating the constraint. Move probability that would have gone to global shake is redistributed to `single_rotor` moves.

**Minimization.** Every MMFF minimization call applies `MMFFAddPositionConstraint` to each constrained atom with a stiff harmonic force constant (default 1000 kcal/mol/Å²). After minimization converges, constrained atom coordinates are snapped back to the exact reference values, eliminating any residual drift. Final refinement applies the same treatment.

The rotor filtering logic uses a BFS traversal: for each bond (i, j) in the rotor list, we compute the set of atoms reachable from j without crossing (i, j). If this moving fragment contains no constrained atoms, the rotor is free. If the moving fragment *does* contain constrained atoms but the opposite side (reachable from i) does not, the rotor is flipped so that the free atoms become the moving side. Rotors where constrained atoms appear on both sides are excluded entirely — they cannot be rotated without moving the core.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from openconf import generate_conformers_from_pose

# MCS-aligned analogue: ethyl group added to a benzene scaffold
mol = Chem.MolFromSmiles("CCc1ccccc1")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())  # or use your aligned pose

# Constrain the benzene ring (heavy-atom indices 2–7)
ensemble = generate_conformers_from_pose(mol, constrained_atoms=list(range(2, 8)))
```

The `"analogue"` preset (50 conformers, 150 steps, softmax parent strategy, full refinement) is the default. Because the free rotor space is much smaller than the full conformational space, 150 steps is usually sufficient for thorough coverage of simple R-groups. For larger substituents with many free rotors, increase `n_steps` accordingly.

**Atom index convention.** `Chem.AddHs` appends new H atoms after all existing atoms, preserving all prior indices. So whether you pass a heavy-atom-only mol (indices 0…N−1 for heavy atoms) or an H-added mol (same heavy-atom indices plus new H indices), the constrained atom indices remain valid after the internal `AddHs` call.

## Tuning Guide

For the most common workflows, named presets are the easiest starting point:

```python
from openconf import generate_conformers

ensemble = generate_conformers(mol, preset="rapid")         # fast virtual screening (~45 ms/mol)
ensemble = generate_conformers(mol, preset="ensemble")      # property prediction (50 conformers)
ensemble = generate_conformers(mol, preset="spectroscopic") # Boltzmann ensemble for NMR/IR
ensemble = generate_conformers(mol, preset="docking")       # maximize bioactive recall
```

Each preset is a fully specified `ConformerConfig`; you can inspect and override individual
fields via `preset_config("docking")`. See the preset comparison table in
[README.md](README.md#configuration-quick-reference) for per-preset values. The parameters
below explain the trade-offs.

### Parameter Details

`max_out` — The maximum number of conformers to return. For docking, 100–200 is usually sufficient; you want enough diversity to capture different binding modes without overwhelming the docking program. For property calculations involving Boltzmann averaging (NMR shifts, optical rotation), you may want more conformers and a wider energy window to ensure adequate statistical sampling.

`n_seeds` — Number of initial ETKDG seed conformers. Defaults to `None`, which triggers automatic computation: `max(20, n_rotatable × 3)` plus bonuses for flippable rings and macrocycles. This is appropriate for most use cases. Override with an explicit integer only when you have a specific reason — for example, to enforce reproducibility of a previous run, or to reduce runtime for large-scale screening.

`n_steps` — Number of Monte Carlo exploration steps. Each step proposes a new conformer by perturbing torsions (or flipping a ring) from an existing pool member. More steps explore more thoroughly but take longer. For molecules with many rotatable bonds, more steps are needed to adequately sample the space.

`pool_max` — Internal pool size limit. Caps memory usage and keeps deduplication fast. Should be several times larger than `max_out` to allow the algorithm to explore before selecting the final diverse set.

`energy_window_kcal` — Conformers more than this many kcal/mol above the minimum are discarded. For docking, 10–12 kcal/mol is typical (conformers much higher than this are unlikely to be bioactive). For Boltzmann-weighted property calculations, 15–20 kcal/mol captures the full thermally accessible ensemble, though conformers beyond ~6 kcal/mol contribute negligibly at room temperature.

### Use Case Examples

The four built-in presets cover the most common workflows. Start from one and override individual fields only when needed.

```python
from openconf import generate_conformers, preset_config

# Docking pose recovery (uniform parent sampling, wide energy window, no final refine).
# Defaults: max_out=250, n_steps=500, energy_window_kcal=18.0. Informed by the
# Iridium benchmark (see docs/benchmark_report.md).
ensemble = generate_conformers(mol, preset="docking")

# Boltzmann-weighted properties (NMR, IR, VCD, optical rotation).
# Defaults: max_out=100, n_steps=400, energy_window_kcal=5.0, final_select="energy".
# The tight 5 kcal window already covers >99% of the Boltzmann population at 300 K.
ensemble = generate_conformers(mol, preset="spectroscopic")

# Balanced ensemble for ML / physics-based property prediction.
# Defaults: max_out=50, n_steps=200, energy_window_kcal=10.0.
ensemble = generate_conformers(mol, preset="ensemble")

# Large-scale virtual screening — 5 diverse conformers per molecule, ~45 ms each.
# Defaults: max_out=5, n_steps=30, do_final_refine=False.
ensemble = generate_conformers(mol, preset="rapid")
```

Pharmacophore searching / shape screening has no matching preset — the goal is more conformers at a wider energy window than `"docking"`. Starting from `"docking"` and widening is usually the right move:

```python
config = preset_config("docking")
config.max_out = 500
config.n_steps = 1000
config.energy_window_kcal = 20.0
ensemble = generate_conformers(mol, config=config)
```

Macrocycles (ring size ≥ 12): openconf is not recommended for macrocyclic ring systems. ETKDGv3 has dedicated macrocycle distance-geometry bounds that openconf does not replicate, and in practice ETKDG+MMFF94s outperforms openconf on both RMSD and ensemble coverage metrics for large rings. For very flexible acyclic molecules (>10 rotatable bonds), increasing `n_steps` and `pool_max` helps ensure thorough exploration.

### PRISM Pruner Settings

`PrismConfig.energy_window_kcal` (default: 15.0) — Only compare conformers within this energy window during pruning. Should be ≥ your main `energy_window_kcal` setting to avoid pruning conformers that are still within your desired energy range.

```python
from openconf import PrismConfig

prism_config = PrismConfig(energy_window_kcal=20.0)

config = ConformerConfig(
    max_out=200,
    energy_window_kcal=15.0,
    prism_config=prism_config,
)
```

### Performance Tips

1. Start small and scale up: run with minimal settings first to verify the molecule processes correctly, then increase for production.

2. Check your output: if you are getting `max_out` conformers with very similar energies and low RMSD diversity, you may need more exploration steps or a larger pool. Consider whether the molecule has ring conformations that the ring flip move should be sampling.

3. Reproducibility: set `random_seed` and an explicit `n_seeds` value if you need deterministic results across runs (auto-computed `n_seeds` is deterministic for a given molecule, but pinning it explicitly guards against formula changes).

4. Rigid molecules: for molecules with few rotatable bonds (0–2) and no non-aromatic rings, most settings will not matter much — there simply are not many conformers to find. Default settings will be fast and sufficient.
