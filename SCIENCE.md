# How openconf Works

## The Science

Conformer generation is fundamentally a search problem: given a molecular graph, find the set of 3D geometries that represent distinct low-energy arrangements of the atoms. The challenge is that conformational space grows exponentially with molecular flexibility — a molecule with *n* rotatable bonds and *k* preferred angles per bond has roughly *kⁿ* possible conformations, most of which are either duplicates (same geometry reached via different paths) or high-energy strained structures.

openconf uses a hybrid strategy combining two complementary ideas. First, a torsion library: a collection of SMARTS patterns encoding preferred dihedral angles for common functional groups. Rather than handwritten rules, openconf ships 365 crystallography-derived rules via the RDKit CrystalFF torsion preferences (Riniker & Landrum, *J. Chem. Inf. Model.* 56, 2016). These rules are stored as Fourier-series coefficients; openconf converts them to discrete preferred angles by numerically minimizing each potential, then weights each minimum by a Boltzmann factor proportional to its depth. The result is a library that knows, for example, that secondary amide bonds strongly prefer planarity (0°), that aryl ethers prefer 0°/180°, and that sp³ C–C bonds prefer staggered 60°/180°/300°, with occupancy weights derived from crystal structure statistics rather than uniform guesses. Second, MCMM-style iterative exploration: starting from seed conformers, it randomly perturbs torsion angles guided by the library, minimizes the resulting geometry with MMFF94s, and keeps the result if it is energetically reasonable and structurally distinct.

Ring conformations require special treatment beyond simple torsion moves. For non-aromatic rings of size 5–7 (cyclohexane chairs, cyclopentane envelopes, piperidine flips, etc.), openconf adds a dedicated ring flip move: it computes the best-fit plane of the ring via SVD, then reflects each ring atom through that plane, generating the opposite face conformation. The reflected geometry is immediately minimized, which corrects any strain in the attached atoms. This move is selected with 10% probability per step when flippable rings exist, and probability is redistributed to other moves for fully rigid or fully aromatic molecules. For macrocycles (rings ≥ 10 atoms), the ETKDGv3 seeding step automatically enables `useMacrocycleTorsions`, which applies crystallography-derived distance bounds specific to large rings. For smaller non-aromatic rings (3–7 atoms), `useSmallRingTorsions` is likewise enabled, providing better starting geometries before exploration begins.

For non-aromatic rings of size ≥ 6, openconf also applies a *crankshaft move*: two non-adjacent ring atoms are chosen as anchors, and the arc of ring atoms between them is rotated rigidly about the axis passing through the anchors. Because both anchors lie on the rotation axis, every bond length in the ring is preserved exactly — only the bond angles at the anchors deform, and those relax in the subsequent MMFF minimization. This gives a closure-preserving macrocycle move that single-torsion rotations cannot match. Rotation angles are drawn from [30°, 120°] with random sign; the arc length is drawn uniformly from 1 to n−3 ring atoms. Substituent subtrees rotate rigidly with their host ring atom. The crankshaft is selected with 12% probability per step when a crankable ring exists, and the move type is exempt from the usual static clash filter — post-rotation geometries are often momentarily strained but MMFF relaxes them reliably. Seed pruning (`pruneRmsThresh`) is also disabled automatically when a macrocycle is present, because the 1.0 Å default collapses distinct puckers before they reach MMFF.

Macrocyclic backbones often hinge on the cis/trans configuration of their amide bonds, and the cis↔trans barrier is far too high for ordinary torsion jitter to cross during exploration. openconf therefore adds an *amide-flip move* targeting in-ring amide bonds (a non-aromatic carbonyl carbon single-bonded to nitrogen, both in a ring of ≥ 8 atoms; both secondary and tertiary amides qualify). The move rotates the rest of the ring 180° (± the usual torsion jitter) about the carbonyl-carbon→nitrogen axis, inverting the amide while — exactly as in the crankshaft — preserving every ring bond length, since both the carbonyl carbon and nitrogen lie on the rotation axis. Only the bond angles at those two atoms strain, and MMFF relaxes them. Each flippable amide bond is associated with the largest ring containing it (the macrocycle in fused systems) so the most flexible arc moves. Like the other ring moves it is selected with a small per-step probability when an in-ring amide exists and is exempt from the static clash filter. This complements the cis/trans seed enumeration: the seeds plant both configurations up front, while the move lets the walk discover a flip after other torsional rearrangements have accumulated.

Seed count is computed automatically from the molecular topology rather than set to a fixed value. The base formula is `max(20, n_rotatable × 3)` (controlled by `seed_n_per_rotor`, default 3), plus 5 seeds per flippable ring and `ring_size²` seeds per macrocycle ring, capped at 500. A simple drug-like molecule with 8 rotatable bonds gets ~24 seeds; a steroid with three non-aromatic rings gets ~35; a 12-membered macrocycle gets ~164; a 16-membered macrocycle gets ~276. The super-linear macrocycle term reflects the fact that the low-energy pucker fraction drops rapidly with ring size, so dense seeding is the cheapest way to ensure the global basin is sampled. For simple low-flexibility acyclic molecules and large flexible hydrocarbons, openconf applies data-backed seed-plan reductions so redundant RDKit embeddings are skipped and MC torsion moves do more of the exploration. You can override this by setting `n_seeds` explicitly in `ConformerConfig`.

The key to making this efficient is aggressive deduplication. Without it, the search quickly fills with near-identical structures that differ only in insignificant ways. openconf uses PRISM Pruner, which implements a cached divide-and-conquer algorithm for comparing conformers by RMSD and moment of inertia. Rather than doing O(N²) all-to-all comparisons, PRISM sorts conformers by energy and recursively partitions them, exploiting the fact that similar structures tend to cluster. This lets openconf maintain large internal pools (thousands of conformers) while keeping only the truly unique ones. The final selection step returns the lowest-energy conformers after PRISM deduplication, ensuring the output ensemble contains distinct, low-strain geometries.

## Low-Mode Following (optional)

When `use_low_mode_following=True`, openconf supplements the ETKDG seeds with a Hessian-guided step inspired by the LMOD procedure of Kolossváry & Guida (*JACS*, 1996). After each source seed is minimized, openconf numerically evaluates the 3N×3N Hessian matrix of the MMFF energy via central finite differences of the gradient: each Cartesian coordinate is displaced by ±`fd_step` (default 0.005 Å) and the gradient at the two displaced geometries is used to estimate one row of the Hessian. The resulting matrix is symmetrized.

Diagonalizing the Hessian yields a spectrum of normal modes. The first several eigenvalues are near zero (rigid-body translations and rotations); openconf detects the rigid-body/conformational boundary via the largest eigenvalue gap in the first eight entries rather than assuming exactly six zero modes, which correctly handles linear molecules and cases where minimization is not fully converged. The next eigenvectors — the **soft modes** — correspond to collective conformational motions: ring puckerings, coupled backbone torsions, and other deformations that cannot be decomposed into independent single-bond rotations.

For each soft mode whose eigenvalue is below `low_mode_eigenvalue_threshold` (default 100 kcal/mol/Å², capturing torsional and ring-puckering modes while excluding bond stretches), openconf scans the geometry in both the positive and negative directions along the eigenvector. At each step the geometry is displaced by `low_mode_scan_step_size` Å (3N Euclidean norm), the MMFF energy is evaluated, and scanning continues until the per-step energy increase exceeds `low_mode_scan_energy_threshold` (default ~2390 kcal/mol, equivalent to the paper's 10 000 kJ/mol) or `low_mode_scan_max_steps` is reached. The large default energy threshold means scanning passes through conformational barriers and stops only at severe steric clashes, placing the displaced endpoint on the far side of a barrier where a new local minimum can be found. Each scan endpoint that differs from the starting geometry is minimized; endpoints that minimize back to the starting geometry are discarded.

The dominant cost is Hessian evaluation: 6N MMFF force-field constructions for an N-atom molecule. `low_mode_n_source_seeds` (default 1) caps how many seeds receive this treatment — only the lowest-energy seeds are selected. With the default settings of `low_mode_n_source_seeds=1` and `low_mode_max_modes=5`, at most 10 new seeds (2 directions × 5 modes) are added per Hessian evaluation.

**When to enable:** macrocycles and other highly coupled flexible systems where independent single-rotor moves miss collective soft motions. The `"macrocycle"` preset enables low-mode following automatically alongside a wide 100 kcal/mol energy window. For simple acyclic drug-like molecules the MCMM exploration already covers these degrees of freedom and the Hessian cost is not worth paying.

| Parameter | Default | Meaning |
|---|---|---|
| `use_low_mode_following` | `False` | Enable low-mode seeding |
| `low_mode_eigenvalue_threshold` | 100.0 | Eigenvalue cutoff (kcal/mol/Å²) for soft-mode selection |
| `low_mode_max_modes` | 5 | Maximum soft modes to scan per source seed |
| `low_mode_scan_step_size` | 0.25 | Displacement per scan step in Å (3N Euclidean norm) |
| `low_mode_scan_energy_threshold` | 2390.0 | Per-step ΔE limit before stopping scan (kcal/mol) |
| `low_mode_scan_max_steps` | 10 | Maximum scan steps per direction |
| `low_mode_n_source_seeds` | 1 | Number of minimized seeds from which Hessians are computed |

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
ensemble = generate_conformers(mol, preset="macrocycle")    # macrocyclic ring systems
```

Each preset is a fully specified `ConformerConfig`; you can inspect and override individual
fields via `preset_config("docking")`. See the preset comparison table in
[README.md](README.md#configuration-quick-reference) for per-preset values. The parameters
below explain the trade-offs.

### Parameter Details

`max_out` — The maximum number of conformers to return. For docking, 100–200 is usually sufficient; you want enough diversity to capture different binding modes without overwhelming the docking program. For property calculations involving Boltzmann averaging (NMR shifts, optical rotation), you may want more conformers and a wider energy window to ensure adequate statistical sampling.

`n_seeds` — Number of initial ETKDG seed conformers. Defaults to `None`, which triggers automatic seed-plan resolution: topology-derived base count, ring and macrocycle bonuses, and conservative data-backed reductions for molecular classes where redundant ETKDG embeddings are common. Override with an explicit integer only when you have a specific reason — for example, to enforce reproducibility of a previous run, or to reduce runtime for large-scale screening.

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

Macrocycles (ring size ≥ 10): openconf uses ETKDGv3 macrocycle distance bounds at seed time, disables seed RMSD pruning (which otherwise collapses distinct puckers), applies crankshaft and kinematic ring-closure moves during exploration, and supports an amide-flip move for macrocycles containing in-ring amide bonds (both secondary and tertiary). On cycloalkanes C6–C14 the default config recovers the ETKDG+MMFF94s reference minimum within 1 kcal/mol. Cyclododecane is an edge case — its global [3333] basin occupies <1% of seeds and needs `n_seeds ≥ 300` for reliable recovery (default is ~164); override `n_seeds` if you need the exact global minimum for this specific ring size. The `"macrocycle"` preset is the recommended starting point: it enables low-mode following for Hessian-guided seeds and uses a 100 kcal/mol energy window to capture the full range of ring-pucker minima. For very flexible acyclic molecules (>10 rotatable bonds), increasing `n_steps` and `pool_max` helps ensure thorough exploration.

### Performance Tips

1. Start small and scale up: run with minimal settings first to verify the molecule processes correctly, then increase for production.

2. Check your output: if you are getting `max_out` conformers with very similar energies and low RMSD diversity, you may need more exploration steps or a larger pool. Consider whether the molecule has ring conformations that the ring flip move should be sampling.

3. Reproducibility: set `random_seed` and an explicit `n_seeds` value if you need deterministic results across runs (auto-computed `n_seeds` is deterministic for a given molecule, but pinning it explicitly guards against formula changes).

4. Rigid molecules: for molecules with few rotatable bonds (0–2) and no non-aromatic rings, most settings will not matter much — there simply are not many conformers to find. Default settings will be fast and sufficient.
