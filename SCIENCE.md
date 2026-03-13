# How openconf Works

## The Science

Conformer generation is fundamentally a search problem: given a molecular graph, find the set of 3D geometries that represent distinct low-energy arrangements of the atoms. The challenge is that conformational space grows exponentially with molecular flexibility — a molecule with *n* rotatable bonds and *k* preferred angles per bond has roughly *kⁿ* possible conformations, most of which are either duplicates (same geometry reached via different paths) or high-energy strained structures.

openconf uses a hybrid strategy combining two complementary ideas. First, a torsion library: a collection of SMARTS patterns encoding preferred dihedral angles for common functional groups. Rather than handwritten rules, openconf ships 365 crystallography-derived rules via the RDKit CrystalFF torsion preferences (Riniker & Landrum, *J. Chem. Inf. Model.* 56, 2016). These rules are stored as Fourier-series coefficients; openconf converts them to discrete preferred angles by numerically minimizing each potential, then weights each minimum by a Boltzmann factor proportional to its depth. The result is a library that knows, for example, that secondary amide bonds strongly prefer planarity (0°), that aryl ethers prefer 0°/180°, and that sp³ C–C bonds prefer staggered 60°/180°/300°, with occupancy weights derived from crystal structure statistics rather than uniform guesses. Second, MCMM-style iterative exploration: starting from seed conformers, it randomly perturbs torsion angles guided by the library, minimizes the resulting geometry with MMFF94s, and keeps the result if it is energetically reasonable and structurally distinct.

Ring conformations require special treatment beyond simple torsion moves. For non-aromatic rings of size 5–7 (cyclohexane chairs, cyclopentane envelopes, piperidine flips, etc.), openconf adds a dedicated ring flip move: it computes the best-fit plane of the ring via SVD, then reflects each ring atom through that plane, generating the opposite face conformation. The reflected geometry is immediately minimized, which corrects any strain in the attached atoms. This move is selected with 10% probability per step when flippable rings exist, and probability is redistributed to other moves for fully rigid or fully aromatic molecules. For macrocycles (rings ≥ 10 atoms), the ETKDGv3 seeding step automatically enables `useMacrocycleTorsions`, which applies crystallography-derived distance bounds specific to large rings. For smaller non-aromatic rings (3–7 atoms), `useSmallRingTorsions` is likewise enabled, providing better starting geometries before exploration begins.

Seed count is computed automatically from the molecular topology rather than set to a fixed value. The formula is `max(20, n_rotatable × 3)` as a base (controlled by `seed_n_per_rotor`, default 3), plus 5 seeds per flippable ring and `ring_size × 3` seeds per macrocycle ring, capped at 500. A simple drug-like molecule with 8 rotatable bonds gets ~24 seeds; a steroid with three non-aromatic rings gets ~35; a 12-membered macrocycle gets ~56. You can override this by setting `n_seeds` explicitly in `ConformerConfig`.

The key to making this efficient is aggressive deduplication. Without it, the search quickly fills with near-identical structures that differ only in insignificant ways. openconf uses PRISM Pruner, which implements a cached divide-and-conquer algorithm for comparing conformers by RMSD and moment of inertia. Rather than doing O(N²) all-to-all comparisons, PRISM sorts conformers by energy and recursively partitions them, exploiting the fact that similar structures tend to cluster. This lets openconf maintain large internal pools (thousands of conformers) while keeping only the truly unique ones. The final selection step returns the lowest-energy conformers after PRISM deduplication, ensuring the output ensemble contains distinct, low-strain geometries.

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
fields via `preset_config("docking")`. The parameters below explain the trade-offs.

### Quick Reference

| Use Case | max_out | n_seeds | n_steps | pool_max | energy_window_kcal |
|----------|---------|---------|---------|----------|-------------------|
| Fast docking prep | 50-100 | auto | 100 | 200 | 10 |
| Thorough docking | 200-500 | auto | 500 | 1000 | 12 |
| Ensemble properties (NMR, etc.) | 100-300 | auto | 500 | 1000 | 15 |
| Exhaustive search | 500+ | 200+ | 2000 | 5000 | 20 |

### Parameter Details

`max_out` — The maximum number of conformers to return. For docking, 100–200 is usually sufficient; you want enough diversity to capture different binding modes without overwhelming the docking program. For property calculations involving Boltzmann averaging (NMR shifts, optical rotation), you may want more conformers and a wider energy window to ensure adequate statistical sampling.

`n_seeds` — Number of initial ETKDG seed conformers. Defaults to `None`, which triggers automatic computation: `max(20, n_rotatable × 3)` plus bonuses for flippable rings and macrocycles. This is appropriate for most use cases. Override with an explicit integer only when you have a specific reason — for example, to enforce reproducibility of a previous run, or to reduce runtime for large-scale screening.

`n_steps` — Number of Monte Carlo exploration steps. Each step proposes a new conformer by perturbing torsions (or flipping a ring) from an existing pool member. More steps explore more thoroughly but take longer. For molecules with many rotatable bonds, more steps are needed to adequately sample the space.

`pool_max` — Internal pool size limit. Caps memory usage and keeps deduplication fast. Should be several times larger than `max_out` to allow the algorithm to explore before selecting the final diverse set.

`energy_window_kcal` — Conformers more than this many kcal/mol above the minimum are discarded. For docking, 10–12 kcal/mol is typical (conformers much higher than this are unlikely to be bioactive). For Boltzmann-weighted property calculations, 15–20 kcal/mol captures the full thermally accessible ensemble, though conformers beyond ~6 kcal/mol contribute negligibly at room temperature.

### Use Case Examples

Docking preparation: use moderate settings and prioritize diversity. The docking program handles fine-grained optimization. The `"docking"` preset's choices (`uniform` parent strategy, wide energy window, no final refinement) were informed by the [Iridium benchmark](docs/benchmark_report.md), which evaluates bioactive conformer recovery across 120 drug-like molecules.

```python
config = ConformerConfig(
    max_out=200,
    n_steps=500,
    pool_max=1000,
    energy_window_kcal=12.0,
)
```

Pharmacophore searching / shape screening: similar to docking, but a wider energy window is useful since you are matching against a static query.

```python
config = ConformerConfig(
    max_out=500,
    n_steps=1000,
    pool_max=2000,
    energy_window_kcal=15.0,
)
```

Boltzmann-weighted properties (NMR chemical shifts, J-couplings, optical rotation): accurate coverage of the low-energy region is important since properties are averaged according to Boltzmann populations.

```python
config = ConformerConfig(
    max_out=300,
    n_steps=1000,
    pool_max=2000,
    energy_window_kcal=15.0,  # ~6 kcal/mol is 99.99% of Boltzmann population at 298 K
)
```

Large-scale screening (thousands of molecules): speed matters more than exhaustiveness.

```python
config = ConformerConfig(
    max_out=50,
    n_seeds=20,   # override auto to keep it fast
    n_steps=100,
    pool_max=200,
    energy_window_kcal=10.0,
)
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
