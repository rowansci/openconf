# openconf

[![License](https://img.shields.io/github/license/rowansci/openconf)](https://github.com/rowansci/openconf/blob/master/LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/rowansci/openconf/test.yml?branch=master&logo=github-actions)](https://github.com/rowansci/openconf/actions/)
[![Codecov](https://img.shields.io/codecov/c/github/rowansci/openconf)](https://codecov.io/gh/rowansci/openconf)


*a conformer generator for drug-like molecules: uses torsional Monte Carlo moves to quickly generate diverse ensembles, uses RDKit/MMFF94s throughout, and runs fast enough for large-scale workflows.*

## Installation

```bash
pip install openconf
```

## Quick Start

### Python API

```python
from rdkit import Chem
from openconf import generate_conformers, ConformerConfig

# From SMILES
mol = Chem.MolFromSmiles("CCCCc1ccccc1")
ensemble = generate_conformers(mol)

print(f"Generated {ensemble.n_conformers} conformers")
print(ensemble.summary())

# Save to SDF
ensemble.to_sdf("output.sdf")

# Or XYZ
ensemble.to_xyz("output.xyz")
```

### Named Presets

Five use-case presets are available out of the box:

```python
from openconf import generate_conformers

ensemble = generate_conformers(mol, preset="rapid")         # fast virtual screening
ensemble = generate_conformers(mol, preset="ensemble")      # property prediction
ensemble = generate_conformers(mol, preset="spectroscopic") # NMR / IR / VCD
ensemble = generate_conformers(mol, preset="docking")       # docking pose recovery
```

For FEP-style analogue generation from a fixed pose, see [`generate_conformers_from_pose`](#5-analogue--fep-style-r-group-exploration) below.

### Custom Configuration

For full control, pass a `ConformerConfig` directly. You can also use
`preset_config()` as a starting point and override individual fields:

```python
from openconf import generate_conformers, preset_config, ConformerConfig

# Start from a preset and tweak
config = preset_config("docking")
config.max_out = 500
config.random_seed = 42
ensemble = generate_conformers(mol, config=config)

# Or build from scratch
config = ConformerConfig(
    max_out=200,              # Maximum conformers to return
    pool_max=2000,            # Internal pool size
    n_steps=500,              # Exploration steps
    energy_window_kcal=12.0,  # Energy window for filtering
    random_seed=42,           # For reproducibility
)
ensemble = generate_conformers(mol, config=config)
```

### Command Line

```bash
# From SMILES file
openconf molecules.smi --max-out 200 --out conformers.sdf

# From SDF file
openconf input.sdf --method hybrid --ew 15 --out output.sdf

# With verbose output
openconf "CCCCc1ccccc1" --max-out 100 -o butylbenzene.sdf -v
```

## Use-Case Examples

The right configuration depends on the downstream task. Four named presets
cover the most common workflows:

```python
from openconf import generate_conformers, preset_config

# One-liner with a preset
ensemble = generate_conformers(mol, preset="docking")

# Or get the config object to inspect / tweak before use
config = preset_config("spectroscopic")
config.max_out = 200          # override a single field
ensemble = generate_conformers(mol, config=config)
```

Available presets: `"rapid"`, `"ensemble"`, `"spectroscopic"`, `"docking"`, `"analogue"`.

Below are representative wall-clock timings measured on a single CPU core
(Apple M2 Pro), mean over 3 runs.

---

### 1. Rapid — fast virtual screening

Enumerate a handful of diverse shapes per molecule as fast as possible.
Appropriate for ligand-based virtual screening at large scale.

- `max_out=5`, `n_steps=30` — minimal per-molecule budget
- `do_final_refine=False` — skip the final MMFF pass (shape tools re-minimize anyway)
- `seed_n_per_rotor=2`, `seed_prune_rms_thresh=1.5` — coarser seeding
- `minimize_batch_size=16` — larger parallel batches for multi-core machines

```python
from openconf import generate_conformers

ensemble = generate_conformers("CC(C)Cc1ccc(cc1)C(C)C(=O)O", preset="rapid")
```

<details><summary>Full config equivalent</summary>

```python
from openconf import ConformerConfig, generate_conformers

config = ConformerConfig(
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
ensemble = generate_conformers("CC(C)Cc1ccc(cc1)C(C)C(=O)O", config=config)
```

</details>

| Molecule | Heavy atoms | Rotors | Time (s) | Conformers |
|---|---|---|---|---|
| butylbenzene | 13 | 3 | 0.043 | 5 |
| ibuprofen | 18 | 5 | 0.046 | 5 |
| celecoxib | 26 | 4 | 0.063 | 5 |
| maraviroc | 34 | 7 | 0.848 | 5 |

At ~45 ms per drug-like molecule on a single core, a 32-core machine processes
roughly 60 M molecules/day — sufficient for 1B-scale campaigns with a cluster.

---

### 2. Ensemble — property prediction

A compact, diverse ensemble for downstream ML or physics-based properties
(logP, pKa, conformational descriptors).

- `max_out=50`, `n_steps=200` — balanced quality/speed
- `energy_window_kcal=10.0` — includes the thermally accessible range
- `final_select="diverse"` — maximize chemical diversity over the ensemble

```python
from openconf import generate_conformers

ensemble = generate_conformers("CC(C)Cc1ccc(cc1)C(C)C(=O)O", preset="ensemble")
```

<details><summary>Full config equivalent</summary>

```python
from openconf import ConformerConfig, generate_conformers

config = ConformerConfig(
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
ensemble = generate_conformers("CC(C)Cc1ccc(cc1)C(C)C(=O)O", config=config)
```

</details>

| Molecule | Heavy atoms | Rotors | Time (s) | Conformers |
|---|---|---|---|---|
| butylbenzene | 13 | 3 | 0.122 | 50 |
| ibuprofen | 18 | 5 | 0.186 | 50 |
| celecoxib | 26 | 4 | 0.275 | 50 |
| maraviroc | 34 | 7 | 1.580 | 50 |

---

### 3. Spectroscopic — NMR, IR, VCD

Exhaustively populate all thermally accessible conformers with accurate
relative MMFF energies for Boltzmann-weighted spectral averaging.

- `energy_window_kcal=5.0` — ~3 kcal covers >99% of the Boltzmann population
  at 300 K; 5 kcal provides margin for MMFF error
- `final_select="energy"` — return lowest-energy conformers; weight by `exp(-E/kT)`
- `parent_strategy="softmax"` — bias sampling toward low-energy basins
- `seed_n_per_rotor=5`, `seed_prune_rms_thresh=0.5` — dense seeding to avoid
  missing shallow minima
- `do_final_refine=True` — accurate relative energies are critical here

```python
import numpy as np
from openconf import generate_conformers

ensemble = generate_conformers("CC(C)Cc1ccc(cc1)C(C)C(=O)O", preset="spectroscopic")

# Boltzmann weights at 300 K
RT = 0.592  # kcal/mol at 300 K
energies = np.array(ensemble.energies)
weights = np.exp(-(energies - energies.min()) / RT)
weights /= weights.sum()
```

<details><summary>Full config equivalent</summary>

```python
from openconf import ConformerConfig, generate_conformers

config = ConformerConfig(
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
ensemble = generate_conformers("CC(C)Cc1ccc(cc1)C(C)C(=O)O", config=config)
```

</details>

| Molecule | Heavy atoms | Rotors | Time (s) | Conformers |
|---|---|---|---|---|
| butylbenzene | 13 | 3 | 0.181 | 91 |
| ibuprofen | 18 | 5 | 0.327 | 100 |
| celecoxib | 26 | 4 | 0.289 | 36 |
| maraviroc | 34 | 7 | 2.374 | 75 |

Fewer conformers for celecoxib/maraviroc reflect the tight 5 kcal window —
rigid, aromatic-rich scaffolds have few populated conformers at room temperature.

---

### 4. Docking Pose Recovery

Maximize the chance that the bioactive conformation is in the output set,
i.e. minimize best-RMSD-to-crystal across the ensemble. This is the right
choice when preparing a single compound for docking where conformer quality
matters. For bulk library preparation (thousands of molecules), `"rapid"` is
usually more appropriate.

- `parent_strategy="uniform"` — broad exploration; energy-biased sampling
  hurts recall of strained bioactive conformers
- `energy_window_kcal=18.0` — bioactive conformations are often 5–15 kcal
  above the MMFF global minimum
- `do_final_refine=False` — docking programs minimize inside the binding site;
  pre-minimized geometries can hurt pose recall
- `max_out=250`, `n_steps=500` — larger ensemble improves recall at acceptable cost

```python
from openconf import generate_conformers

ensemble = generate_conformers("CC(C)Cc1ccc(cc1)C(C)C(=O)O", preset="docking")
ensemble.to_sdf("output.sdf")
```

<details><summary>Full config equivalent</summary>

```python
from openconf import ConformerConfig, PrismConfig, generate_conformers

config = ConformerConfig(
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
ensemble = generate_conformers("CC(C)Cc1ccc(cc1)C(C)C(=O)O", config=config)
ensemble.to_sdf("docking_input.sdf")
```

</details>

| Molecule | Heavy atoms | Rotors | Time (s) | Conformers |
|---|---|---|---|---|
| butylbenzene | 13 | 3 | 0.232 | 140 |
| ibuprofen | 18 | 5 | 0.326 | 169 |
| celecoxib | 26 | 4 | 0.397 | 231 |
| maraviroc | 34 | 7 | 1.826 | 172 |

---

### 5. Analogue / FEP-style R-group exploration

Generate conformers for an MCS-aligned analogue while keeping the core scaffold
exactly fixed at the input pose. The correct entry point here is
`generate_conformers_from_pose` rather than `generate_conformers`.

- Starts from the **supplied conformer** — no ETKDG seeding
- Only **free terminal rotors** are explored (those whose moving fragment is
  entirely outside the constrained core)
- MMFF minimization uses stiff **position restraints** on all constrained atoms,
  then snaps them to exact starting coordinates so there is zero drift
- **Global shake** is suppressed to avoid thrashing the starting pose

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from openconf import generate_conformers_from_pose

# Suppose we have an MCS-aligned analogue with a propyl substituent
# replacing the butyl chain on a benzene scaffold.
mol = Chem.MolFromSmiles("CCCc1ccccc1")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

# Ring heavy-atom indices — these must not move
ring_atoms = [3, 4, 5, 6, 7, 8]

ensemble = generate_conformers_from_pose(mol, constrained_atoms=ring_atoms)
ensemble.to_sdf("analogues.sdf")
```

The default preset (`"analogue"`) returns up to 50 conformers. Pass `preset=` or
`config=` to override:

```python
from openconf import ConformerConfig, generate_conformers_from_pose

# Fewer conformers, faster turnaround
config = ConformerConfig(max_out=10, n_steps=60, pool_max=200)
ensemble = generate_conformers_from_pose(mol, constrained_atoms=ring_atoms, config=config)
```

<details><summary>Full analogue preset equivalent</summary>

```python
from openconf import ConformerConfig, PrismConfig, generate_conformers_from_pose

config = ConformerConfig(
    max_out=50,
    pool_max=500,
    n_steps=150,
    energy_window_kcal=10.0,
    do_final_refine=True,
    minimize_batch_size=8,
    parent_strategy="softmax",
    final_select="diverse",
    prism_config=PrismConfig(energy_window_kcal=10.0),
)
ensemble = generate_conformers_from_pose(mol, constrained_atoms=ring_atoms, config=config)
```

</details>

---

### Configuration Quick Reference

| Parameter | Rapid | Ensemble | Spectroscopic | Docking | Analogue |
|---|---|---|---|---|---|
| `max_out` | 5 | 50 | 100 | 250 | 50 |
| `n_steps` | 30 | 200 | 400 | 500 | 150 |
| `energy_window_kcal` | 20 | 10 | 5 | 18 | 10 |
| `seed_n_per_rotor` | 2 | 3 | 5 | 4 | — |
| `seed_prune_rms_thresh` | 1.5 | 1.0 | 0.5 | 0.8 | — |
| `do_final_refine` | False | True | True | False | True |
| `parent_strategy` | softmax | softmax | softmax | uniform | softmax |
| `final_select` | diverse | diverse | energy | diverse | diverse |

*Analogue mode uses `generate_conformers_from_pose`; seeding parameters are unused because ETKDG is skipped.*

## How It Works

### 1. Seeding

Generates initial conformers using RDKit's ETKDGv3 algorithm. The seed count is computed automatically from molecular topology. For molecules with small non-aromatic rings, ETKDGv3's `useSmallRingTorsions` is enabled; for macrocycles (rings ≥ 10 atoms), `useMacrocycleTorsions` is enabled, applying crystallography-derived distance bounds for better ring-closure geometries.

### 2. Hybrid Exploration

The default "hybrid" strategy combines:

- Torsion library: 365 crystallography-derived SMARTS rules (RDKit CrystalFF, Riniker & Landrum 2016) with Boltzmann-weighted angle preferences
- MCMM moves: random torsion perturbations biased by the library
- Correlated moves: simultaneous changes to adjacent rotors
- Ring flip moves: SVD plane-reflection of non-aromatic 5–7-membered rings to sample chair/envelope inversions
- Global shakes: periodic large perturbations to escape local basins

### 3. Minimization

Each proposed conformer is minimized with MMFF94s to ensure physically reasonable geometries.

### 4. Deduplication

Uses PRISM Pruner for efficient duplicate removal via moment-of-inertia filtering followed by RMSD-based pruning.

### 5. Selection

Final selection returns the lowest-energy conformers after PRISM deduplication.

For the full algorithm description and parameter tuning guide, see [SCIENCE.md](SCIENCE.md).

## Benchmarks

Validated on the [Iridium benchmark](docs/benchmark_report.md) (120 drug-like molecules, bioactive conformer recovery from crystal structures). At N=200, openconf achieves a median best-RMSD of 0.58 Å vs. 0.63 Å for ETKDG+MMFF94s, at 10–15× lower wall time. The advantage is concentrated in flexible molecules (7–9 rotatable bonds), where openconf's torsion-library biasing and ring flip moves outperform pure distance-geometry seeding.

openconf is not recommended for macrocycles (ring size ≥ 12). On macrocyclic ring systems, ETKDG+MMFF94s with a large conformer budget outperforms openconf in both RMSD and ensemble coverage metrics. ETKDGv3 has dedicated macrocycle distance-geometry bounds that openconf does not replicate.

## API Reference

### Main Functions

- `generate_conformers(mol, method="hybrid", config=None)` - Main entry point
- `generate_conformers_from_smiles(smiles, ...)` - Convenience wrapper
- `generate_conformers_from_pose(mol, constrained_atoms, config=None)` - FEP-style analogue generation from an aligned pose

### Configuration Classes

- `ConformerConfig` - Main configuration
- `PrismConfig` - PRISM Pruner settings
- `ConstraintSpec` - Positional constraints for pose-locked generation

### Data Classes

- `ConformerEnsemble` - Container for conformers and metadata
- `ConformerRecord` - Per-conformer metadata

### Lower-Level Components

- `prepare_molecule(mol)` - Sanitize and add hydrogens
- `build_rotor_model(mol)` - Identify rotatable bonds and flippable rings
- `TorsionLibrary` - 365 crystallography-derived SMARTS torsion rules; load custom rules with `TorsionLibrary.from_json(path)`
- `prism_dedupe(mol, conf_ids, config)` - Deduplication

## Dependencies

- RDKit >= 2022.03
- NumPy >= 1.20
- prism-pruner >= 0.0.3

## License

MIT License

## Citation

If you use openconf in your research, please cite:

```bibtex
@software{openconf,
  title = {openconf: Modular conformer generation for docking and ensemble workflows},
  year = {2026},
  url = {https://github.com/rowansci/openconf}
}
```

## Acknowledgments

- [PRISM Pruner](https://github.com/ntampellini/prism_pruner) by Nicolò Tampellini for efficient conformer deduplication
- [RDKit](https://www.rdkit.org/) for cheminformatics infrastructure and the CrystalFF torsion library (Riniker & Landrum, *J. Chem. Inf. Model.* 56, 2016)
