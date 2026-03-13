# OpenConf Macrocycle / Cyclic-Peptide Benchmark Report

**Dataset:** MPCONF196GEN
**Date:** 2026-03-12
**Commit:** `e7b7ba2`
**Methods evaluated:** etkdg\_raw, etkdg\_mmff, openconf
**Budgets:** N = 50, 200 conformers
**Seeds:** 3 per (method, budget, molecule)

---

## 1. Dataset and Motivation

The [MPCONF196GEN benchmark](https://github.com/rowansci/MPCONF196GEN-benchmark) is a publicly available collection of macrocycles and cyclic peptides assembled by Riniker et al. Each entry provides a CREST iMTD-GC conformational ensemble generated at the GFN2-xTB level of theory. Unlike the Iridium benchmark (crystal structures → single bioactive conformer), MPCONF196GEN evaluates *landscape coverage*: can a method find the low-energy conformers that a high-quality quantum-mechanics–guided search finds?

The dataset used here comprises **13 molecules** in two classes:

| Class | Molecules | Description |
|-------|-----------|-------------|
| Macrocycles | 8 | Ring size ≥ 12; CAMVES\_I, CHPSAR\_I, COHVAW\_I, GS464992\_I, GS557577\_I, POXTRD\_I, SANGLI\_I, YIVNOG\_I |
| Linear / cyclic peptides | 5 | FGG55, GFA01, GGF01, WG01, WGG01 |

---

## 2. Methodology

### 2.1 Reference ensemble construction

For each molecule the CREST `crest_conformers.xyz` multi-frame file was fetched directly from the GitHub repository. Frames were sorted by GFN2-xTB energy (Eh, from the XYZ comment line) and the lowest-energy **20** were retained as the reference ensemble. Fewer frames are used when the CREST run produced fewer than 20 distinct conformers.

XYZ frames were converted to RDKit `Mol` objects using `rdDetermineBonds.DetermineBonds` (distance-based bond inference), followed by sanitization and hydrogen removal. Any frame that failed to sanitize was silently discarded; the remaining frames form the reference set `R`.

### 2.2 Methods

All three methods share the benchmark infrastructure from `benchmark_lib.py`:

| Method | Description |
|--------|-------------|
| **etkdg\_raw** | RDKit ETKDGv3 (seed + 1 to avoid degeneracy at seed=0), no post-embedding minimization |
| **etkdg\_mmff** | ETKDGv3 + MMFF94s minimization on every embedded conformer before pruning |
| **openconf** | OpenConf hybrid proposer with `energy_window_kcal=18.0`, `parent_strategy="uniform"`, `final_select="diverse"`, `do_final_refine=True` |

After generation, **shared postprocessing** identical to the Iridium benchmark is applied:

1. Canonical MMFF94s minimization (ε = 1.0, 500 steps) on every raw conformer
2. Greedy RMSD pruning at **1.0 Å** (sorted by energy, discard if nearest retained < threshold)

The postprocessed set is the *generated ensemble* `G` used for all metric calculations.

### 2.3 Metrics

Unlike the Iridium benchmark (single nearest-crystal-conformer RMSD), MPCONF196GEN uses *coverage* metrics:

**coverage@X** — fraction of reference conformers in `R` that have at least one generated conformer within X Å RMSD:

```
coverage@X = |{r ∈ R : min_{g ∈ G} RMSD(g, r) < X}| / |R|
```

**best\_ref\_rmsd** — minimum RMSD from any generated conformer to any reference conformer:

```
best_ref_rmsd = min_{g ∈ G, r ∈ R} RMSD(g, r)
```

RMSD is computed with `GetBestRMS` (RDKit, symmetry-corrected via graph automorphisms). When direct atom-mapping fails (e.g. aromatic bond order disagreement between XYZ and SMILES-derived mol), a maximum common substructure (MCS) fallback with manual atom-map alignment is attempted.

Coverage is reported at thresholds **1.0, 1.5, 2.0, and 3.0 Å**.

---

## 3. Results

### 3.1 Overall (all 13 molecules)

| Budget | Method | Median RMSD | Cov@1.0% | Cov@1.5% | Cov@2.0% | Cov@3.0% | Median t (s) | Med Uniq |
|--------|--------|-------------|-----------|-----------|-----------|-----------|--------------|----------|
| N=50 | etkdg\_raw | 1.221 | 3.8 | 48.3 | 92.0 | 99.2 | 2.23 | 45 |
| N=50 | etkdg\_mmff | 0.888 | 10.5 | 61.5 | 97.4 | 100.0 | 4.84 | 46 |
| N=50 | **openconf** | 0.855 | 10.4 | 59.6 | 92.1 | 96.2 | 1.42 | 46 |
| N=200 | etkdg\_raw | 1.185 | 4.5 | 58.8 | 83.9 | 99.9 | 9.71 | 181 |
| N=200 | etkdg\_mmff | 0.888 | 14.6 | 79.3 | 93.5 | 100.0 | 19.14 | 188 |
| N=200 | **openconf** | 0.970 | 13.4 | 64.4 | 83.7 | 91.7 | 1.69 | 130 |

### 3.2 Stratified by molecule class

**Macrocycles (n=8):**

| Budget | Method | Median RMSD | Cov@1.5% | Cov@2.0% | Cov@3.0% |
|--------|--------|-------------|-----------|-----------|-----------|
| N=50 | etkdg\_raw | 1.293 | 38.3 | 88.5 | 99.8 |
| N=50 | etkdg\_mmff | 1.020 | 50.8 | 96.2 | 100.0 |
| N=50 | **openconf** | 1.293 | 38.1 | 87.5 | 87.5 |
| N=200 | etkdg\_raw | 1.247 | 43.3 | 74.2 | 99.8 |
| N=200 | etkdg\_mmff | **0.971** | **66.5** | **89.4** | **100.0** |
| N=200 | **openconf** | 1.293 | 38.5 | 69.8 | 87.5 |

**Linear peptides (n=5):**

| Budget | Method | Median RMSD | Cov@1.5% | Cov@2.0% | Cov@3.0% |
|--------|--------|-------------|-----------|-----------|-----------|
| N=50 | etkdg\_raw | 0.974 | 61.3 | 93.3 | 100.0 |
| N=50 | etkdg\_mmff | 0.443 | 90.0 | 100.0 | 100.0 |
| N=50 | **openconf** | **0.351** | **91.0** | **99.3** | **100.0** |
| N=200 | etkdg\_raw | 0.882 | 84.0 | 98.3 | 100.0 |
| N=200 | etkdg\_mmff | 0.439 | 100.0 | 100.0 | 100.0 |
| N=200 | **openconf** | **0.340** | **100.0** | **100.0** | **100.0** |

### 3.3 Per-molecule detail (N=200)

| Molecule | RB | Macro | etkdg\_raw | etkdg\_mmff | openconf |
|----------|----|-------|------------|-------------|----------|
| CAMVES\_I | 0 | Y | 1.309 / 100% | 0.888 / 100% | 1.278 / 83% |
| CHPSAR\_I | 0 | Y | 1.185 / 100% | 0.972 / 100% | 1.309 / 100% |
| COHVAW\_I | 6 | Y | 0.809 / 100% | 0.971 / 100% | 0.855 / 100% |
| FGG55 | 7 | N | 1.036 / 100% | 0.613 / 100% | 0.508 / 100% |
| GFA01 | 7 | N | 0.966 / 100% | 0.433 / 100% | **0.426 / 100%** |
| GGF01 | 7 | N | 0.708 / 100% | 0.293 / 100% | **0.206 / 100%** |
| GS464992\_I | 8 | Y | 1.134 / 92% | **0.360 / 100%** | 0.584 / 75% |
| GS557577\_I | 4 | Y | 1.367 / 100% | 1.147 / 100% | 1.435 / 80% |
| POXTRD\_I | 0 | Y | 0.846 / 100% | **0.667 / 100%** | 0.968 / 100% |
| SANGLI\_I | 6 | Y | 2.253 / 0% | **1.567 / 93%** | 3.009 / 0% |
| WG01 | 5 | N | 0.400 / 100% | 0.573 / 100% | **0.337 / 100%** |
| WGG01 | 7 | N | 0.882 / 92% | 0.439 / 100% | **0.340 / 100%** |
| YIVNOG\_I | 4 | Y | 1.986 / 2% | **1.661 / 22%** | 1.790 / 20% |

*(format: best\_ref\_rmsd Å / cov@2.0 Å%)*

---

## 4. Discussion

### 4.1 Linear peptides: OpenConf is competitive

On the five linear-peptide molecules, OpenConf achieves the best or tied-best median RMSD at both N=50 and N=200 (0.351 Å and 0.340 Å vs. etkdg\_mmff 0.443/0.439 Å). Full 100% coverage at 2 Å and 3 Å at N=200 matches etkdg\_mmff. This is consistent with the Iridium benchmark finding that OpenConf excels on molecules with moderate flexibility (RB 5–9).

### 4.2 Macrocycles: OpenConf struggles, especially at scale

On macrocycles OpenConf is **weaker than etkdg\_mmff across all metrics**, and the gap widens with budget:

- **Coverage saturation**: OpenConf's unique conformer count plateaus around 100–170 even at N=200 (vs. 180–200 for ETKDG methods), because its diversity-guided selection strategy aggressively prunes similar conformers. This is desirable for drug-discovery applications (avoiding redundancy) but penalizes coverage benchmarks where exhaustive landscape sampling is the goal.
- **SANGLI\_I / YIVNOG\_I**: Two macrocycles where OpenConf achieves near-zero coverage at 2 Å. etkdg\_mmff achieves 93% and 22% coverage respectively on the same molecules, suggesting the MMFF force field combined with ETKDGv3's macrocycle-specific sampling does a substantially better job on these structures.
- **Cov@3.0 gap**: On macrocycles at N=200, OpenConf reaches only 87.5% coverage at 3.0 Å vs. 100% for etkdg\_mmff and 99.8% for etkdg\_raw. This suggests some low-energy CREST macrocycle conformers are simply inaccessible to the current OpenConf proposer.

### 4.3 Runtime

OpenConf is dramatically faster, especially at large budgets:

| Method | N=50 median t | N=200 median t |
|--------|--------------|----------------|
| etkdg\_raw | ~3 s | ~10 s |
| etkdg\_mmff | ~5 s | ~20 s |
| openconf | ~1.5 s | ~1.7 s |

OpenConf's runtime is almost budget-independent because it early-terminates via diversity-based selection — it stops generating once the ensemble is sufficiently diverse. This is a deliberate design trade-off.

### 4.4 Why macrocycles are hard for OpenConf

OpenConf's hybrid proposer (torsional + ring-aware moves) was not specifically tuned for large macrocyclic rings. Key issues:

1. **Ring closure constraint**: ETKDGv3 has dedicated macrocycle distance-geometry bounds (added in RDKit 2022.09), which explicitly handle ring-closure geometry. OpenConf currently uses MMFF-minimized small-ring torsional libraries as seed geometries, which may not adequately sample the macrocycle ring flip space.
2. **Energy window mismatch**: The 18 kcal/mol energy window used by OpenConf (with ε=4.0 dielectric) may map differently onto the GFN2-xTB energy landscape used to define the CREST reference. Some CREST conformers that are low-energy by GFN2-xTB may appear high-energy under MMFF/ε=4 and be rejected.
3. **Coverage saturation**: The `final_select="diverse"` strategy is optimized to return a compact, diverse set. On macrocycles with wide, flat energy landscapes, this means fewer conformers are retained — penalizing coverage-based metrics.

### 4.5 Comparison with the Iridium benchmark

| Benchmark | OpenConf vs. etkdg\_mmff | Context |
|-----------|--------------------------|---------|
| Iridium (120 mol, all budgets) | **+5–10% RMSD improvement** | Drug-like molecules, bioactive conformer recovery |
| MPCONF196GEN peptides (n=5) | **Competitive / slight win** | Linear/cyclic peptides, ensemble coverage |
| MPCONF196GEN macrocycles (n=8) | **etkdg\_mmff wins** | Large ring systems, CREST landscape coverage |

---

## 5. Limitations and Caveats

1. **Small macrocycle sample**: Only 8 macrocycles. Variance is high; several conclusions (e.g. the SANGLI\_I / YIVNOG\_I failures) are driven by individual molecules.

2. **Reference quality**: CREST conformers are generated at GFN2-xTB, a semi-empirical method. They are a reasonable but imperfect proxy for the true conformational landscape. Neither MMFF94s nor GFN2-xTB is a gold standard.

3. **Coverage metric vs. single-conformer recovery**: Coverage@X rewards methods that hit every corner of the reference ensemble. For pharmaceutical applications (e.g. docking), recovering the single bioactive conformation matters more — a metric not captured here.

4. **Atom-mapping fidelity**: Converting CREST XYZ frames to RDKit mols via `rdDetermineBonds` occasionally fails or produces bonds inconsistent with the SMILES-derived mol. Such frames are discarded silently, which may bias reference ensembles toward more "MMFF-compatible" conformers.

5. **OpenConf macrocycle support is nascent**: OpenConf currently falls back to UFF for organometallic/exotic atoms. For purely organic macrocycles MMFF94s is used, but without the specialized macrocycle ring-closure geometry that ETKDGv3 uses. This is a known area for future development.

---

## 6. Conclusion

OpenConf is **competitive with etkdg\_mmff on linear and cyclic peptides** and matches the Iridium benchmark's finding of strong performance on flexible drug-like molecules. However, on **macrocyclic ring systems** (ring size ≥ 12), etkdg\_mmff outperforms OpenConf substantially in both RMSD and coverage metrics, while etkdg\_raw provides adequate coverage at lower cost.

The primary bottleneck is not runtime (OpenConf is 10–15x faster than etkdg\_mmff at N=200) but **conformer quality and diversity on macrocyclic ring topologies**. Dedicated macrocycle sampling improvements — such as incorporating ring-flip move types, using ETKDGv3's macrocycle bounds as initial geometries, or widening the energy window for large rings — would likely close this gap.

For immediate recommendations:
- **Peptides / flexible small molecules**: Prefer OpenConf (faster, equivalent or better quality).
- **Macrocycles**: Use etkdg\_mmff with N≥200 until OpenConf macrocycle support matures.
