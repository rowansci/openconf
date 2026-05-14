"""Main API for openconf conformer generation."""

import dataclasses
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from rdkit import Chem

from .config import ConformerConfig, ConformerPreset, ConstraintSpec, preset_config
from .perceive import (
    StereoSignature,
    build_rotor_model,
    conformer_matches_specified_stereochemistry,
    prepare_molecule,
    specified_stereochemistry,
)
from .pool import ConformerRecord
from .propose.hybrid import run_hybrid_generation, run_low_flex_generation
from .propose.stats import GenerationStat
from .torsionlib import TorsionLibrary
from .tuning import get_runtime_tuning

# Gas constant in kcal/(mol·K); kT at 298.15 K ≈ 0.5924 kcal/mol.
_R_KCAL_PER_MOL_K = 1.987204e-3


def _filter_stereochemistry_consistent_conformers(
    mol: Chem.Mol,
    conf_ids: list[int],
    energies: list[float],
    reference_stereo: StereoSignature,
) -> tuple[list[int], list[float]]:
    """Filter final conformers that changed input-specified stereochemistry.

    Args:
        mol: molecule containing final conformers
        conf_ids: final conformer IDs
        energies: energies aligned to `conf_ids`
        reference_stereo: input-specified stereochemistry labels

    Returns:
        Filtered conformer IDs and energies

    Raises:
        ValueError: all final conformers changed specified stereochemistry
    """
    if not reference_stereo.tetrahedral and not reference_stereo.bonds:
        return conf_ids, energies

    kept_ids: list[int] = []
    kept_energies: list[float] = []
    rejected_ids: list[int] = []
    for conf_id, energy in zip(conf_ids, energies, strict=True):
        if conformer_matches_specified_stereochemistry(mol, conf_id, reference_stereo):
            kept_ids.append(conf_id)
            kept_energies.append(energy)
        else:
            rejected_ids.append(conf_id)

    for conf_id in rejected_ids:
        mol.RemoveConformer(conf_id)

    if conf_ids and not kept_ids:
        raise ValueError("All generated conformers changed input-specified stereochemistry.")

    return kept_ids, kept_energies


@dataclass
class ConformerEnsemble:
    """Collection of conformers with metadata.

    Attributes:
        mol: molecule containing all conformers
        records: conformer metadata records
        generation_stats: benchmark timings and counters collected
            during generation when `ConformerConfig.collect_stats` is enabled.
    """

    mol: Chem.Mol
    records: list[ConformerRecord]
    generation_stats: dict[str, GenerationStat] = field(default_factory=dict)

    @property
    def conf_ids(self) -> list[int]:
        """Conformer IDs in record order."""
        return [r.conf_id for r in self.records]

    @property
    def energies(self) -> list[float]:
        """Energies in record order."""
        return [r.energy_kcal if r.energy_kcal is not None else float("inf") for r in self.records]

    @property
    def n_conformers(self) -> int:
        """Conformer count."""
        return len(self.records)

    def coords(self, idx: int) -> list[tuple[float, float, float]]:
        """Get coordinates for a conformer by index.

        Args:
            idx: position in `records`

        Returns:
            Coordinates for each atom
        """
        conf = self.mol.GetConformer(self.records[idx].conf_id)
        return [(p.x, p.y, p.z) for p in (conf.GetAtomPosition(i) for i in range(self.mol.GetNumAtoms()))]

    def to_sdf(self, output_path: str, include_metadata: bool = True) -> None:
        """Write ensemble to SDF file.

        Args:
            output_path: destination file path
            include_metadata: include energy and source in SDF properties
        """
        from .io import write_sdf

        metadata = None
        if include_metadata:
            metadata = {r.conf_id: {"source": r.source, **r.tags} for r in self.records}

        write_sdf(
            self.mol,
            self.conf_ids,
            output_path,
            energies=self.energies,
            metadata=metadata,
        )

    def to_xyz(self, output_path: str) -> None:
        """Write ensemble to XYZ file.

        Args:
            output_path: destination file path
        """
        from .io import write_xyz

        write_xyz(
            self.mol,
            self.conf_ids,
            output_path,
            energies=self.energies,
        )

    def boltzmann_weights(self, temperature: float = 298.15) -> np.ndarray:
        """Normalized Boltzmann weights from the ensemble energies.

        Weights are `exp(-(E - Emin) / RT)` normalized to sum to 1. Conformers
        with missing energies are assigned weight 0.

        Args:
            temperature: temperature in Kelvin; default 298.15 K (25 °C)

        Returns:
            Normalized weights aligned with `records`

        Raises:
            ValueError: temperature is not positive
            ValueError: ensemble has no conformers with finite energies
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        energies = np.array(
            [r.energy_kcal if r.energy_kcal is not None else np.inf for r in self.records],
            dtype=float,
        )
        finite = np.isfinite(energies)
        if not finite.any():
            raise ValueError("Cannot compute Boltzmann weights: no conformer has a finite energy.")

        shifted = energies - energies[finite].min()
        weights = np.zeros_like(energies)
        weights[finite] = np.exp(-shifted[finite] / (_R_KCAL_PER_MOL_K * temperature))
        weights /= weights.sum()
        return weights

    def rmsd_to(self, ref_idx: int = 0, heavy_only: bool = True) -> list[float]:
        """Symmetry-corrected RMSDs of every conformer to a reference.

        Uses :func:`rdkit.Chem.rdMolAlign.GetBestRMS`, which searches over
        graph automorphisms and optimally superimposes each pair before
        measuring RMSD. The ensemble's own conformer coordinates are not
        modified.

        Args:
            ref_idx: position in `records` of reference conformer
            heavy_only: use heavy atoms only

        Returns:
            RMSDs in Å aligned with `records`; reference entry is 0.0
        """
        from rdkit.Chem import rdMolAlign

        if not 0 <= ref_idx < self.n_conformers:
            raise IndexError(f"ref_idx {ref_idx} out of range for ensemble of size {self.n_conformers}")

        # GetBestRMS aligns prbMol in place — work on a copy to keep the
        # ensemble's stored coordinates unchanged.
        mol_copy = Chem.Mol(self.mol)

        atom_map = None
        if heavy_only:
            heavy = [a.GetIdx() for a in mol_copy.GetAtoms() if a.GetAtomicNum() > 1]
            atom_map = [list(zip(heavy, heavy, strict=True))]

        ref_id = self.records[ref_idx].conf_id
        rmsds: list[float] = []
        for i, record in enumerate(self.records):
            if i == ref_idx:
                rmsds.append(0.0)
                continue
            rmsd = rdMolAlign.GetBestRMS(
                mol_copy,
                mol_copy,
                prbId=record.conf_id,
                refId=ref_id,
                map=atom_map,
            )
            rmsds.append(rmsd)
        return rmsds

    def pairwise_rmsd(self, heavy_only: bool = True) -> np.ndarray:
        """Symmetric pairwise RMSD matrix for all conformers.

        Uses :func:`rdkit.Chem.rdMolAlign.GetBestRMS` with optimal superposition
        per pair. Cost is O(N² · automorphisms); expensive for large ensembles
        of highly symmetric molecules. The ensemble's stored coordinates are
        not modified.

        Args:
            heavy_only: use heavy atoms only

        Returns:
            Symmetric RMSD matrix aligned with `records`
        """
        from rdkit.Chem import rdMolAlign

        n = self.n_conformers
        matrix = np.zeros((n, n), dtype=float)
        if n <= 1:
            return matrix

        mol_copy = Chem.Mol(self.mol)

        atom_map = None
        if heavy_only:
            heavy = [a.GetIdx() for a in mol_copy.GetAtoms() if a.GetAtomicNum() > 1]
            atom_map = [list(zip(heavy, heavy, strict=True))]

        conf_ids = [r.conf_id for r in self.records]
        for i in range(n):
            for j in range(i + 1, n):
                rmsd = rdMolAlign.GetBestRMS(
                    mol_copy,
                    mol_copy,
                    prbId=conf_ids[j],
                    refId=conf_ids[i],
                    map=atom_map,
                )
                matrix[i, j] = rmsd
                matrix[j, i] = rmsd
        return matrix

    def summary(self) -> str:
        """Get a summary of the ensemble."""
        from .io import get_conformer_summary

        return get_conformer_summary(
            self.mol,
            self.conf_ids,
            self.energies,
        )

    @classmethod
    def from_sdf(cls, input_path: str) -> Self:
        """Read an ensemble from an SDF file, preserving metadata.

        Round-trip inverse of :meth:`to_sdf`. Recovers per-conformer
        `energy_kcal` (from the `Energy_kcal` property), `source` (from
        the `source` property), and any additional tags written by
        :meth:`to_sdf`. The `_Name` and `ConfID` bookkeeping properties
        produced by the writer are ignored.

        Args:
            input_path: SDF file path

        Returns:
            Ensemble with one record per conformer in file
        """
        supplier = Chem.SDMolSupplier(str(input_path), removeHs=False)

        _RESERVED = {"_Name", "ConfID", "Energy_kcal"}

        mol: Chem.Mol | None = None
        records: list[ConformerRecord] = []

        for mol_i in supplier:
            if mol_i is None:
                continue

            if mol is None:
                mol = mol_i
                conf_id = mol.GetConformer().GetId()
            else:
                conf = mol_i.GetConformer()
                conf_id = mol.AddConformer(conf, assignId=True)

            energy: float | None = None
            if mol_i.HasProp("Energy_kcal"):
                try:
                    energy = float(mol_i.GetProp("Energy_kcal"))
                except ValueError:
                    energy = None

            source = mol_i.GetProp("source") if mol_i.HasProp("source") else "unknown"

            tags = {k: mol_i.GetProp(k) for k in mol_i.GetPropNames() if k not in _RESERVED and k != "source"}

            records.append(
                ConformerRecord(
                    conf_id=conf_id,
                    energy_kcal=energy,
                    source=source,
                    tags=tags,
                )
            )

        if mol is None:
            raise ValueError(f"No valid molecules in {input_path}")

        return cls(mol=mol, records=records)


def generate_conformers(
    mol: Chem.Mol | str,
    method: str = "hybrid",
    config: ConformerConfig | None = None,
    preset: ConformerPreset | None = None,
    torsion_library: TorsionLibrary | None = None,
    add_hs: bool = True,
) -> ConformerEnsemble:
    """Generate a diverse conformer ensemble.

    This is the main entry point for conformer generation. It uses
    RDKit ETKDG for seeding, applies torsion-biased moves for exploration,
    minimizes with MMFF, and deduplicates with PRISM Pruner.

    Args:
        mol: molecule or SMILES string
        method: generation method; `"hybrid"` is the default and recommended
        config: configuration options; defaults are used when omitted
        preset: named use-case preset; one of `"rapid"`,
            `"ensemble"`, `"spectroscopic"`, `"docking"`. Mutually
            exclusive with *config*; raises ValueError if both are supplied.
        torsion_library: torsion library override; when omitted, uses
            the bundled cached CrystalFF-derived library.
        add_hs: add explicit hydrogens before embedding; set to
            `False` when *mol* already has all hydrogens present (e.g. a
            3D structure read from SDF, or a radical with explicit H atoms)
            to prevent RDKit from inserting additional implicit H atoms.

    Returns:
        Generated conformers with metadata

    Raises:
        ValueError: mol is invalid, method is unknown, or both *config*
            and *preset* are supplied.

    Examples:
        >>> from rdkit import Chem
        >>> from openconf import generate_conformers, ConformerConfig
        >>> mol = Chem.MolFromSmiles("CCCCc1ccccc1")
        >>> ensemble = generate_conformers(mol, preset="docking")  # doctest: +SKIP
        >>> ensemble = generate_conformers(mol, config=ConformerConfig(max_out=100))  # doctest: +SKIP
    """
    if config is not None and preset is not None:
        raise ValueError("Specify at most one of 'config' or 'preset', not both.")

    if isinstance(mol, str):
        from .io import smiles_to_mol

        mol = smiles_to_mol(mol)

    if preset is not None:
        config = preset_config(preset)
    elif config is None:
        config = ConformerConfig()

    mol = prepare_molecule(mol, add_hs=add_hs)
    reference_stereo = specified_stereochemistry(mol)
    rotor_model = build_rotor_model(mol)

    if method == "hybrid":
        low_flex_tuning = get_runtime_tuning().low_flex_path
        use_low_flex_path = (
            config.constraint_spec is None
            and rotor_model.n_rotatable <= low_flex_tuning.max_rotatable
            and (low_flex_tuning.allow_macrocycles or not rotor_model.ring_info.get("has_macrocycle"))
            and (low_flex_tuning.allow_rings or not rotor_model.ring_info.get("ring_sizes"))
        )
        runner = run_low_flex_generation if use_low_flex_path else run_hybrid_generation
        mol, conf_ids, energies, generation_stats = runner(mol, rotor_model, config, torsion_library=torsion_library)
    else:
        raise ValueError(f"Unknown method: {method}. Available: 'hybrid'")

    conf_ids, energies = _filter_stereochemistry_consistent_conformers(mol, conf_ids, energies, reference_stereo)

    records = [
        ConformerRecord(
            conf_id=cid,
            energy_kcal=energy,
            source=method,
        )
        for cid, energy in zip(conf_ids, energies, strict=True)
    ]

    return ConformerEnsemble(mol=mol, records=records, generation_stats=generation_stats)


def generate_conformers_from_pose(
    mol: Chem.Mol,
    constrained_atoms: list[int] | frozenset[int],
    config: ConformerConfig | None = None,
    preset: ConformerPreset | None = None,
    torsion_library: TorsionLibrary | None = None,
) -> ConformerEnsemble:
    """Generate conformers for an MCS-aligned pose, keeping core atoms fixed.

    Designed for FEP-style analogue generation: you supply a molecule with an
    existing conformer (e.g. the result of MCS alignment) and the indices of
    the atoms that must remain fixed (the MCS core / scaffold). Only the free
    terminal rotors are explored, so the search is faster and stays consistent
    with the bound pose.

    **Atom index convention:** pass indices as they appear in *mol*. If *mol*
    already has explicit hydrogens the indices are used as-is. If *mol* has only
    heavy atoms, `Chem.AddHs` is called internally — it appends new H atoms at
    the end and leaves all existing indices unchanged, so heavy-atom indices
    remain valid after H addition.

    Args:
        mol: molecule with at least one conformer; when multiple conformers
            are present, first conformer is used as seed.
        constrained_atoms: atom indices of the core scaffold that must not move.
            These are indices into *mol* as supplied (see note above).
        config: configuration options; defaults to the `"analogue"` preset.
        preset: named preset; defaults to `"analogue"` when neither *config*
            nor *preset* is given. Mutually exclusive with *config*.
        torsion_library: torsion library override; when omitted, uses
            the bundled cached CrystalFF-derived library.

    Returns:
        Ensemble with terminal-group conformational diversity while
        preserving the input core geometry.

    Raises:
        ValueError: mol has no conformers, both *config* and *preset* are
            supplied, or if the method is unknown.

    Examples:
        >>> from rdkit import Chem
        >>> from openconf import generate_conformers_from_pose
        >>> mol = Chem.MolFromSmiles("CCCc1ccccc1")
        >>> # (assume mol already has an aligned conformer)
        >>> ensemble = generate_conformers_from_pose(  # doctest: +SKIP
        ...     mol, constrained_atoms=[4, 5, 6, 7, 8, 9]
        ... )
    """
    if config is not None and preset is not None:
        raise ValueError("Specify at most one of 'config' or 'preset', not both.")

    if mol.GetNumConformers() == 0:
        raise ValueError(
            "mol must have at least one conformer for pose-constrained generation. "
            "Supply the MCS-aligned pose as conformer 0."
        )

    if config is not None:
        resolved_config = dataclasses.replace(config, constraint_spec=None)
    elif preset is not None:
        resolved_config = preset_config(preset)
    else:
        resolved_config = preset_config("analogue")

    resolved_config.constraint_spec = ConstraintSpec(constrained_atoms=frozenset(constrained_atoms))
    prepped_mol = prepare_molecule(mol, add_hs=True)
    reference_stereo = specified_stereochemistry(prepped_mol)

    rotor_model = build_rotor_model(prepped_mol)

    prepped_mol, conf_ids, energies, generation_stats = run_hybrid_generation(
        prepped_mol,
        rotor_model,
        resolved_config,
        torsion_library=torsion_library,
    )
    conf_ids, energies = _filter_stereochemistry_consistent_conformers(
        prepped_mol,
        conf_ids,
        energies,
        reference_stereo,
    )

    records = [
        ConformerRecord(
            conf_id=cid,
            energy_kcal=energy,
            source="analogue",
        )
        for cid, energy in zip(conf_ids, energies, strict=True)
    ]

    return ConformerEnsemble(mol=prepped_mol, records=records, generation_stats=generation_stats)
