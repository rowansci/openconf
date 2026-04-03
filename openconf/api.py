"""Main API for openconf conformer generation."""

from dataclasses import dataclass

from rdkit import Chem

from .config import ConformerConfig, ConformerPreset, ConstraintSpec, preset_config
from .perceive import build_rotor_model, prepare_molecule
from .pool import ConformerRecord
from .propose.hybrid import run_hybrid_generation


@dataclass
class ConformerEnsemble:
    """Collection of conformers with metadata.

    Attributes:
        mol: RDKit molecule containing all conformers.
        records: List of ConformerRecord objects.
    """

    mol: Chem.Mol
    records: list[ConformerRecord]

    @property
    def conf_ids(self) -> list[int]:
        """List of conformer IDs."""
        return [r.conf_id for r in self.records]

    @property
    def energies(self) -> list[float]:
        """List of energies."""
        return [r.energy_kcal or float("inf") for r in self.records]

    @property
    def n_conformers(self) -> int:
        """Number of conformers."""
        return len(self.records)

    def subset(self, indices: list[int]) -> "ConformerEnsemble":
        """Get a subset of conformers.

        Args:
            indices: Indices of conformers to keep.

        Returns:
            New ConformerEnsemble with selected conformers.
        """
        new_records = [self.records[i] for i in indices]
        return ConformerEnsemble(mol=self.mol, records=new_records)

    def coords(self, idx: int) -> list[tuple[float, float, float]]:
        """Get coordinates for a conformer by index.

        Args:
            idx: Index into records list.

        Returns:
            List of (x, y, z) tuples.
        """
        conf_id = self.records[idx].conf_id
        conf = self.mol.GetConformer(conf_id)

        coords = []
        for i in range(self.mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append((pos.x, pos.y, pos.z))

        return coords

    def to_sdf(self, output_path: str, include_metadata: bool = True) -> None:
        """Write ensemble to SDF file.

        Args:
            output_path: Output file path.
            include_metadata: Include energy and source in SDF properties.
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
            output_path: Output file path.
        """
        from .io import write_xyz

        write_xyz(
            self.mol,
            self.conf_ids,
            output_path,
            energies=self.energies,
        )

    def summary(self) -> str:
        """Get a summary of the ensemble."""
        from .io import get_conformer_summary

        return get_conformer_summary(
            self.mol,
            self.conf_ids,
            self.energies,
        )


def generate_conformers(
    mol: Chem.Mol | str,
    method: str = "hybrid",
    config: ConformerConfig | None = None,
    preset: ConformerPreset | None = None,
) -> ConformerEnsemble:
    """Generate a diverse conformer ensemble.

    This is the main entry point for conformer generation. It uses
    RDKit ETKDG for seeding, applies torsion-biased moves for exploration,
    minimizes with MMFF, and deduplicates with PRISM Pruner.

    Args:
        mol: RDKit molecule or SMILES string.
        method: Generation method ("hybrid" is the default and recommended).
        config: Configuration options. If None, uses defaults.
        preset: Named use-case preset. One of ``"rapid"``,
            ``"ensemble"``, ``"spectroscopic"``, ``"docking"``. Mutually
            exclusive with *config*; raises ValueError if both are supplied.

    Returns:
        Generated conformers with metadata.

    Raises:
        ValueError: If mol is invalid, method is unknown, or both *config*
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

    # Handle SMILES input
    if isinstance(mol, str):
        from .io import smiles_to_mol

        mol = smiles_to_mol(mol)

    # Resolve config
    if preset is not None:
        config = preset_config(preset)
    elif config is None:
        config = ConformerConfig()

    # Prepare molecule
    mol = prepare_molecule(mol, add_hs=True)

    # Build rotor model
    rotor_model = build_rotor_model(mol)

    # Run generation based on method
    if method == "hybrid":
        mol, conf_ids, energies = run_hybrid_generation(mol, rotor_model, config)
    else:
        raise ValueError(f"Unknown method: {method}. Available: 'hybrid'")

    # Build ensemble
    records = [
        ConformerRecord(
            conf_id=cid,
            energy_kcal=energy,
            source=method,
        )
        for cid, energy in zip(conf_ids, energies, strict=True)
    ]

    return ConformerEnsemble(mol=mol, records=records)


def generate_conformers_from_pose(
    mol: Chem.Mol,
    constrained_atoms: list[int] | frozenset[int],
    config: ConformerConfig | None = None,
    preset: ConformerPreset | None = None,
) -> ConformerEnsemble:
    """Generate conformers for an MCS-aligned pose, keeping core atoms fixed.

    Designed for FEP-style analogue generation: you supply a molecule with an
    existing conformer (e.g. the result of MCS alignment) and the indices of
    the atoms that must remain fixed (the MCS core / scaffold). Only the free
    terminal rotors are explored, so the search is faster and stays consistent
    with the bound pose.

    **Atom index convention:** pass indices as they appear in *mol*. If *mol*
    already has explicit hydrogens the indices are used as-is. If *mol* has only
    heavy atoms, ``Chem.AddHs`` is called internally — it appends new H atoms at
    the end and leaves all existing indices unchanged, so heavy-atom indices
    remain valid after H addition.

    Args:
        mol: RDKit molecule with at least one conformer (the MCS-aligned pose).
            If multiple conformers are present, the first is used as the seed.
        constrained_atoms: Atom indices of the core scaffold that must not move.
            These are indices into *mol* as supplied (see note above).
        config: Configuration options. Defaults to the ``"analogue"`` preset.
        preset: Named preset. Defaults to ``"analogue"`` when neither *config*
            nor *preset* is given. Mutually exclusive with *config*.

    Returns:
        ConformerEnsemble with terminal-group conformational diversity while
        preserving the input core geometry.

    Raises:
        ValueError: If mol has no conformers, if both *config* and *preset* are
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

    # Resolve config — default to "analogue" preset for this entry point.
    if config is not None:
        resolved_config = ConformerConfig(**{
            k: v for k, v in config.__dict__.items() if k != "constraint_spec"
        })
    elif preset is not None:
        resolved_config = preset_config(preset)
    else:
        resolved_config = preset_config("analogue")

    # Attach the constraint spec (overrides any constraint_spec already in config).
    resolved_config.constraint_spec = ConstraintSpec(
        constrained_atoms=frozenset(constrained_atoms)
    )

    # Prepare molecule — AddHs is a no-op if Hs are already present.
    prepped_mol = prepare_molecule(mol, add_hs=True)

    rotor_model = build_rotor_model(prepped_mol)

    prepped_mol, conf_ids, energies = run_hybrid_generation(prepped_mol, rotor_model, resolved_config)

    records = [
        ConformerRecord(
            conf_id=cid,
            energy_kcal=energy,
            source="analogue",
        )
        for cid, energy in zip(conf_ids, energies, strict=True)
    ]

    return ConformerEnsemble(mol=prepped_mol, records=records)


def generate_conformers_from_smiles(
    smiles: str,
    method: str = "hybrid",
    config: ConformerConfig | None = None,
) -> ConformerEnsemble:
    """Generate conformers from a SMILES string.

    Convenience function that wraps generate_conformers.

    Args:
        smiles: SMILES string.
        method: Generation method.
        config: Configuration options.

    Returns:
        ConformerEnsemble containing the generated conformers.
    """
    return generate_conformers(smiles, method=method, config=config)
