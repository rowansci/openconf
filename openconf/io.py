"""I/O utilities for conformer ensembles."""

from pathlib import Path
from typing import Any, Sequence

from rdkit import Chem


def write_sdf(
    mol: Chem.Mol,
    conf_ids: Sequence[int],
    output_path: str | Path,
    energies: Sequence[float] | None = None,
    metadata: dict[int, dict[str, Any]] | None = None,
) -> None:
    """Write conformers to an SDF file.

    Args:
        mol: RDKit molecule with conformers.
        conf_ids: List of conformer IDs to write.
        output_path: Output file path.
        energies: Optional energies for each conformer.
        metadata: Optional metadata dict mapping conf_id to properties.
    """
    writer = Chem.SDWriter(str(output_path))

    for i, conf_id in enumerate(conf_ids):
        # Create a copy with just this conformer
        mol_copy = Chem.Mol(mol)

        # Remove all conformers except this one
        conf_ids_to_remove = [c.GetId() for c in mol_copy.GetConformers() if c.GetId() != conf_id]
        for cid in conf_ids_to_remove:
            mol_copy.RemoveConformer(cid)

        # Add properties
        mol_copy.SetProp("_Name", f"conf_{conf_id}")
        mol_copy.SetProp("ConfID", str(conf_id))

        if energies is not None:
            mol_copy.SetProp("Energy_kcal", f"{energies[i]:.4f}")

        if metadata is not None and conf_id in metadata:
            for key, value in metadata[conf_id].items():
                if isinstance(value, float):
                    mol_copy.SetProp(str(key), f"{value:.6f}")
                else:
                    mol_copy.SetProp(str(key), str(value))

        writer.write(mol_copy, confId=conf_id)

    writer.close()


def write_xyz(
    mol: Chem.Mol,
    conf_ids: Sequence[int],
    output_path: str | Path,
    energies: Sequence[float] | None = None,
) -> None:
    """Write conformers to an XYZ file (concatenated).

    Args:
        mol: RDKit molecule with conformers.
        conf_ids: List of conformer IDs to write.
        output_path: Output file path.
        energies: Optional energies for each conformer.
    """
    with open(output_path, "w") as f:
        for i, conf_id in enumerate(conf_ids):
            conf = mol.GetConformer(conf_id)
            n_atoms = mol.GetNumAtoms()

            # Header
            f.write(f"{n_atoms}\n")

            # Comment line with energy if available
            if energies is not None:
                f.write(f"conf_{conf_id} Energy={energies[i]:.6f} kcal/mol\n")
            else:
                f.write(f"conf_{conf_id}\n")

            # Atom coordinates
            for atom_idx in range(n_atoms):
                atom = mol.GetAtomWithIdx(atom_idx)
                pos = conf.GetAtomPosition(atom_idx)
                f.write(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")


def read_sdf(input_path: str | Path) -> tuple[Chem.Mol, list[int], list[float]]:
    """Read conformers from an SDF file.

    Args:
        input_path: Input file path.

    Returns:
        Tuple of (mol, conf_ids, energies). Energies will be empty if not present.
    """
    supplier = Chem.SDMolSupplier(str(input_path), removeHs=False)

    mol = None
    conf_ids: list[int] = []
    energies: list[float] = []

    for mol_i in supplier:
        if mol_i is None:
            continue

        if mol is None:
            mol = mol_i
            conf_ids.append(mol.GetConformer().GetId())
        else:
            # Add conformer to base molecule
            conf = mol_i.GetConformer()
            new_id = mol.AddConformer(conf, assignId=True)
            conf_ids.append(new_id)

        # Get energy if present
        if mol_i.HasProp("Energy_kcal"):
            try:
                energies.append(float(mol_i.GetProp("Energy_kcal")))
            except ValueError:
                energies.append(float("inf"))
        elif mol_i.HasProp("energy"):
            try:
                energies.append(float(mol_i.GetProp("energy")))
            except ValueError:
                energies.append(float("inf"))

    if mol is None:
        raise ValueError(f"No valid molecules in {input_path}")

    return mol, conf_ids, energies


def mol_to_smiles(mol: Chem.Mol, canonical: bool = True) -> str:
    """Convert molecule to SMILES string.

    Args:
        mol: RDKit molecule.
        canonical: Whether to canonicalize.

    Returns:
        SMILES string.
    """
    # Remove Hs for cleaner SMILES
    mol_no_h = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol_no_h, canonical=canonical)


def smiles_to_mol(smiles: str, add_hs: bool = True) -> Chem.Mol:
    """Convert SMILES to molecule.

    Args:
        smiles: SMILES string.
        add_hs: Whether to add hydrogens.

    Returns:
        RDKit molecule.

    Raises:
        ValueError: If SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    if add_hs:
        mol = Chem.AddHs(mol)

    return mol


def get_conformer_summary(
    mol: Chem.Mol,
    conf_ids: Sequence[int],
    energies: Sequence[float],
) -> str:
    """Get a summary string for a conformer ensemble.

    Args:
        mol: RDKit molecule.
        conf_ids: List of conformer IDs.
        energies: List of energies.

    Returns:
        Summary string.
    """
    n_confs = len(conf_ids)
    n_atoms = mol.GetNumAtoms()
    n_heavy = mol.GetNumHeavyAtoms()

    if energies:
        min_e = min(energies)
        max_e = max(energies)
        mean_e = sum(energies) / len(energies)
        spread = max_e - min_e
    else:
        min_e = max_e = mean_e = spread = float("nan")

    lines = [
        "Conformer Ensemble Summary",
        "=" * 40,
        f"SMILES: {mol_to_smiles(mol)}",
        f"Atoms: {n_atoms} ({n_heavy} heavy)",
        f"Conformers: {n_confs}",
        f"Energy range: {min_e:.2f} - {max_e:.2f} kcal/mol",
        f"Mean energy: {mean_e:.2f} kcal/mol",
        f"Energy spread: {spread:.2f} kcal/mol",
    ]

    return "\n".join(lines)
