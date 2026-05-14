"""PRISM Pruner adapter for RMSD-based conformer deduplication."""

import numpy as np
from prism_pruner.pruner import prune_by_moment_of_inertia
from rdkit import Chem


def _mol_to_arrays(
    mol: Chem.Mol,
    conf_ids: list[int],
    use_heavy_atoms_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert RDKit conformers to numpy arrays for PRISM.

    Args:
        mol: molecule with conformers
        conf_ids: conformer IDs to extract
        use_heavy_atoms_only: use heavy atoms only

    Returns:
        Coordinates and atom symbols for PRISM
    """
    # Get atom indices to use
    if use_heavy_atoms_only:
        atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    else:
        atom_indices = list(range(mol.GetNumAtoms()))

    n_atoms = len(atom_indices)
    n_confs = len(conf_ids)

    # Extract coordinates
    coords = np.zeros((n_confs, n_atoms, 3), dtype=np.float64)

    atom_indices_arr = np.array(atom_indices)
    for i, conf_id in enumerate(conf_ids):
        conf = mol.GetConformer(int(conf_id))
        coords[i] = conf.GetPositions()[atom_indices_arr]

    # Get atom symbols as numpy array
    atoms = np.array([mol.GetAtomWithIdx(idx).GetSymbol() for idx in atom_indices])

    return coords, atoms


def prism_dedupe(
    mol: Chem.Mol,
    conf_ids: list[int],
    use_heavy_atoms_only: bool = True,
) -> list[int]:
    """Deduplicate conformers using PRISM Pruner.

    Args:
        mol: molecule with conformers
        conf_ids: conformer IDs to process
        use_heavy_atoms_only: use only heavy atoms for comparison

    Returns:
        Identifiers to keep
    """
    if len(conf_ids) <= 1:
        return conf_ids

    coords, atoms = _mol_to_arrays(mol, conf_ids, use_heavy_atoms_only)
    _, mask = prune_by_moment_of_inertia(coords, atoms)
    return [conf_id for conf_id, keep in zip(conf_ids, mask, strict=True) if keep]
