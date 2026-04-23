"""PRISM Pruner adapter for conformer deduplication.

Provides a clean interface to PRISM Pruner for removing duplicate conformers.
"""

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
        mol: RDKit molecule with conformers.
        conf_ids: List of conformer IDs to extract.
        use_heavy_atoms_only: If True, only use heavy atoms.

    Returns:
        Tuple of (coords array [n_confs, n_atoms, 3], atom symbols array).
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
        mol: RDKit molecule with conformers.
        conf_ids: List of conformer IDs to process.
        use_heavy_atoms_only: Use only heavy atoms for comparison.

    Returns:
        List of conformer IDs to keep.
    """
    if len(conf_ids) <= 1:
        return conf_ids

    coords, atoms = _mol_to_arrays(mol, conf_ids, use_heavy_atoms_only)
    _, mask = prune_by_moment_of_inertia(coords, atoms)
    return [conf_ids[i] for i in range(len(conf_ids)) if mask[i]]
