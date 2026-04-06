"""openconf - Modular conformer generation for docking and ensemble workflows.

A unified, modular conformer engine optimized for docking recall and
diverse ensemble workflows, using RDKit for heavy lifting
and PRISM Pruner for fast, robust deduplication.

Examples:
--------
    >>> from openconf import generate_conformers, ConformerConfig
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CCCCc1ccccc1")
    >>> ensemble = generate_conformers(mol)  # doctest: +SKIP
"""

__version__ = "0.1.0"

from .api import (
    ConformerEnsemble,
    ConformerRecord,
    generate_conformers,
    generate_conformers_from_pose,
    generate_conformers_from_smiles,
)
from .config import ConformerConfig, ConformerPreset, ConstraintSpec, PrismConfig, preset_config
from .dedupe import prism_dedupe
from .io import mol_to_smiles, read_sdf, smiles_to_mol, write_sdf, write_xyz
from .perceive import Rotor, RotorModel, build_rotor_model, filter_constrained_rotors, prepare_molecule
from .relax import RDKitMMFFMinimizer, get_minimizer
from .torsionlib import TorsionLibrary, TorsionRule

__all__ = [
    "ConformerConfig",
    "ConformerEnsemble",
    "ConformerPreset",
    "ConformerRecord",
    "ConstraintSpec",
    "PrismConfig",
    "RDKitMMFFMinimizer",
    "Rotor",
    "RotorModel",
    "TorsionLibrary",
    "TorsionRule",
    "build_rotor_model",
    "filter_constrained_rotors",
    "generate_conformers",
    "generate_conformers_from_pose",
    "generate_conformers_from_smiles",
    "get_minimizer",
    "mol_to_smiles",
    "prepare_molecule",
    "preset_config",
    "prism_dedupe",
    "read_sdf",
    "smiles_to_mol",
    "write_sdf",
    "write_xyz",
]
