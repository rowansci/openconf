"""Molecule perception and rotor modeling for conformer generation."""

from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem


@dataclass
class Rotor:
    """Represents a rotatable bond.

    Attributes:
        bond_idx: Index of the rotatable bond.
        atom_idxs: Tuple of (begin_atom, end_atom) indices.
        dihedral_atoms: Tuple of 4 atom indices defining the dihedral.
        rotor_type: Classification of the rotor (e.g., "sp3_sp3", "amide", "biaryl").
        neighbors: Indices of adjacent rotors in the rotor graph.
    """

    bond_idx: int
    atom_idxs: tuple[int, int]
    dihedral_atoms: tuple[int, int, int, int]
    rotor_type: str = "generic"
    neighbors: list[int] = field(default_factory=list)


@dataclass
class RingFlip:
    """Non-aromatic ring that can undergo conformational flipping.

    Attributes:
        ring_atoms: Tuple of atom indices in ring traversal order.
        ring_size: Number of atoms in the ring.
    """

    ring_atoms: tuple[int, ...]
    ring_size: int


@dataclass
class RotorModel:
    """Model of all rotatable bonds in a molecule.

    Attributes:
        mol: RDKit molecule object.
        rotors: List of Rotor objects.
        adj: Adjacency list for the rotor graph.
        ring_info: Ring membership information.
        ring_flips: List of flippable non-aromatic rings (size 5-7).
        heavy_atom_indices: Indices of non-hydrogen atoms.
        n_rotatable: Number of rotatable bonds.
    """

    mol: Chem.Mol
    rotors: list[Rotor]
    adj: list[list[int]]
    ring_info: dict[str, Any]
    ring_flips: list[RingFlip]
    heavy_atom_indices: list[int]
    n_rotatable: int


# Atomic numbers considered metals for the purposes of rotor/ring-flip filtering.
# Bonds to metal centers and chelate rings are excluded from torsion sampling.
_METAL_ATOMIC_NUMS: frozenset[int] = frozenset(
    [
        # Alkali metals
        3,
        11,
        19,
        37,
        55,
        87,
        # Alkaline earth metals
        4,
        12,
        20,
        38,
        56,
        88,
        # Transition metals (d-block rows 4-6)
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,  # Sc-Zn
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,  # Y-Cd
        57,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,  # La, Hf-Hg
        # Lanthanides
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        # Main-group metals
        13,
        31,
        49,
        50,
        81,
        82,
        83,  # Al, Ga, In, Sn, Tl, Pb, Bi
    ]
)


def _is_metal(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() in _METAL_ATOMIC_NUMS


# SMARTS patterns for special rotor types
ROTOR_TYPE_SMARTS = {
    "amide": "[NX3][CX3](=[OX1])",  # Amide bond (restricted rotation)
    "ester": "[OX2][CX3](=[OX1])",  # Ester (restricted rotation)
    "biaryl": "c-c",  # Biaryl linkage
    "sulfonamide": "[NX3][SX4](=[OX1])(=[OX1])",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
}


def _get_dihedral_atoms(mol: Chem.Mol, bond_idx: int) -> tuple[int, int, int, int] | None:
    """Get four atoms defining the dihedral for a rotatable bond.

    Args:
        mol: RDKit molecule.
        bond_idx: Index of the bond.

    Returns:
        Tuple of 4 atom indices or None if not possible.
    """
    bond = mol.GetBondWithIdx(bond_idx)
    atom_i = bond.GetBeginAtomIdx()
    atom_j = bond.GetEndAtomIdx()

    # Get neighbors of atom_i (excluding atom_j)
    neighbors_i = [n.GetIdx() for n in mol.GetAtomWithIdx(atom_i).GetNeighbors() if n.GetIdx() != atom_j]
    if not neighbors_i:
        return None

    # Get neighbors of atom_j (excluding atom_i)
    neighbors_j = [n.GetIdx() for n in mol.GetAtomWithIdx(atom_j).GetNeighbors() if n.GetIdx() != atom_i]
    if not neighbors_j:
        return None

    # Prefer heavy atoms over hydrogens
    def sort_key(idx: int) -> tuple[int, int]:
        atom = mol.GetAtomWithIdx(idx)
        return (atom.GetAtomicNum() == 1, idx)  # H last

    neighbors_i.sort(key=sort_key)
    neighbors_j.sort(key=sort_key)

    return (neighbors_i[0], atom_i, atom_j, neighbors_j[0])


def _classify_rotor(mol: Chem.Mol, bond: Chem.Bond) -> str:
    """Classify a rotatable bond by chemical type.

    Args:
        mol: RDKit molecule.
        bond: The bond to classify.

    Returns:
        String classification of the rotor type.
    """
    atom1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
    atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())

    # Check for aromatic-aromatic (biaryl)
    if atom1.GetIsAromatic() and atom2.GetIsAromatic():
        return "biaryl"

    # Check for sp3-sp3
    hyb1 = atom1.GetHybridization()
    hyb2 = atom2.GetHybridization()

    if hyb1 == Chem.HybridizationType.SP3 and hyb2 == Chem.HybridizationType.SP3:
        # Check if it's C-C, C-N, C-O, etc.
        symbols = sorted([atom1.GetSymbol(), atom2.GetSymbol()])
        if symbols == ["C", "C"]:
            return "sp3_sp3_CC"
        elif "N" in symbols:
            return "sp3_sp3_CN"
        elif "O" in symbols:
            return "sp3_sp3_CO"
        return "sp3_sp3"

    # Check for sp2-sp3
    if (hyb1 == Chem.HybridizationType.SP2 and hyb2 == Chem.HybridizationType.SP3) or (
        hyb1 == Chem.HybridizationType.SP3 and hyb2 == Chem.HybridizationType.SP2
    ):
        return "sp2_sp3"

    return "generic"


def _build_rotor_adjacency(rotors: list[Rotor], mol: Chem.Mol) -> list[list[int]]:
    """Build adjacency list for the rotor graph.

    Two rotors are adjacent if they share an atom (i.e., correlated motion).

    Args:
        rotors: List of Rotor objects.
        mol: RDKit molecule.

    Returns:
        Adjacency list where adj[i] contains indices of rotors adjacent to rotor i.
    """
    n = len(rotors)
    adj: list[list[int]] = [[] for _ in range(n)]

    # Build atom to rotor mapping
    atom_to_rotors: dict[int, list[int]] = {}
    for i, rotor in enumerate(rotors):
        for atom_idx in rotor.atom_idxs:
            if atom_idx not in atom_to_rotors:
                atom_to_rotors[atom_idx] = []
            atom_to_rotors[atom_idx].append(i)

    # Rotors sharing an atom are adjacent
    for i, rotor in enumerate(rotors):
        neighbors = set()
        for atom_idx in rotor.atom_idxs:
            for j in atom_to_rotors.get(atom_idx, []):
                if j != i:
                    neighbors.add(j)
        adj[i] = sorted(neighbors)
        rotor.neighbors = adj[i]

    return adj


def prepare_molecule(mol: Chem.Mol, add_hs: bool = True) -> Chem.Mol:
    """Prepare a molecule for conformer generation.

    Args:
        mol: Input RDKit molecule.
        add_hs: Whether to add hydrogens.

    Returns:
        Prepared molecule with hydrogens and sanitization.

    Raises:
        ValueError: If molecule cannot be sanitized.
    """
    mol = Chem.Mol(mol)  # Make a copy

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise ValueError(f"Could not sanitize molecule: {e}") from e

    # Assign stereochemistry
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Add hydrogens if requested
    if add_hs:
        mol = Chem.AddHs(mol)

    return mol


def _find_ring_flips(mol: Chem.Mol, atom_rings: list[tuple[int, ...]]) -> list[RingFlip]:
    """Identify non-aromatic rings of size 5-7 that can undergo conformational flipping.

    Aromatic rings are rigid and excluded. Saturated and partially saturated
    rings of these sizes have accessible alternative conformations (chair/boat
    for 6-membered, envelope/twist for 5-membered).

    Args:
        mol: RDKit molecule.
        atom_rings: List of atom index tuples, one per ring.

    Returns:
        List of RingFlip objects.
    """
    flips = []
    for ring in atom_rings:
        size = len(ring)
        if size < 5 or size > 7:
            continue
        # Skip fully aromatic rings
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            continue
        # Skip chelate rings (rings containing a metal center): the metal's
        # coordination geometry constrains the ring and SVD reflection moves
        # would produce unphysical geometries.
        if any(_is_metal(mol.GetAtomWithIdx(idx)) for idx in ring):
            continue
        flips.append(RingFlip(ring_atoms=tuple(ring), ring_size=size))
    return flips


def build_rotor_model(mol: Chem.Mol) -> RotorModel:
    """Build a rotor model for a molecule.

    Identifies all rotatable bonds and builds a graph representation
    of their connectivity for correlated torsion moves.

    Args:
        mol: RDKit molecule (should have Hs added).

    Returns:
        RotorModel containing rotor information.
    """
    # Get rotatable bonds using RDKit
    rot_bonds = mol.GetSubstructMatches(Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"))

    # Alternative: use Lipinski.NumRotatableBonds for count only
    # But we need the actual bonds, so we use the SMARTS approach

    rotors: list[Rotor] = []
    seen_bonds = set()

    for match in rot_bonds:
        atom_i, atom_j = match
        bond = mol.GetBondBetweenAtoms(atom_i, atom_j)
        if bond is None:
            continue

        bond_idx = bond.GetIdx()
        if bond_idx in seen_bonds:
            continue
        seen_bonds.add(bond_idx)

        # Skip bonds in rings (already handled by !@)
        if bond.IsInRing():
            continue

        # Skip metal-ligand bonds: rotation around M-L bonds is not a
        # meaningful torsion degree of freedom for coordination chemistry.
        if _is_metal(mol.GetAtomWithIdx(atom_i)) or _is_metal(mol.GetAtomWithIdx(atom_j)):
            continue

        # Get dihedral atoms
        dihedral = _get_dihedral_atoms(mol, bond_idx)
        if dihedral is None:
            continue

        # Classify the rotor
        rotor_type = _classify_rotor(mol, bond)

        rotor = Rotor(
            bond_idx=bond_idx,
            atom_idxs=(atom_i, atom_j),
            dihedral_atoms=dihedral,
            rotor_type=rotor_type,
        )
        rotors.append(rotor)

    # Build adjacency
    adj = _build_rotor_adjacency(rotors, mol)

    # Ring info
    atom_rings = list(mol.GetRingInfo().AtomRings())
    ring_sizes = [len(r) for r in atom_rings]
    max_ring_size = max(ring_sizes, default=0)
    ring_info = {
        "ring_sizes": ring_sizes,
        "ring_atoms": atom_rings,
        "max_ring_size": max_ring_size,
        "has_macrocycle": max_ring_size >= 10,
        "has_small_ring": any(s <= 7 for s in ring_sizes),
    }

    # Ring flips: non-aromatic rings of size 5-7
    ring_flips = _find_ring_flips(mol, atom_rings)

    # Heavy atom indices
    heavy_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]

    return RotorModel(
        mol=mol,
        rotors=rotors,
        adj=adj,
        ring_info=ring_info,
        ring_flips=ring_flips,
        heavy_atom_indices=heavy_atom_indices,
        n_rotatable=len(rotors),
    )
