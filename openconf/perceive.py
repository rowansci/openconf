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
        junction_atoms: Atoms shared with other rings (ring-fusion bonds).
            These define the reflection plane but do not move during the flip.
        stereo_sensitive: Whether simple plane reflection would touch specified
            tetrahedral stereochemistry and must use stereo-preserving moves.
    """

    ring_atoms: tuple[int, ...]
    ring_size: int
    junction_atoms: frozenset[int] = field(default_factory=frozenset)
    stereo_sensitive: bool = False


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


@dataclass(frozen=True)
class StereoSignature:
    """Specified stereochemistry labels for graph-level validation."""

    tetrahedral: dict[int, str]
    bonds: dict[int, Chem.BondStereo]


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

_SPECIFIED_BOND_STEREO: frozenset[Chem.BondStereo] = frozenset(
    {
        Chem.BondStereo.STEREOCIS,
        Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOZ,
    }
)


def _is_metal(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() in _METAL_ATOMIC_NUMS


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

    # Metal-ligand bonds have a very shallow torsional potential.
    if _is_metal(atom1) or _is_metal(atom2):
        return "metal_ligand"

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

    atom_to_rotors: dict[int, list[int]] = {}
    for i, rotor in enumerate(rotors):
        for atom_idx in rotor.atom_idxs:
            atom_to_rotors.setdefault(atom_idx, []).append(i)

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
    except (RuntimeError, ValueError) as e:
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

    Fused rings are supported: junction atoms (shared with another ring) define
    the reflection plane but do not move.  A ring is eligible when it has at
    least one non-junction atom so there is something to reflect. Rings whose
    reflection would touch specified tetrahedral stereochemistry are marked for
    stereo-preserving moves because plane reflection is an improper transform.

    Args:
        mol: RDKit molecule.
        atom_rings: List of atom index tuples, one per ring.

    Returns:
        List of RingFlip objects.
    """
    flips = []

    # Atoms that belong to more than one ring are ring-fusion junction atoms.
    atom_ring_count: dict[int, int] = {}
    for ring in atom_rings:
        for a in ring:
            atom_ring_count[a] = atom_ring_count.get(a, 0) + 1
    all_junction_atoms = frozenset(a for a, count in atom_ring_count.items() if count > 1)

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

        ring_junction = frozenset(ring) & all_junction_atoms
        # Need at least one free (non-junction) atom — a single atom popping
        # above/below the ring plane is the minimal envelope move.
        if size - len(ring_junction) < 1:
            continue

        stereo_sensitive = _ring_flip_touches_specified_tetrahedral_stereo(mol, tuple(ring), ring_junction)
        flips.append(
            RingFlip(
                ring_atoms=tuple(ring),
                ring_size=size,
                junction_atoms=ring_junction,
                stereo_sensitive=stereo_sensitive,
            )
        )
    return flips


def _specified_tetrahedral_centers(mol: Chem.Mol) -> frozenset[int]:
    """Return atom indices with explicitly assigned tetrahedral chirality."""
    specified_tags = {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    return frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetChiralTag() in specified_tags)


def specified_stereochemistry(mol: Chem.Mol) -> StereoSignature:
    """Return specified tetrahedral and double-bond stereochemistry.

    Args:
        mol: RDKit molecule.

    Returns:
        Stereochemistry labels assigned in molecule graph.
    """
    work = Chem.Mol(mol)
    Chem.AssignStereochemistry(work, cleanIt=True, force=True)
    specified_centers = _specified_tetrahedral_centers(work)
    center_labels = dict(Chem.FindMolChiralCenters(work, includeUnassigned=False, useLegacyImplementation=False))
    tetrahedral = {idx: center_labels[idx] for idx in specified_centers if idx in center_labels}
    bonds = {bond.GetIdx(): bond.GetStereo() for bond in work.GetBonds() if bond.GetStereo() in _SPECIFIED_BOND_STEREO}
    return StereoSignature(tetrahedral=tetrahedral, bonds=bonds)


def stereochemistry_from_conformer(mol: Chem.Mol, conf_id: int) -> StereoSignature:
    """Perceive tetrahedral and double-bond stereochemistry from conformer coordinates.

    Args:
        mol: RDKit molecule.
        conf_id: Conformer ID.

    Returns:
        Stereochemistry labels perceived from 3D coordinates.
    """
    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in work.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)

    Chem.AssignStereochemistryFrom3D(work, confId=int(conf_id), replaceExistingTags=True)
    Chem.AssignStereochemistry(work, cleanIt=True, force=True)

    center_labels = dict(Chem.FindMolChiralCenters(work, includeUnassigned=False, useLegacyImplementation=False))
    bonds = {bond.GetIdx(): bond.GetStereo() for bond in work.GetBonds() if bond.GetStereo() in _SPECIFIED_BOND_STEREO}
    return StereoSignature(tetrahedral=center_labels, bonds=bonds)


def conformer_matches_specified_stereochemistry(
    mol: Chem.Mol,
    conf_id: int,
    reference: StereoSignature,
) -> bool:
    """Return whether conformer geometry preserves specified stereochemistry.

    Args:
        mol: RDKit molecule.
        conf_id: Conformer ID.
        reference: Stereochemistry labels from input molecule.

    Returns:
        True when all specified labels match geometry perceived from conformer.
    """
    if not reference.tetrahedral and not reference.bonds:
        return True

    observed = stereochemistry_from_conformer(mol, conf_id)
    for atom_idx, label in reference.tetrahedral.items():
        if observed.tetrahedral.get(atom_idx) != label:
            return False
    for bond_idx, stereo in reference.bonds.items():
        if observed.bonds.get(bond_idx) != stereo:
            return False
    return True


def _ring_flip_touches_specified_tetrahedral_stereo(
    mol: Chem.Mol,
    ring_atoms: tuple[int, ...],
    junction_atoms: frozenset[int] = frozenset(),
) -> bool:
    """Return whether reflection would affect specified tetrahedral chirality.

    Ring flip is implemented as plane reflection of ring atoms plus attached
    subtrees. Reflection has determinant -1, so applying it to stereocenter
    coordinates can invert handedness while leaving RDKit chiral tags unchanged.

    Args:
        mol: RDKit molecule.
        ring_atoms: Ring atom indices.
        junction_atoms: Ring-fusion atoms excluded from reflection.

    Returns:
        True when reflected atom set includes specified stereocenter or one of
        its directly attached atoms.
    """
    moving_atoms = _ring_flip_moving_atoms(mol, ring_atoms, junction_atoms)
    if not moving_atoms:
        return False

    for center_idx in _specified_tetrahedral_centers(mol):
        if center_idx in moving_atoms:
            return True
        center = mol.GetAtomWithIdx(center_idx)
        if any(neighbor.GetIdx() in moving_atoms for neighbor in center.GetNeighbors()):
            return True
    return False


def _ring_flip_moving_atoms(
    mol: Chem.Mol,
    ring_atoms: tuple[int, ...],
    junction_atoms: frozenset[int] = frozenset(),
) -> frozenset[int]:
    """Return ring atoms and attached subtrees moved by plane reflection.

    Junction atoms (shared with another ring) define the reflection plane but
    do not move, so they and their extra-ring subtrees are excluded from the
    returned set.
    """
    ring_set = frozenset(ring_atoms)
    moving: set[int] = set()
    for ring_atom in ring_atoms:
        if ring_atom in junction_atoms:
            continue
        visited: set[int] = {ring_atom}
        stack = [ring_atom]
        while stack:
            cur = stack.pop()
            for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
                idx = nb.GetIdx()
                if idx in visited:
                    continue
                if idx in ring_set and idx != ring_atom:
                    continue
                visited.add(idx)
                stack.append(idx)
        moving.update(visited)
    return frozenset(moving)


def _atoms_on_side(mol: Chem.Mol, start_atom: int, excluded_atom: int) -> frozenset[int]:
    """BFS from start_atom without crossing the bond to excluded_atom.

    Returns the set of atom indices reachable from start_atom when the bond
    between excluded_atom and start_atom is severed. For acyclic bonds this
    gives the atoms in one fragment; ring bonds are already excluded from the
    rotor list so the result is always well-defined.

    Args:
        mol: RDKit molecule.
        start_atom: Index to start BFS from (one end of the bond).
        excluded_atom: The other end of the bond — never visited.

    Returns:
        frozenset of atom indices on start_atom's side of the bond.
    """
    visited: set[int] = {start_atom}
    queue = [start_atom]
    while queue:
        current = queue.pop()
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            nb_idx = neighbor.GetIdx()
            if nb_idx not in visited and nb_idx != excluded_atom:
                visited.add(nb_idx)
                queue.append(nb_idx)
    return frozenset(visited)


def filter_constrained_rotors(rotor_model: "RotorModel", constrained_atoms: frozenset[int]) -> "RotorModel":
    """Return a new RotorModel containing only rotors whose moving fragment is constraint-free.

    A rotor around bond (i, j) is kept when the *distal* atoms on one side —
    those beyond the bond-axis atom that actually translate during
    ``SetDihedralDeg`` — contain no constrained atoms.  The bond-axis atoms
    (atom_i and atom_j) lie on the rotation axis and never physically move, so
    they are excluded from the movability check.  This allows boundary-attachment
    bonds at the edge of a pinned scaffold to be kept even when both bond atoms
    are in the constrained set, provided the substituent beyond the axis atom is
    entirely free.

    When the free distal fragment is on the i-side rather than the j-side, the
    rotor is flipped so the moving fragment is always the free one.

    Ring flips are kept only when the entire ring is free of constrained atoms.

    Args:
        rotor_model: Full rotor model built by build_rotor_model.
        constrained_atoms: Atom indices that must not move.

    Returns:
        New RotorModel with only free rotors and free ring flips.
    """
    mol = rotor_model.mol
    free_rotors: list[Rotor] = []

    for rotor in rotor_model.rotors:
        atom_i, atom_j = rotor.atom_idxs
        moving_j = _atoms_on_side(mol, atom_j, atom_i)
        distal_j = moving_j - {atom_j}

        if not constrained_atoms & distal_j:
            # j-side distal atoms are free — use rotor as-is
            free_rotors.append(rotor)
        else:
            moving_i = _atoms_on_side(mol, atom_i, atom_j)
            distal_i = moving_i - {atom_i}
            if not constrained_atoms & distal_i:
                # i-side distal atoms are free — flip so the free side is the moving side
                a, i, j, b = rotor.dihedral_atoms
                free_rotors.append(
                    Rotor(
                        bond_idx=rotor.bond_idx,
                        atom_idxs=(atom_j, atom_i),
                        dihedral_atoms=(b, j, i, a),
                        rotor_type=rotor.rotor_type,
                    )
                )
            # else: constrained atoms in distal fragments on both sides — exclude this rotor

    # Ring flips: keep only flips whose full reflected atom set is constraint-free.
    free_ring_flips = [
        rf
        for rf in rotor_model.ring_flips
        if not constrained_atoms & _ring_flip_moving_atoms(mol, rf.ring_atoms, rf.junction_atoms)
    ]

    adj = _build_rotor_adjacency(free_rotors, mol)

    return RotorModel(
        mol=mol,
        rotors=free_rotors,
        adj=adj,
        ring_info=rotor_model.ring_info,
        ring_flips=free_ring_flips,
        heavy_atom_indices=rotor_model.heavy_atom_indices,
        n_rotatable=len(free_rotors),
    )


def build_rotor_model(mol: Chem.Mol) -> RotorModel:
    """Build a rotor model for a molecule.

    Identifies all rotatable bonds and builds a graph representation
    of their connectivity for correlated torsion moves.

    Args:
        mol: RDKit molecule (should have Hs added).

    Returns:
        RotorModel containing rotor information.
    """
    rot_bonds = mol.GetSubstructMatches(Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"))
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

        # Skip metal-metal bonds (M-M bonds in clusters/carbonyls are not
        # meaningful torsion degrees of freedom). Metal-ligand bonds are kept
        # and classified as "metal_ligand" rotors with a flat angle distribution.
        if _is_metal(mol.GetAtomWithIdx(atom_i)) and _is_metal(mol.GetAtomWithIdx(atom_j)):
            continue

        dihedral = _get_dihedral_atoms(mol, bond_idx)
        if dihedral is None:
            continue

        rotor = Rotor(
            bond_idx=bond_idx,
            atom_idxs=(atom_i, atom_j),
            dihedral_atoms=dihedral,
            rotor_type=_classify_rotor(mol, bond),
        )
        rotors.append(rotor)

    adj = _build_rotor_adjacency(rotors, mol)

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

    ring_flips = _find_ring_flips(mol, atom_rings)
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
