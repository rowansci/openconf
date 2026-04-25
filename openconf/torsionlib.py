"""Torsion library for biased conformer generation.

Provides SMARTS-based torsion angle preferences derived from the RDKit CrystalFF
torsion library (J. Chem. Inf. Model. 56, 1, 2016). Fourier series coefficients
were converted to preferred angles by numerical minimization.
"""

import json
from dataclasses import dataclass, field
from functools import cache
from importlib.resources import files
from pathlib import Path

from rdkit import Chem


@dataclass
class TorsionRule:
    """Torsion angle rule defined by SMARTS pattern.

    Attributes:
        smarts: SMARTS pattern for matching the dihedral (atoms :1,:2,:3,:4).
        angles_deg: Preferred torsion angles in degrees.
        weights: Sampling weights for each angle (need not sum to 1).
        name: Human-readable name for the rule.
    """

    smarts: str
    angles_deg: list[float]
    weights: list[float] = field(default_factory=list)
    name: str = ""

    def __post_init__(self) -> None:
        """Validate and initialize default weights."""
        if not self.weights:
            self.weights = [1.0] * len(self.angles_deg)
        if len(self.weights) != len(self.angles_deg):
            msg = f"weights length ({len(self.weights)}) must match angles_deg length ({len(self.angles_deg)})"
            raise ValueError(msg)


def _load_default_rules() -> list[TorsionRule]:
    """Load the bundled torsion library JSON.

    Returns:
        List of TorsionRule objects from the default CrystalFF library.
    """
    data_path = files("openconf.data").joinpath("torsion_library.json")
    with data_path.open() as f:
        data = json.load(f)
    return [
        TorsionRule(
            smarts=r["smarts"],
            angles_deg=r["angles_deg"],
            weights=r.get("weights") or [],
            name=r.get("name", ""),
        )
        for r in data["rules"]
    ]


class TorsionLibrary:
    """Library of torsion angle preferences for biased sampling.

    Matches rotatable bonds against SMARTS patterns to determine preferred
    torsion angles. Rules are matched in order; the first match wins, so
    more specific patterns should come first.

    The default library ships 365 crystallography-derived rules
    (RDKit CrystalFF v2, J. Chem. Inf. Model. 56, 1, 2016).
    """

    def __init__(self, rules: list[TorsionRule] | None = None):
        """Initialize the torsion library.

        Args:
            rules: Rules to use. If None, loads the bundled CrystalFF library.
        """
        self.rules = rules if rules is not None else _load_default_rules()
        self._compiled: list[tuple[Chem.Mol, TorsionRule, int, int]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile SMARTS patterns, skipping any that RDKit cannot parse.

        Also pre-computes the query-atom positions for map numbers :2 and :3
        (the central bond atoms), since branch atoms in parentheses shift the
        default indices and make positional indexing unreliable.
        """
        for rule in self.rules:
            pattern = Chem.MolFromSmarts(rule.smarts)
            if pattern is None:
                continue
            # Find the query-atom positions for map numbers 2 and 3.
            pos2 = pos3 = None
            for i in range(pattern.GetNumAtoms()):
                m = pattern.GetAtomWithIdx(i).GetAtomMapNum()
                if m == 2:
                    pos2 = i
                elif m == 3:
                    pos3 = i
            if pos2 is None or pos3 is None:
                continue  # malformed pattern
            self._compiled.append((pattern, rule, pos2, pos3))

    def _match_dihedral(
        self,
        mol: Chem.Mol,
        dihedral_atoms: tuple[int, int, int, int],
    ) -> TorsionRule | None:
        """Return the first matching rule for this dihedral, or None.

        The CrystalFF patterns define 4-atom contexts that may not agree with
        the specific terminal atoms we chose in ``_get_dihedral_atoms`` (e.g.,
        amide patterns always include the carbonyl O as atom :1, but our
        dihedral might use a different neighbor).  We therefore match on the
        central bond only: atoms :2 and :3 in the SMARTS (``match[1]`` and
        ``match[2]``) must equal the rotatable bond atoms, in either direction.
        """
        central = (dihedral_atoms[1], dihedral_atoms[2])
        central_rev = (dihedral_atoms[2], dihedral_atoms[1])

        for pattern, rule, pos2, pos3 in self._compiled:
            for match in mol.GetSubstructMatches(pattern):
                mc = (match[pos2], match[pos3])
                if mc in (central, central_rev):
                    return rule
        return None

    def get_preferred_angles(
        self,
        mol: Chem.Mol,
        dihedral_atoms: tuple[int, int, int, int],
    ) -> tuple[list[float], list[float]]:
        """Get preferred torsion angles for a dihedral.

        Args:
            mol: RDKit molecule.
            dihedral_atoms: Four atom indices (a, b, c, d) defining the dihedral.

        Returns:
            (angles_deg, weights). Falls back to generic staggered angles
            (60°/180°/300°) when no rule matches.
        """
        rule = self._match_dihedral(mol, dihedral_atoms)
        if rule is not None:
            return rule.angles_deg.copy(), rule.weights.copy()
        # Generic fallback: staggered sp3
        return [60.0, 180.0, 300.0], [1.0, 1.0, 1.0]

    def match_rotor(
        self,
        mol: Chem.Mol,
        dihedral_atoms: tuple[int, int, int, int],
    ) -> TorsionRule | None:
        """Return the matching TorsionRule for a rotor, or None.

        Args:
            mol: RDKit molecule.
            dihedral_atoms: Four atom indices defining the dihedral.

        Returns:
            Matching TorsionRule, or None if no rule matches.
        """
        return self._match_dihedral(mol, dihedral_atoms)

    @classmethod
    def from_json(cls, path: str | Path) -> "TorsionLibrary":
        """Load a torsion library from a JSON file.

        Args:
            path: Path to a JSON file with a ``rules`` list, each entry having
                ``smarts``, ``angles_deg``, and optionally ``weights`` / ``name``.

        Returns:
            TorsionLibrary loaded from the file.
        """
        with open(path) as f:
            data = json.load(f)
        rules = [
            TorsionRule(
                smarts=r["smarts"],
                angles_deg=r["angles_deg"],
                weights=r.get("weights") or [],
                name=r.get("name", ""),
            )
            for r in data["rules"]
        ]
        return cls(rules=rules)

    def to_json(self, path: str | Path) -> None:
        """Save this library to a JSON file.

        Args:
            path: Output path.
        """
        data = {
            "rules": [
                {
                    "smarts": r.smarts,
                    "angles_deg": r.angles_deg,
                    "weights": r.weights,
                    "name": r.name,
                }
                for r in self.rules
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def __len__(self) -> int:
        """Return number of rules in the library."""
        return len(self.rules)


@cache
def get_default_torsion_library() -> TorsionLibrary:
    """Return the cached default torsion library.

    The bundled CrystalFF-derived library is immutable in normal use, so a
    shared instance avoids repeated JSON parsing and SMARTS compilation across
    conformer-generation calls.
    """
    return TorsionLibrary()
