"""Supplementary seeds from in-ring tertiary amide cis/trans enumeration.

Tertiary amides (N-substituted, no N-H) in macrocyclic rings have a significant
rotational barrier on the N-C(=O) bond, giving distinct cis and trans
conformational families. For cyclic peptoids and similar molecules this creates
2^k families (k = number of such bonds) that ETKDG does not systematically cover.

This module identifies those bonds and uses MMFF torsion constraints to derive
one seed conformer for each flip combination, up to 2^_MAX_FLIP_BONDS variants.
"""

from __future__ import annotations

import itertools
import math

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

_MAX_FLIP_BONDS = 4  # enumerate at most 2^4 = 16 variant seeds
_CONSTRAIN_ITERS = 100  # MMFF steps with torsion constraints active
_RELAX_ITERS = 30  # MMFF steps after removing constraints
_TORSION_WINDOW_DEG = 30.0
_TORSION_FORCE_K = 500.0  # kcal/mol/rad²
_MIN_MACROCYCLE_SIZE = 10


def find_ring_tertiary_amide_dihedrals(mol: Chem.Mol) -> list[tuple[int, int, int, int]]:
    """Return O=C-N-C dihedral indices for in-ring tertiary amides in macrocycles.

    Only reports bonds where:
      - N has no hydrogens (tertiary / N-substituted)
      - N is non-aromatic
      - Both N and the adjacent C(=O) are members of a ring of size >= 10
    """
    ring_info = mol.GetRingInfo()
    macro_rings = [set(r) for r in ring_info.AtomRings() if len(r) >= _MIN_MACROCYCLE_SIZE]
    if not macro_rings:
        return []

    results: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()

    for ring_set in macro_rings:
        for n_idx in ring_set:
            n_atom = mol.GetAtomWithIdx(n_idx)
            if n_atom.GetAtomicNum() != 7:
                continue
            if n_atom.GetTotalNumHs() > 0:
                continue  # secondary amide (N-H) — less flexible, skip
            if n_atom.GetIsAromatic():
                continue

            for bond in n_atom.GetBonds():
                if not bond.IsInRing():
                    continue
                c_atom = bond.GetOtherAtom(n_atom)
                c_idx = c_atom.GetIdx()
                if c_atom.GetAtomicNum() != 6 or c_idx not in ring_set:
                    continue
                if c_atom.GetIsAromatic():
                    continue

                bond_key = (min(n_idx, c_idx), max(n_idx, c_idx))
                if bond_key in seen:
                    continue

                # Require an exocyclic =O on the carbonyl carbon
                o_idx: int | None = None
                for cb in c_atom.GetBonds():
                    if cb.GetBondTypeAsDouble() < 1.9:
                        continue
                    o_atom = cb.GetOtherAtom(c_atom)
                    if o_atom.GetAtomicNum() == 8 and not o_atom.GetIsAromatic():
                        o_idx = o_atom.GetIdx()
                        break
                if o_idx is None:
                    continue

                # Fourth atom for the dihedral: any N-neighbor besides the carbonyl C
                c_sub: int | None = None
                for nb_bond in n_atom.GetBonds():
                    nb = nb_bond.GetOtherAtom(n_atom)
                    if nb.GetIdx() != c_idx:
                        c_sub = nb.GetIdx()
                        break
                if c_sub is None:
                    continue

                seen.add(bond_key)
                results.append((o_idx, c_idx, n_idx, c_sub))

    return results


def generate_amide_variant_seeds(
    mol: Chem.Mol,
    mmff_props,
    base_conf_id: int,
    amide_dihedrals: list[tuple[int, int, int, int]],
) -> list[tuple[int, float]]:
    """Generate seed conformers from cis/trans variants of in-ring amide bonds.

    For each non-trivial subset of flips (up to 2^_MAX_FLIP_BONDS - 1 = 15
    combinations), copies the base conformer, applies MMFF torsion constraints
    to drive selected bonds to the opposite configuration, minimizes with those
    constraints active, then does a free relaxation pass. Returns conformers
    with finite MMFF energy as (conf_id, energy_kcal) pairs.

    Args:
        mol: Working molecule; new conformers are added in-place.
        mmff_props: MMFFMolProperties from the fast minimizer (molecule-specific).
        base_conf_id: ID of the lowest-energy ETKDG seed to use as starting geometry.
        amide_dihedrals: Output of find_ring_tertiary_amide_dihedrals.

    Returns:
        List of (conf_id, energy_kcal) for successfully relaxed variants.
    """
    k = min(len(amide_dihedrals), _MAX_FLIP_BONDS)
    dihedrals = amide_dihedrals[:k]

    base_conf = mol.GetConformer(int(base_conf_id))
    current_angles = [rdMolTransforms.GetDihedralDeg(base_conf, *quad) for quad in dihedrals]

    results: list[tuple[int, float]] = []

    for flip_bits in itertools.product((False, True), repeat=k):
        if not any(flip_bits):
            continue  # all-False = original configuration, already in ETKDG seeds

        new_cid = mol.AddConformer(Chem.Conformer(base_conf), assignId=True)

        try:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=int(new_cid))
            if ff is None:
                mol.RemoveConformer(new_cid)
                continue

            for (o, c, n, cs), cur, flip in zip(dihedrals, current_angles, flip_bits, strict=True):
                if not flip:
                    continue
                target = cur + 180.0
                if target > 180.0:
                    target -= 360.0
                ff.MMFFAddTorsionConstraint(
                    o,
                    c,
                    n,
                    cs,
                    False,
                    target - _TORSION_WINDOW_DEG,
                    target + _TORSION_WINDOW_DEG,
                    _TORSION_FORCE_K,
                )

            ff.Minimize(maxIts=_CONSTRAIN_ITERS)

            # Free pass — remove constraint strain, let ring close properly
            ff2 = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=int(new_cid))
            if ff2 is not None:
                ff2.Minimize(maxIts=_RELAX_ITERS)
                energy = float(ff2.CalcEnergy())
            else:
                energy = float(ff.CalcEnergy())

        except (ValueError, RuntimeError):
            mol.RemoveConformer(new_cid)
            continue

        if math.isfinite(energy):
            results.append((new_cid, energy))
        else:
            mol.RemoveConformer(new_cid)

    return results
