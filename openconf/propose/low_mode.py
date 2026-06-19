"""Low mode following for conformer seeding.

Numerically evaluates Hessians via gradient finite differences and scans
minimized geometries along low-eigenvalue eigenvectors to generate structurally
diverse starting points. Most effective for macrocycles and correlated flexible
systems where independent torsion moves fail to sample collective soft motions.
"""

from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

if TYPE_CHECKING:
    from ..relax import Minimizer

_DEFAULT_FD_STEP: float = 0.005
_DEFAULT_EIGENVALUE_THRESHOLD: float = 100.0
_DEFAULT_MAX_MODES: int = 5
_DEFAULT_SCAN_STEP_SIZE: float = 0.25
_DEFAULT_SCAN_ENERGY_THRESHOLD: float = 2390.0  # kcal/mol ≈ 10 000 kJ/mol (paper value)
_DEFAULT_SCAN_MAX_STEPS: int = 10


def _compute_hessian(
    mol: Chem.Mol,
    ff_props: object,
    conf_id: int,
    step: float = _DEFAULT_FD_STEP,
) -> np.ndarray:
    """Compute numerical Hessian via central differences of MMFF gradients.

    Perturbs each Cartesian coordinate of each atom by ±step, evaluates
    the MMFF gradient at each displaced geometry, and assembles the
    3N×3N Hessian matrix. Conformer positions are restored on exit.

    Args:
        mol: molecule containing the conformer
        ff_props: pre-prepared MMFFMoleculeProperties used to build force fields
            at displaced geometries
        conf_id: ID of the conformer at which to evaluate the Hessian;
            should be at or near a local minimum for meaningful low modes
        step: finite difference displacement in Å

    Returns:
        Symmetric 3N×3N Hessian matrix in kcal/mol/Å²
    """
    conf = mol.GetConformer(conf_id)
    n_atoms = mol.GetNumAtoms()
    n_dof = 3 * n_atoms
    pos0 = conf.GetPositions().copy()
    hessian = np.zeros((n_dof, n_dof))

    for i in range(n_dof):
        atom_i = i // 3
        coord_i = i % 3
        original = pos0[atom_i].copy()

        fwd = original.copy()
        fwd[coord_i] += step
        conf.SetAtomPosition(atom_i, fwd.tolist())
        ff_fwd = AllChem.MMFFGetMoleculeForceField(mol, ff_props, confId=conf_id)
        if ff_fwd is None:
            conf.SetAtomPosition(atom_i, original.tolist())
            continue
        grad_plus = np.array(ff_fwd.CalcGrad())

        bwd = original.copy()
        bwd[coord_i] -= step
        conf.SetAtomPosition(atom_i, bwd.tolist())
        ff_bwd = AllChem.MMFFGetMoleculeForceField(mol, ff_props, confId=conf_id)
        if ff_bwd is None:
            conf.SetAtomPosition(atom_i, original.tolist())
            continue
        grad_minus = np.array(ff_bwd.CalcGrad())

        conf.SetAtomPosition(atom_i, original.tolist())
        hessian[i] = (grad_plus - grad_minus) / (2.0 * step)

    return (hessian + hessian.T) * 0.5


def _null_space(a: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the null space of a, via SVD."""
    _, s, vt = np.linalg.svd(a, full_matrices=True)
    rcond = max(a.shape) * np.finfo(float).eps * (s[0] if len(s) > 0 else 1.0)
    rank = int(np.sum(s > rcond))
    return vt[rank:].T


def _build_vibrational_basis(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    """Orthonormal basis for the mass-weighted vibrational subspace, shape (3N, 3N−6).

    Builds the six rigid-body vectors (3 translations + 3 rotations) in
    mass-weighted Cartesian coordinates q = M^{1/2} x, then returns their null
    space — the vibrational subspace. Mass-weighting follows the GF-matrix
    treatment (Wilson, Decius & Cross) so the projected Hessian yields
    eigenvalues proportional to ω² rather than to force constants alone.

    Near-zero rotation vectors (linear molecules have one) are dropped before
    the SVD so the basis always spans exactly 3N−6 or 3N−5 dimensions.

    Args:
        mol: molecule supplying atom positions and masses
        conf_id: conformer from which positions are taken

    Returns:
        Array of shape (3N, 3N−6) or (3N, 3N−5) for linear molecules
    """
    n_atoms = mol.GetNumAtoms()
    n_dof = 3 * n_atoms
    pt = Chem.GetPeriodicTable()
    masses = np.array([pt.GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetAtoms()])
    pos = mol.GetConformer(conf_id).GetPositions()  # (N, 3)

    com = np.sum(masses[:, None] * pos, axis=0) / np.sum(masses)
    r = pos - com  # COM-centred positions (N, 3)

    sqrt_masses = np.sqrt(masses)  # (N,)

    # Mass-weighted translation: t_k[3i+k] = sqrt(m_i), zero otherwise.
    # Represents uniform COM displacement in mass-weighted coordinates.
    trans = np.zeros((3, n_dof))
    for axis in range(3):
        trans[axis, axis::3] = sqrt_masses
    trans /= np.linalg.norm(trans, axis=1, keepdims=True)

    # Mass-weighted rotation: R_k[3i:3i+3] = sqrt(m_i) * (r_i x e_k).
    rot = np.zeros((3, n_dof))
    for k in range(n_atoms):
        sqrt_mk = sqrt_masses[k]
        rx, ry, rz = r[k]
        rot[0, 3 * k : 3 * k + 3] = sqrt_mk * np.array([0.0, rz, -ry])   # r x e_x
        rot[1, 3 * k : 3 * k + 3] = sqrt_mk * np.array([-rz, 0.0, rx])   # r x e_y
        rot[2, 3 * k : 3 * k + 3] = sqrt_mk * np.array([ry, -rx, 0.0])   # r x e_z

    # Threshold scales with sqrt(total_mass) since mass-weighted norms grow
    # as sqrt(M_total * r^2). Drops near-zero vectors for linear molecules.
    rot_threshold = 1e-8 * np.sqrt(float(np.sum(masses)))
    tr_vecs = list(trans)
    for d in rot:
        norm = float(np.linalg.norm(d))
        if norm > rot_threshold:
            tr_vecs.append(d / norm)

    return _null_space(np.stack(tr_vecs))


def _select_low_modes(
    hessian: np.ndarray,
    mol: Chem.Mol,
    conf_id: int,
    eigenvalue_threshold: float,
    max_modes: int,
) -> np.ndarray:
    """Select low-frequency eigenvectors by projecting out rigid-body modes.

    Works in mass-weighted Cartesian coordinates following the GF-matrix method.
    The mass-weighted Hessian F_mw = M^{-1/2} H M^{-1/2} is projected into
    the vibrational subspace and diagonalised; eigenvectors (proportional to ω²)
    are unweighted back to Cartesian displacement vectors and renormalised.
    This correctly accounts for atomic mass so that heavy-atom torsions (large m,
    small k) are ranked as soft modes rather than being penalised by their mass.

    Args:
        hessian: symmetric 3N×3N Hessian matrix in kcal/mol/Å²
        mol: molecule providing atom positions and masses
        conf_id: conformer ID used to build the translation/rotation basis
        eigenvalue_threshold: upper eigenvalue bound (kcal/mol/Å²·Da⁻¹); modes
            below this value are treated as conformationally soft
        max_modes: maximum number of modes to return

    Returns:
        Array of shape (3N, k) where k ≤ max_modes; columns are unit-norm
        Cartesian displacement vectors in ascending eigenvalue order;
        shape (3N, 0) when no conformational modes satisfy the threshold
    """
    pt = Chem.GetPeriodicTable()
    masses = np.array([pt.GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetAtoms()])
    # sqrt_masses[3i] = sqrt_masses[3i+1] = sqrt_masses[3i+2] = sqrt(m_i)
    sqrt_masses = np.repeat(np.sqrt(masses), 3)  # (3N,)

    # Mass-weighted Hessian: F_mw[i,j] = H[i,j] / (sqrt(m_i) * sqrt(m_j))
    H_mw = hessian / np.outer(sqrt_masses, sqrt_masses)

    # Vibrational basis is in mass-weighted coordinate space
    d_vib = _build_vibrational_basis(mol, conf_id)
    h_vib = d_vib.T @ H_mw @ d_vib
    eigenvalues, eigenvectors = np.linalg.eigh(h_vib)

    # d_vib @ eigenvectors: mass-weighted eigenvectors L_mw
    # Unweight to Cartesian: L_cart[i] = L_mw[i] / sqrt(m_i)
    vecs_mw = d_vib @ eigenvectors          # (3N, n_vib) in mass-weighted space
    vecs = vecs_mw / sqrt_masses[:, None]   # (3N, n_vib) in Cartesian space

    # Renormalise: mass-unweighting changes column norms
    col_norms = np.linalg.norm(vecs, axis=0, keepdims=True)
    col_norms[col_norms == 0.0] = 1.0
    vecs /= col_norms

    mask = eigenvalues < eigenvalue_threshold
    if not np.any(mask):
        return np.empty((hessian.shape[0], 0))
    return vecs[:, mask][:, :max_modes]


def _scan_along_mode(
    mol: Chem.Mol,
    ff_props: object,
    start_conf_id: int,
    direction: np.ndarray,
    step_size: float,
    energy_threshold: float,
    max_steps: int,
) -> np.ndarray:
    """Scan from a minimized conformer along a unit direction vector.

    Takes discrete steps of step_size Å in direction, evaluating the MMFF
    energy after each step and stopping as soon as the per-step energy increase
    exceeds energy_threshold. Returns the positions from the last accepted step,
    or the starting positions when the first step already exceeds the threshold.

    A temporary conformer is created and removed; the start conformer is
    never modified.

    Args:
        mol: molecule to scan; receives a temporary conformer during the call
        ff_props: pre-prepared MMFFMoleculeProperties for energy evaluation
        start_conf_id: ID of the minimized starting conformer
        direction: unit displacement direction of shape (n_atoms, 3) in Å;
            each step moves the geometry by step_size × direction in 3N space
        step_size: distance (Å) to move in 3N Cartesian space per step
        energy_threshold: maximum allowed per-step energy increase (kcal/mol);
            scanning stops when ΔE in a single step exceeds this value
        max_steps: maximum number of steps regardless of energy criterion

    Returns:
        Positions of shape (n_atoms, 3) at the last accepted scan point
    """
    n_atoms = mol.GetNumAtoms()
    src_conf = mol.GetConformer(start_conf_id)
    start_pos = src_conf.GetPositions().copy()

    ff0 = AllChem.MMFFGetMoleculeForceField(mol, ff_props, confId=start_conf_id)
    if ff0 is None:
        return start_pos
    prev_energy = float(ff0.CalcEnergy())

    working_conf = Chem.Conformer(src_conf)
    working_id = mol.AddConformer(working_conf, assignId=True)

    current_pos = start_pos.copy()
    accepted_pos = start_pos.copy()

    try:
        for _ in range(max_steps):
            current_pos = current_pos + step_size * direction
            working = mol.GetConformer(working_id)
            for i in range(n_atoms):
                working.SetAtomPosition(i, current_pos[i].tolist())

            ff = AllChem.MMFFGetMoleculeForceField(mol, ff_props, confId=working_id)
            if ff is None:
                break

            curr_energy = float(ff.CalcEnergy())
            if curr_energy - prev_energy > energy_threshold:
                break

            accepted_pos = current_pos.copy()
            prev_energy = curr_energy
    finally:
        mol.RemoveConformer(working_id)

    return accepted_pos


def generate_low_mode_seeds(
    mol: Chem.Mol,
    ff_props: object,
    conf_id: int,
    minimizer: "Minimizer",
    *,
    eigenvalue_threshold: float = _DEFAULT_EIGENVALUE_THRESHOLD,
    max_modes: int = _DEFAULT_MAX_MODES,
    scan_step_size: float = _DEFAULT_SCAN_STEP_SIZE,
    scan_energy_threshold: float = _DEFAULT_SCAN_ENERGY_THRESHOLD,
    scan_max_steps: int = _DEFAULT_SCAN_MAX_STEPS,
    fd_step: float = _DEFAULT_FD_STEP,
) -> list[tuple[int, float]]:
    """Generate conformers by scanning along low-frequency Hessian eigenvectors.

    Numerically evaluates the MMFF Hessian at the given minimized conformer,
    identifies eigenvectors with eigenvalues below eigenvalue_threshold (soft
    conformational modes), and for each such mode scans in both the positive and
    negative directions using discrete steps of scan_step_size Å. Scanning along
    a direction stops when the per-step energy increase exceeds scan_energy_threshold
    or scan_max_steps is reached. Each scan endpoint is then minimized to a local
    minimum.

    This mirrors the LMOD procedure of Kolossváry & Guida (JACS 1996): the scan
    naturally traverses soft conformational barriers and terminates at the onset
    of severe steric clashes, placing the starting geometry for minimization on
    the far side of a barrier.

    New conformers are added to mol in place. Callers are responsible for
    removing conformers that are not kept (e.g., pool rejects).

    Note:
        Hessian evaluation requires 6N MMFF force-field constructions where N
        is the atom count. This is the dominant cost; each scan step adds two
        further force-field constructions (one per direction).

    Args:
        mol: molecule containing the conformer; receives new conformers in place
        ff_props: pre-prepared MMFFMoleculeProperties used to build force fields
        conf_id: ID of a minimized conformer to compute modes from
        minimizer: minimizer applied to each scan endpoint
        eigenvalue_threshold: Hessian eigenvalue cutoff in mass-weighted
            coordinates (kcal/mol/Å²/Da, proportional to omega^2); modes
            below this value are treated as conformationally soft
        max_modes: maximum number of low modes to scan per conformer
        scan_step_size: distance to advance per scan step in Å (3N Euclidean
            norm of the displacement vector); smaller values give finer
            resolution of the stopping point
        scan_energy_threshold: maximum per-step energy increase (kcal/mol)
            before scanning stops; the default (~2390 kcal/mol ≈ 10 000 kJ/mol)
            follows the paper and effectively allows the scan to pass through
            conformational barriers, stopping only at severe steric clashes
        scan_max_steps: upper bound on scan steps per direction regardless of
            the energy criterion; acts as a safety cap on total displacement
        fd_step: finite difference step size for numerical Hessian in Å

    Returns:
        Pairs of (conformer_id, energy_kcal_mol) for each successfully
        minimized scan endpoint; at most 2 results per mode (the two scan
        senses); empty when no low modes satisfy the threshold or all
        minimizations fail
    """
    hessian = _compute_hessian(mol, ff_props, conf_id, fd_step)
    low_vecs = _select_low_modes(hessian, mol, conf_id, eigenvalue_threshold, max_modes)
    if low_vecs.shape[1] == 0:
        return []

    n_atoms = mol.GetNumAtoms()
    start_pos = mol.GetConformer(conf_id).GetPositions().copy()

    results: list[tuple[int, float]] = []
    for col in range(low_vecs.shape[1]):
        direction = low_vecs[:, col].reshape(n_atoms, 3)

        for sign in (1.0, -1.0):
            final_pos = _scan_along_mode(
                mol,
                ff_props,
                conf_id,
                sign * direction,
                scan_step_size,
                scan_energy_threshold,
                scan_max_steps,
            )

            if np.allclose(final_pos, start_pos, atol=1e-8):
                continue

            new_conf = Chem.Conformer(mol.GetConformer(conf_id))
            new_conf_id = mol.AddConformer(new_conf, assignId=True)
            displaced = mol.GetConformer(new_conf_id)
            for i in range(n_atoms):
                displaced.SetAtomPosition(i, final_pos[i].tolist())

            energy = minimizer.minimize(mol, new_conf_id)
            if not np.isfinite(energy):
                mol.RemoveConformer(new_conf_id)
                continue

            results.append((new_conf_id, energy))

    return results
