#!/usr/bin/env python
"""Download PDBbind refined set ligands for benchmarking.

The PDBbind database contains protein-ligand complexes with experimentally
determined binding affinities. We use the ligand structures from the refined
set as a benchmark for conformer generation.

Note: PDBbind requires registration at http://pdbbind.org.cn/ to download.
This script processes already-downloaded PDBbind data or provides instructions.

Usage:
    # Process existing PDBbind data
    python scripts/download_pdbbind.py --pdbbind-dir /path/to/pdbbind/refined-set

    # Show download instructions
    python scripts/download_pdbbind.py --help-download
"""

import argparse
import shutil
import sys
from pathlib import Path

from rdkit import Chem


def extract_ligands_from_pdbbind(pdbbind_dir: Path, output_dir: Path, max_ligands: int | None = None) -> int:
    """Extract ligand SDF files from PDBbind refined set.

    The PDBbind refined set contains protein-ligand complexes. For each complex,
    we extract the ligand coordinates from the *_ligand.mol2 or *_ligand.sdf file.

    Args:
        pdbbind_dir: Path to PDBbind refined-set directory.
        output_dir: Directory to write extracted ligand SDFs.
        max_ligands: Maximum number of ligands to extract.

    Returns:
    -------
        Number of successfully extracted ligands.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # PDBbind structure: refined-set/XXXX/XXXX_ligand.mol2 or .sdf
    complex_dirs = sorted([d for d in pdbbind_dir.iterdir() if d.is_dir()])
    if max_ligands:
        complex_dirs = complex_dirs[:max_ligands]

    extracted = 0
    failed = []

    for complex_dir in complex_dirs:
        pdb_id = complex_dir.name

        # Try different ligand file formats
        ligand_file = None
        for suffix in ["_ligand.sdf", "_ligand.mol2"]:
            candidate = complex_dir / f"{pdb_id}{suffix}"
            if candidate.exists():
                ligand_file = candidate
                break

        if ligand_file is None:
            failed.append(pdb_id)
            continue

        # Load ligand
        if ligand_file.suffix == ".sdf":
            suppl = Chem.SDMolSupplier(str(ligand_file), removeHs=False)
            mol = next(suppl, None)
        elif ligand_file.suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(ligand_file), removeHs=False)
        else:
            failed.append(pdb_id)
            continue

        if mol is None:
            failed.append(pdb_id)
            continue

        # Write to output directory
        output_file = output_dir / f"{pdb_id}_ligand.sdf"
        writer = Chem.SDWriter(str(output_file))
        writer.write(mol)
        writer.close()

        extracted += 1

    if failed:
        print(f"Warning: Failed to extract {len(failed)} ligands")
        if len(failed) <= 10:
            print(f"  Failed: {', '.join(failed)}")

    return extracted


def filter_by_size(input_dir: Path, output_dir: Path, min_heavy: int = 10, max_heavy: int = 50) -> int:
    """Filter ligands by heavy atom count.

    Args:
        input_dir: Directory with ligand SDFs.
        output_dir: Directory for filtered output.
        min_heavy: Minimum heavy atoms.
        max_heavy: Maximum heavy atoms.

    Returns:
    -------
        Number of ligands passing filter.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    passed = 0
    for sdf_file in input_dir.glob("*.sdf"):
        suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
        mol = next(suppl, None)
        if mol is None:
            continue

        n_heavy = len([a for a in mol.GetAtoms() if a.GetAtomicNum() > 1])
        if min_heavy <= n_heavy <= max_heavy:
            shutil.copy(sdf_file, output_dir / sdf_file.name)
            passed += 1

    return passed


def print_download_instructions():
    """Print instructions for downloading PDBbind."""
    instructions = """
PDBbind Download Instructions
=============================

PDBbind is a curated database of protein-ligand complexes with experimentally
determined binding data. The refined set (~5000 complexes) provides high-quality
ligand structures for benchmarking.

Steps to download:

1. Register at http://pdbbind.org.cn/register.php
   (Free for academic use)

2. After approval (usually within 24 hours), log in and navigate to:
   Downloads -> PDBbind refined set

3. Download the "refined-set" archive (PDBbind_v20XX_refined.tar.gz)
   Note: The file is several GB in size

4. Extract the archive:
   tar -xzf PDBbind_v20XX_refined.tar.gz

5. Run this script to extract ligands:
   python scripts/download_pdbbind.py --pdbbind-dir /path/to/refined-set

Alternative: Use PDB directly
-----------------------------

If PDBbind registration is not available, you can also benchmark using
structures directly from the PDB. This script can process any directory
containing *_ligand.sdf or *_ligand.mol2 files.

For quick testing, the Iridium dataset (already included in static/iridium/)
provides 120 high-quality ligand structures.
"""
    print(instructions)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract PDBbind ligands for benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pdbbind-dir",
        type=Path,
        help="Path to PDBbind refined-set directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "static" / "pdbbind",
        help="Output directory for extracted ligands",
    )
    parser.add_argument(
        "--max-ligands",
        type=int,
        default=None,
        help="Maximum number of ligands to extract",
    )
    parser.add_argument(
        "--min-heavy",
        type=int,
        default=10,
        help="Minimum heavy atom count (default: 10)",
    )
    parser.add_argument(
        "--max-heavy",
        type=int,
        default=50,
        help="Maximum heavy atom count (default: 50)",
    )
    parser.add_argument(
        "--help-download",
        action="store_true",
        help="Show download instructions",
    )
    args = parser.parse_args()

    if args.help_download:
        print_download_instructions()
        return

    if args.pdbbind_dir is None:
        print("Error: --pdbbind-dir is required (or use --help-download for instructions)")
        print("\nQuick start:")
        print("  python scripts/download_pdbbind.py --help-download")
        sys.exit(1)

    if not args.pdbbind_dir.exists():
        print(f"Error: PDBbind directory not found: {args.pdbbind_dir}")
        sys.exit(1)

    print("PDBbind Ligand Extraction")
    print("=" * 50)
    print(f"Input: {args.pdbbind_dir}")
    print(f"Output: {args.output_dir}")
    print()

    # Extract ligands
    print("Extracting ligands...")
    raw_dir = args.output_dir / "raw"
    n_extracted = extract_ligands_from_pdbbind(
        args.pdbbind_dir,
        raw_dir,
        max_ligands=args.max_ligands,
    )
    print(f"  Extracted: {n_extracted} ligands")

    # Filter by size
    print(f"\nFiltering by size ({args.min_heavy}-{args.max_heavy} heavy atoms)...")
    n_filtered = filter_by_size(
        raw_dir,
        args.output_dir,
        min_heavy=args.min_heavy,
        max_heavy=args.max_heavy,
    )
    print(f"  Passed: {n_filtered} ligands")

    # Clean up raw directory
    shutil.rmtree(raw_dir)

    print(f"\nDone! Ligands saved to: {args.output_dir}")
    print("Run benchmark with: pixi run -e bench python scripts/comprehensive_benchmark.py --dataset pdbbind")


if __name__ == "__main__":
    main()
