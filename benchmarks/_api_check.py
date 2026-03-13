"""Verify RDKit confId APIs before using them."""

import inspect

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, rdFreeSASA

mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
mol = Chem.AddHs(mol)
AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42)

# Check confId param on Descriptors3D
for fn_name in ["RadiusOfGyration", "PBF", "NPR1", "NPR2"]:
    fn = getattr(Descriptors3D, fn_name)
    sig = inspect.signature(fn)
    accepts_confid = "confId" in sig.parameters
    print(f"Descriptors3D.{fn_name} accepts confId: {accepts_confid}  (params: {list(sig.parameters)})")

# Try calling with confId
conf_id = mol.GetConformer(1).GetId()
try:
    rog = Descriptors3D.RadiusOfGyration(mol, confId=conf_id)
    print(f"\nRadiusOfGyration(mol, confId={conf_id}) = {rog:.3f}  OK")
except Exception as e:
    print(f"\nRadiusOfGyration with confId FAILED: {e}")

# Check rdFreeSASA.CalcSASA signature
sig2 = inspect.signature(rdFreeSASA.CalcSASA)
print(f"\nrdFreeSASA.CalcSASA params: {list(sig2.parameters)}")

# Try CalcSASA with confIdx
radii = rdFreeSASA.classifyAtoms(mol)
try:
    sasa = rdFreeSASA.CalcSASA(mol, radii, confIdx=1)  # confIdx is 0-based index
    print(f"CalcSASA(mol, radii, confIdx=1) = {sasa:.3f}  OK")
    # Check atom SASA props
    n_with_sasa = sum(1 for a in mol.GetAtoms() if a.HasProp("SASA"))
    print(f"  Atoms with SASA prop: {n_with_sasa}/{mol.GetNumAtoms()}")
except Exception as e:
    print(f"CalcSASA with confIdx FAILED: {e}")
