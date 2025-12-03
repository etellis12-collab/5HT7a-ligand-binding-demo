import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


@dataclass
class Ligand:
    name: str
    smiles: str
    pki: float  # negative log10 Ki (higher = tighter binding)


def build_5ht7a_dataset() -> List[Ligand]:
    """
    Construct a small toy dataset of 5-HT7A ligands.
    pKi values are illustrative, not experimental.
    """
    ligands = [
        Ligand("Ligand_A", "CCN(CC)CCOc1ccc2nc(SCC)sc2c1", 8.1),
        Ligand("Ligand_B", "COc1ccc2nc(SCCN(C)C)sc2c1", 7.8),
        Ligand("Ligand_C", "CCOC(=O)NCCOc1ccc2nc(Cl)sc2c1", 7.2),
        Ligand("Ligand_D", "CCN(CCO)CCOc1ccc2nc(Cl)sc2c1", 7.5),
        Ligand("Ligand_E", "COc1ccc2nc(S(=O)(=O)NCC)sc2c1", 7.0),
        Ligand("Ligand_F", "COc1ccc2nc(S(=O)(=O)NCCN)sc2c1", 7.4),
        Ligand("Ligand_G", "CCN(CC)CCOc1ccc2nc(S(=O)(=O)N)sc2c1", 8.3),
        Ligand("Ligand_H", "COc1ccc2nc(SCCN)sc2c1", 6.9),
    ]
    return ligands


def featurize_mol(smiles: str) -> np.ndarray:
    """
    Generate a combined feature vector:
    - Morgan fingerprint
    - Simple physicochemical descriptors
    This is meant to approximate information relevant to 5-HT7A binding,
    such as aromaticity, polarity, and size.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Morgan fingerprint (ECFP-like)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_arr = np.zeros((1,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)

    # Basic descriptors: MW, LogP, HBD, HBA, TPSA, Rotatable Bonds
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)

    physchem = np.array([mw, logp, hbd, hba, tpsa, rot_bonds])

    return np.concatenate([fp_arr, physchem])


def build_feature_matrix(ligands: List[Ligand]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for lig in ligands:
        X.append(featurize_mol(lig.smiles))
        y.append(lig.pki)
    return np.vstack(X), np.array(y)


def train_5ht7a_model(random_state: int = 42) -> None:
    """
    Train and evaluate a simple Random Forest model for 5-HT7A pKi prediction.
    """
    ligands = build_5ht7a_dataset()
    X, y = build_feature_matrix(ligands)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print("5-HT7A QSAR Demo")
    print("----------------")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"R^2:   {r2:.3f}")
    print(f"RMSE:  {rmse:.3f} pKi units\n")

    # Show predictions for test ligands
    print("Test set predictions:")
    for i, idx in enumerate(range(len(y_test))):
        print(f"  Ligand {i}: true pKi = {y_test[idx]:.2f}, pred = {y_pred[idx]:.2f}")


if __name__ == "__main__":
    train_5ht7a_model()
