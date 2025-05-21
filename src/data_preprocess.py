import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

import config


def load_tox21(path: str, target: str = "all") -> pd.DataFrame:
    # Load dataset from csv
    df = pd.read_csv(path)
    # Select target
    if target.lower() != "all" and (target in config.TARGETS_LIST):
        # Keep only smiles and target
        df = df[['smiles', target]]
    # Drop rows with missing label for the selected endpoint
    df = df.dropna()

    return df.reset_index(drop=True)

def validate_smiles(df: pd.DataFrame) -> pd.DataFrame:
    # Obtain molecule object
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    # Drop the non-molecules
    df = df[df['mol'].notnull()]
    return df.reset_index(drop=True)

def get_fingerprint(mol: Chem.Mol, radius=2, n_bits=2048) -> np.ndarray:
    generator = GetMorganGenerator(radius=radius,fpSize=n_bits)
    fp = generator.GetFingerprintAsNumPy(mol)
    return fp

def generate_features(df: pd.DataFrame) -> np.ndarray:
    mols = df["mol"]
    fps = [get_fingerprint(mol) for mol in mols]
    features = np.array([np.array(fp) for fp in fps])
    return features

def generate_labels(df: pd.DataFrame, target: str) -> np.ndarray:
    if target == "all":
        return df[config.TARGETS_LIST].to_numpy()
    elif target not in config.TARGETS_LIST:
        raise ValueError(f"Selected target: {target} is not present in the dataset.")
    return df[target].to_numpy()
 
def save_preprocessed(features: np.ndarray, labels: np.ndarray):
    np.save("data/processed/features.npy", features)
    np.save("data/processed/labels.npy", labels)


if __name__ == "__main__":
    df = load_tox21(path=config.DATASET_PATH, target=config.TARGET)
    df = validate_smiles(df=df)
    features = generate_features(df=df)
    labels = generate_labels(df=df, target=config.TARGET)
    save_preprocessed(features=features, labels=labels)