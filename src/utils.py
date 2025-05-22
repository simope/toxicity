import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, roc_auc_score
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem


def save_report(model_name, y_true, y_pred, probs=None):
    filepath="models/classification_reports.txt"
    report = str(classification_report(y_true, y_pred, digits=4))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if model_name == "Neural Network" and (probs is not None):
        rocauc = str(roc_auc_score(y_true=y_true, y_score=probs))
        report += "\n"
        report += f"ROC-AUC score: {rocauc:.2f}"

    with open(filepath, "a") as f:
        f.write(f"\n=== Model: {model_name} | {timestamp} ===\n")
        f.write(report)
        f.write("\n")
    print(f"Report saved in {filepath}.")
        
def load_features_target() -> tuple:
    X = np.load("data/processed/features.npy")
    y = np.load("data/processed/labels.npy")
    return X, y

def visualize_molecule_3d(smiles: str, width: int = 400, height: int = 400) -> str:
    """
    Generate a 3D visualization of a molecule from its SMILES string.
    
    Args:
        smiles (str): SMILES string of the molecule
        width (int): Width of the visualization in pixels
        height (int): Height of the visualization in pixels
        
    Returns:
        str: HTML string containing the 3D visualization
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Convert to 3Dmol view
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(Chem.MolToMolBlock(mol), "mol")
    
    # Set visualization style
    viewer.setStyle({'stick': {'radius': 0.2, 'color': 'spectrum'}})
    viewer.setBackgroundColor('#0E1117')  # Streamlit's dark background color
    viewer.zoomTo()
    
    # Return the HTML
    return viewer._make_html()

def load_model_and_predict(smiles: str) -> float:
    """
    Load the trained model and make a prediction for a given SMILES string.
    
    Args:
        smiles (str): SMILES string of the molecule
        
    Returns:
        float: Toxicity probability (0-1)
    """
    # TODO: Add proper feature generation from SMILES
    # For now, return a dummy prediction
    return 0.75  # Placeholder