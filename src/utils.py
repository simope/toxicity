import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report


def save_report(model_name, y_true, y_pred):
    filepath="models/classification_reports.txt"
    report = str(classification_report(y_true, y_pred, digits=4))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, "a") as f:
        f.write(f"\n=== Model: {model_name} | {timestamp} ===\n")
        f.write(report)
        f.write("\n")
    print(f"Report saved in {filepath}.")
        
def load_features_target() -> tuple:
    X = np.load("data/processed/features.npy")
    y = np.load("data/processed/labels.npy")
    return X, y