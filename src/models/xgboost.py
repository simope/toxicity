"""
XGBoost model implementation for toxicity classification.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

from utils import load_features_target, save_report


def train_XGB():
    """
    Train the XGBoost model.
    """
    print("Training XGBoost...")
    X, y = load_features_target()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification report:")
    print(f"\n{classification_report(y_test, y_pred)}")
    save_report(model_name="XGBoost", y_true=y_test, y_pred=y_pred)

    parent = "models"
    filename = "xgb_model.pkl"
    path = os.path.join(parent, filename)
    os.makedirs(parent, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved in: {path}")