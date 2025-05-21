"""
Neural Network model implementation for toxicity classification.
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils import load_features_target, save_report

class ToxicityNN(nn.Module):
    def __init__(self, input_dim=2048):
        super(ToxicityNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def train_NN(epochs=20, batch_size=64, lr=1e-3, verbose=False, plot_loss=False):
    """
    Train and evaluate the Neural Network model.
    """

    print("Training NN...")
    X, y = load_features_target()
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).unsqueeze(1)

    # Data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = ToxicityNN(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
        # for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if verbose:
            print(f"Epoch {epoch+1} loss: {epoch_loss/len(train_loader):.4f}")

    # Plot training loss
    if plot_loss:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), losses, marker="o")
        plt.title("Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    model.eval()
    with torch.no_grad():
        probs = model(X_test_tensor).numpy().flatten()
        y_pred = (probs > 0.05).astype(int)

        print("Classification report:")
        print(f"\n{classification_report(y_test, y_pred)}")
        save_report(model_name="Neural Network", y_true=y_test, y_pred=y_pred)

    parent = "models"
    filename = "nn_model.pt"
    path = os.path.join(parent, filename)
    os.makedirs(parent, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved in: {path}")
