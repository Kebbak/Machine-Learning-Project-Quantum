import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=10, lr=0.001, batch_size=128, seed=42):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.model = None
    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        self.model = SimpleMLP(self.input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.classes_ = np.array([0, 1])
        for epoch in range(self.epochs):
            permutation = torch.randperm(X_tensor.size()[0])
            for i in range(0, X_tensor.size()[0], self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch_x, batch_y = X_tensor[indices], y_tensor[indices]
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self
    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        with torch.no_grad():
            probs = self.model(X_tensor).numpy().flatten()
        return np.vstack([1-probs, probs]).T
    def predict(self, X):
        proba = self.predict_proba(X)[:,1]
        return (proba > 0.5).astype(int)
