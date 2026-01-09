import numpy as np
import pandas as pd
import random
import torch
import os

# Set seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# Load processed data
train = pd.read_csv(os.path.join('splits', 'train_processed.csv'))
test = pd.read_csv(os.path.join('splits', 'test_processed.csv'))

X_train = train.drop('Label', axis=1)
y_train = train['Label']
X_test = test.drop('Label', axis=1)
y_test = test['Label']

# Baseline and additional models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

# Additional models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# PyTorch MLP
import torch.nn as nn
import torch.optim as optim
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
        self.classes_ = np.array([0, 1])  # For sklearn compatibility
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

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'PyTorchMLP': TorchMLPClassifier(input_dim=X_train.shape[1], epochs=10),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'CatBoost': cb.CatBoostClassifier(verbose=0, random_state=42)
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = {}


best_auc = -np.inf
best_model_name = None
best_model_instance = None

for name, model in models.items():
    print(f"Evaluating {name}...")
    aucs = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    accs = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    results[name] = {
        'cv_auc_mean': np.mean(aucs),
        'cv_auc_std': np.std(aucs),
        'cv_acc_mean': np.mean(accs),
        'cv_acc_std': np.std(accs)
    }
    print(f"{name}: AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}, Acc {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    # Track best model by mean CV AUC
    if np.mean(aucs) > best_auc:
        best_auc = np.mean(aucs)
        best_model_name = name
        # Fit on all training data for saving
        best_model_instance = model.fit(X_train, y_train)

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv('cv_results_all_models.csv')

# Save best model by mean CV AUC
if best_model_instance is not None:
    joblib.dump(best_model_instance, f'best_model_{best_model_name}.joblib')
    print(f'Best model by CV AUC: {best_model_name} (AUC={best_auc:.4f}) saved as best_model_{best_model_name}.joblib')
