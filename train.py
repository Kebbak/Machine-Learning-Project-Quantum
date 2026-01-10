import numpy as np
import pandas as pd
import random
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Webapps')))

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
import model_defs
SimpleMLP = model_defs.SimpleMLP
TorchMLPClassifier = model_defs.TorchMLPClassifier
import joblib

# Additional models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# PyTorch MLP

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
