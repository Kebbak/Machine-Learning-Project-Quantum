import numpy as np
import pandas as pd
import random
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Webapps')))
import model_defs
SimpleMLP = model_defs.SimpleMLP
TorchMLPClassifier = model_defs.TorchMLPClassifier
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

# Set seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# Load processed test set and best model

test = pd.read_csv(os.path.join('splits', 'test_processed.csv'))
X_test = test.drop('Label', axis=1)
y_test = test['Label']

# Allow user to specify model filename as argument, default to best_model.joblib for backward compatibility
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = 'best_model.joblib'

# Define TorchMLPClassifier and SimpleMLP classes
from Webapps.model_defs import SimpleMLP, TorchMLPClassifier

model = joblib.load(model_path)
y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

auc = roc_auc_score(y_test, y_proba)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Test AUC: {auc:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# Save results
with open('test_results.txt', 'w') as f:
    f.write(f"Test AUC: {auc:.4f}\n")
    f.write(f"Test Accuracy: {acc:.4f}\n")
    f.write("Confusion Matrix:\n" + str(cm) + "\n")
    f.write("Classification Report:\n" + report + "\n")
print('Test results saved to test_results.txt')
