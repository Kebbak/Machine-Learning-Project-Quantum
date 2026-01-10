import sys
from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import os
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import model_defs
SimpleMLP = model_defs.SimpleMLP
TorchMLPClassifier = model_defs.TorchMLPClassifier
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


app = Flask(__name__)

# Health check endpoint for CI/CD smoke test
@app.route('/health')
def health():
    return 'OK', 200

# Load the best model (robust path for deployment)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model_PyTorchMLP.joblib')
model = joblib.load(MODEL_PATH)

# Robust path for processed train set
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = pd.read_csv(os.path.join(BASE_DIR, 'splits', 'train_processed.csv')).drop('Label', axis=1).columns.tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    demo_row = pd.read_csv(os.path.join(BASE_DIR, 'splits', 'train_processed.csv')).iloc[0][FEATURES].to_dict()
    prediction = None
    confidence = None
    if request.method == 'POST':
        input_data = [float(request.form.get(f, 0)) for f in FEATURES]
        df = pd.DataFrame([input_data], columns=FEATURES)
        pred = model.predict(df)[0]
        prediction = 'Malware' if pred == 1 else 'Goodware'
        # Try to get probability/confidence if supported
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            confidence = max(proba)
        elif hasattr(model, 'decision_function'):
            # For models with decision_function (e.g., SVM)
            decision = model.decision_function(df)[0]
            confidence = 1 / (1 + np.exp(-decision))
    # Feature importance (if supported)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = sorted(zip(FEATURES, importances), key=lambda x: -abs(x[1]))[:10]
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0] if hasattr(model.coef_, '__len__') and len(model.coef_.shape) > 1 else model.coef_
        feature_importance = sorted(zip(FEATURES, importances), key=lambda x: -abs(x[1]))[:10]
    return render_template('index.html', features=FEATURES, demo_row=demo_row, prediction=prediction, confidence=confidence, feature_importance=feature_importance)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    file = request.files['file']
    df = pd.read_csv(file)
    preds = model.predict(df[FEATURES])
    df['Prediction'] = ['Malware' if p == 1 else 'Goodware' for p in preds]
    metrics = None
    if 'Label' in df.columns:
        auc = roc_auc_score(df['Label'], preds)
        acc = accuracy_score(df['Label'], preds)
        cm = confusion_matrix(df['Label'], preds)
        metrics = {
            'AUC': auc,
            'Accuracy': acc,
            'ConfusionMatrix': cm.tolist(),
            'ClassificationReport': classification_report(df['Label'], preds, output_dict=True)
        }
    return jsonify({'results': df.to_dict(orient='records'), 'metrics': metrics})

if __name__ == '__main__':
    app.run(debug=True)
