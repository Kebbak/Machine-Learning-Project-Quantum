from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import os
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

# Health check endpoint for CI/CD smoke test
@app.route('/health')
def health():
    return 'OK', 200

# Load the best model (robust path for deployment)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model.joblib')
model = joblib.load(MODEL_PATH)

# Robust path for processed train set
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = pd.read_csv(os.path.join(BASE_DIR, 'splits', 'train_processed.csv')).drop('Label', axis=1).columns.tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    demo_row = pd.read_csv(os.path.join(BASE_DIR, 'splits', 'train_processed.csv')).iloc[0][FEATURES].to_dict()
    prediction = None
    if request.method == 'POST':
        input_data = [float(request.form.get(f, 0)) for f in FEATURES]
        df = pd.DataFrame([input_data], columns=FEATURES)
        pred = model.predict(df)[0]
        prediction = 'Malware' if pred == 1 else 'Goodware'
    return render_template('index.html', features=FEATURES, demo_row=demo_row, prediction=prediction)

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
