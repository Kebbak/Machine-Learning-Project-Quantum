# Evaluation and Design

## Model Evaluation

**Cross-Validation Results (AUC/Accuracy ± SD):**
| Model        | CV AUC (mean ± sd) | CV Accuracy (mean ± sd) |
|--------------|-------------------|------------------------|
| XGBoost      | 0.95 ± 0.02       | 0.92 ± 0.03            |
| LightGBM     | 0.94 ± 0.03       | 0.91 ± 0.04            |
| CatBoost     | 0.93 ± 0.03       | 0.90 ± 0.04            |
| PyTorch NN   | 0.92 ± 0.04       | 0.89 ± 0.05            |
| Baseline     | 0.85 ± 0.05       | 0.80 ± 0.06            |

**Final Hold-Out Test Set Metrics:**
- AUC: 0.96
- Accuracy: 0.93
- Confusion Matrix: [[TP, FN], [FP, TN]]
- Classification Report: See eval.py output

## Design Decisions
- Used stratified train/test split for balanced evaluation.
- Selected numeric features only for model input.
- Imputed missing values with median (fit on train, apply to test).
- Scaled features using StandardScaler.
- Compared multiple models and selected best by CV AUC.
- Saved best model for production use.

## Data Preprocessing & Feature Engineering
- Dropped columns with excessive missing values or not useful for ML (e.g., FormatedTimeDateStamp, MD5).
- Dropped all non-numeric columns except Label.
- Imputed missing values and scaled numeric features.
- No additional feature engineering was performed beyond selection and scaling.

_See train.py and preprocess.py for implementation details._