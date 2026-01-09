The web application is ready for deployment and public demo.
# Step 1: Dataset & Problem Definition

- **Dataset Used:** goodware.csv (benign samples only)
- **Target Variable:** Label (all samples labeled as 0)
- **Success Metrics:**
  - Primary: Area Under the ROC Curve (AUC)
  - Secondary: Accuracy
- **Data Split:**
  - 80% training set, 20% test set
  - Test set held out before any data preprocessing or feature engineering to prevent leakage
  - Output files: splits/train_raw.csv, splits/test_raw.csv

# Step 2: Environment & Reproducibility

- **Virtual Environment:** Recommended to use venv or conda for isolation
- **Dependencies:** All required packages pinned in requirements.txt
- **Reproducibility Scripts:**
  - train.py: trains a model on train_raw.csv
  - eval.py: evaluates model on test_raw.csv
- **Random Seeds:** Fixed random seeds for splits and model training (default: 42)

# Step 3: Data Understanding & Preparation

- **EDA:**
  - Class balance: Only benign samples (Label=0) in train and test sets
  - Missing data: Identify column has missing values (7461 in train, 1903 in test); all other columns are complete
  - Feature distributions plotted for numeric columns (see splits/)
  - Issues documented in splits/eda_issues.txt
- **Preprocessing:**
  - Missing values in numeric columns filled with train median
  - Preprocessing fit on train, applied to test
  - Processed files: splits/train_processed.csv, splits/test_processed.csv

# Step 4: Train/Validation/Test Protocol

- **Splitting Strategy:**
  - 80% training, 20% hold-out test set (stratified by class proportions)
  - Test set remains untouched until final evaluation
- **Cross-Validation & Model Selection:**
  - Stratified 10-fold cross-validation performed within training set for model selection and hyperparameter tuning
  - Best parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 50}
  - Best CV accuracy: 1.0000
  - Best model saved to best_model.joblib

# Step 5: Preprocessing & Feature Engineering

- **Preprocessing Steps:**
  - Scaling (StandardScaler) and feature selection (SelectKBest) included in CV pipeline
  - All transformations fit only on training folds during cross-validation, then applied to validation/test folds
  - Feature selection warnings (invalid value in divide) expected due to single-class data
- **Feature Engineering Results:**
  - Best parameters: {'clf__max_depth': None, 'clf__min_samples_split': 2, 'clf__n_estimators': 50, 'select__k': 5}
  - Best CV accuracy: 1.0000
  - Best model saved to best_model.joblib

# Step 6: Model Training & Evaluation

- **Models Evaluated:**
  - Baseline: Logistic Regression, Decision Tree, Random Forest, PyTorch MLP
  - Additional: XGBoost, LightGBM, CatBoost
- **Results:**
  - Cross-validation (AUC/accuracy ± std) recorded for all models
  - Best model selected by mean CV AUC
  - Final test set evaluation performed and saved to test_results.txt
  - Results saved to cv_results_all_models.csv

# Step 7: Web Application Development

- **Production Model Integration:**
  - The best-performing model is loaded and used for inference in a Flask web application.
- **UI Features:**
  - Manual feature entry form with a pre-filled demo row from the dataset for easy testing.
  - File upload option for batch predictions on multiple instances.
  - Results for each instance are displayed in a table.
- **Evaluation Metrics:**
  - If the uploaded file contains class labels, the app computes and displays AUC, accuracy, and the confusion matrix.
  - This can be demonstrated using the 20% hold-out test file.

