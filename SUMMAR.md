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

**EDA:**
  - Class balance in training set:
    - Malware (Label=1): 40,144 samples
    - Goodware (Label=0): 16,892 samples
    - Noted class imbalance (malware > goodware)
  - Missing data:
    - Columns with missing values: FormatedTimeDateStamp, Identify, MD5, Name, FirstSeenDate
    - Example missing counts:
      - FormatedTimeDateStamp: 40,144
      - Identify: 18,847
      - MD5: 40,144
      - Name: 40,144
      - FirstSeenDate: 16,892
  - Feature distributions and correlation heatmap plotted (see eda_train_feature_distributions.png, eda_train_corr_heatmap.png)
  - Issues documented in eda_issues.txt

**Preprocessing Plan:**
  - Will handle missing values in columns with high missingness (drop or impute as appropriate)
  - Preprocessing (scaling, encoding, imputation) will be fit only on training data, then applied to validation/test sets
  - All transformations will be documented and reproducible
  - Processed files will be saved as splits/train_processed.csv and splits/test_processed.csv

# Step 4: Train/Validation/Test Protocol
  - The dataset was split into 80% training and 20% hold-out test set, stratified by class proportions to maintain balance.
  - The test set was held out and not used for any preprocessing, feature engineering, or model selection to prevent data leakage.
  - Within the training set, stratified 10-fold cross-validation (CV) was performed for model selection and hyperparameter tuning.
  - All model selection and tuning was based only on the training data and CV results.
  - The final evaluation was performed on the untouched test set after all model development was complete.

# Step 5: Feature Engineering & Preprocessing
  - Dropped columns with excessive missing values or not useful for ML: FormatedTimeDateStamp, MD5, Name, FirstSeenDate.
  - All non-numeric columns were removed to ensure compatibility with scikit-learn models.
  - Only numeric features (plus Label) retained for modeling.
  - Missing values in numeric columns imputed with the median (fit on train, applied to test).
  - Numeric features scaled using StandardScaler (fit on train, applied to test).
  - All preprocessing steps fit only on training data, then applied to validation/test sets.
  - Processed files: splits/train_processed.csv, splits/test_processed.csv

  - No additional feature selection or dimensionality reduction was applied at this stage.
  - All transformations are reproducible and documented in preprocess.py.


# Step 6: Model Training & Evaluation

**Models Evaluated:**
  - Baseline: Logistic Regression, Decision Tree, Random Forest, PyTorch MLP
  - Additional: XGBoost, LightGBM, CatBoost

**Cross-Validation Results:**
  - For each model, 10-fold stratified cross-validation was performed.
  - Mean ± std dev for AUC and accuracy were recorded in cv_results_all_models.csv.
  - The best model by mean CV AUC was selected for final evaluation.

**Final Test Set Evaluation:**
  - Test AUC: 0.7295
  - Test Accuracy: 0.6586
  - Confusion Matrix:
    [[ 425 3798]
     [1070 8967]]
  - Classification Report:
    - Class 0 (goodware): precision 0.28, recall 0.10, f1-score 0.15, support 4223
    - Class 1 (malware): precision 0.70, recall 0.89, f1-score 0.79, support 10037
    - Macro avg: precision 0.49, recall 0.50, f1-score 0.47
    - Weighted avg: precision 0.58, recall 0.66, f1-score 0.60
  - The model is much better at detecting malware than goodware, likely due to class imbalance.
  - Full results saved to test_results.txt.

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

