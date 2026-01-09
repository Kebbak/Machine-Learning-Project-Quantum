import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

# Paths
TRAIN_RAW = os.path.join('splits', 'train_raw.csv')
TEST_RAW = os.path.join('splits', 'test_raw.csv')
TRAIN_OUT = os.path.join('splits', 'train_processed.csv')
TEST_OUT = os.path.join('splits', 'test_processed.csv')

# Load data
train = pd.read_csv(TRAIN_RAW)
test = pd.read_csv(TEST_RAW)

# Drop columns with too many missing values or not useful for ML
cols_to_drop = ['FormatedTimeDateStamp', 'MD5', 'Name', 'FirstSeenDate']
train = train.drop(columns=cols_to_drop, errors='ignore')
test = test.drop(columns=cols_to_drop, errors='ignore')

# Drop all non-numeric columns except Label
keep_cols = [col for col in train.columns if train[col].dtype in [np.float64, np.int64] or col == 'Label']
train = train[keep_cols]
test = test[keep_cols]

# Select numeric features for scaling
numeric_cols = [col for col in train.columns if col != 'Label']

# Impute missing numeric values with median (fit on train, apply to both)
imputer = SimpleImputer(strategy='median')
train[numeric_cols] = imputer.fit_transform(train[numeric_cols])
test[numeric_cols] = imputer.transform(test[numeric_cols])

# Scale numeric features (fit on train, apply to both)
scaler = StandardScaler()
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])

# Save processed files
train.to_csv(TRAIN_OUT, index=False)
test.to_csv(TEST_OUT, index=False)
print('Preprocessing complete. Only numeric features retained. Processed files saved.')
