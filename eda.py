import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
train_path = os.path.join('splits', 'train_raw.csv')
test_path = os.path.join('splits', 'test_raw.csv')
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 1. Class balance
class_counts = train['Label'].value_counts()
print('Class balance in training set:')
print(class_counts)

# 2. Missing data
missing = train.isnull().sum()
print('\nMissing values per column:')
print(missing[missing > 0])

# 3. Feature distributions (numeric)
numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns.drop('Label')
train[numeric_cols].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.savefig('eda_train_feature_distributions.png')
plt.close()

# 4. Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(train[numeric_cols].corr(), cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.savefig('eda_train_corr_heatmap.png')
plt.close()

# 5. Save Exploratory data analysis summary
data_issues = missing[missing > 0]
with open('eda_issues.txt', 'w') as f:
    f.write('Class balance:\n')
    f.write(str(class_counts) + '\n\n')
    f.write('Missing values per column:\n')
    f.write(str(data_issues) + '\n')

print('EDA complete. Plots and summary saved.')
