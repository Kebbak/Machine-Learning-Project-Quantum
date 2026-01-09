import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
GOODWARE_PATH = 'goodware.csv'
MALWARE_PATH = 'malware.csv'
SPLITS_DIR = 'splits'
TRAIN_OUT = os.path.join(SPLITS_DIR, 'train_raw.csv')
TEST_OUT = os.path.join(SPLITS_DIR, 'test_raw.csv')

# Ensure splits directory exists
os.makedirs(SPLITS_DIR, exist_ok=True)

# Load data
print('Loading goodware...')
goodware = pd.read_csv(GOODWARE_PATH)
goodware['Label'] = 0

print('Loading malware...')
malware = pd.read_csv(MALWARE_PATH)
malware['Label'] = 1

# Combine
data = pd.concat([goodware, malware], ignore_index=True)

# Stratified split
print('Splitting data (80% train, 20% test, stratified)...')
train, test = train_test_split(
    data,
    test_size=0.2,
    stratify=data['Label'],
    random_state=42
)

# Save
print(f'Saving train to {TRAIN_OUT}')
train.to_csv(TRAIN_OUT, index=False)
print(f'Saving test to {TEST_OUT}')
test.to_csv(TEST_OUT, index=False)
print('Done.')
