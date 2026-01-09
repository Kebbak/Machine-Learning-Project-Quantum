import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def test_imputer():
    df = pd.DataFrame({'A': [1, np.nan, 3]})
    imputer = SimpleImputer(strategy='median')
    result = imputer.fit_transform(df)
    assert np.allclose(result, [[1],[2],[3]])

def test_scaler():
    df = pd.DataFrame({'A': [1, 2, 3]})
    scaler = StandardScaler()
    result = scaler.fit_transform(df)
    assert np.isclose(np.mean(result), 0)
    assert np.isclose(np.std(result), 1)
