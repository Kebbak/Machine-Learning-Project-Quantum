
import pytest
from flask import Flask
from flask.testing import FlaskClient
import sys
import os
sys.path.append('Webapps')
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    # Use demo data from the app
    features = client.application.view_functions['index'].__globals__['FEATURES']
    pd_mod = client.application.view_functions['index'].__globals__['pd']
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    demo_row = pd_mod.read_csv(os.path.join(BASE_DIR, 'splits', 'train_processed.csv')).iloc[0][features].to_dict()
    data = {f: demo_row[f] for f in features}
    response = client.post('/', data=data)
    assert response.status_code == 200
    assert b'Prediction' in response.data

def test_health_endpoint(client):
    # Add a /health endpoint to app.py for this test to pass
    response = client.get('/health')
    assert response.status_code == 200
    assert b'OK' in response.data
