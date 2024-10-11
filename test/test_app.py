import base64
import bz2
import io
import pickle
import numpy as np
from flask import Flask
import pytest
import pandas as pd
import json



# Data generation functions
def generate_dataset(N, x1_min, x1_max, x2_sigma, x3_sigma):
    x1 = np.random.uniform(x1_min, x1_max, size=int(N))
    x2 = np.random.normal(loc=x1, scale=x2_sigma)
    x3 = np.random.uniform(x1_min, x1_max, size=int(N))
    return np.stack((x1, x2, x3), axis=-1)

def predict(x):
    y = 7*x[:, 0] - 3*x[:, 1] + 4*x[:, 2]
    return y

@pytest.fixture
def client():
    app = Flask(__name__)
    app.config['TESTING'] = True
    return app.test_client()

def test_feature_effect(client):
    # Generate the dataset
    np.random.seed(21)
    N = 1000
    x1_min = 0
    x1_max = 1
    x2_sigma = .1
    x3_sigma = 1.
    X = generate_dataset(N, x1_min, x1_max, x2_sigma, x3_sigma)

    # Convert dataset to CSV
    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # Serialize and compress the predict function
    compressed_model = bz2.compress(pickle.dumps(predict))

    # Encode both model and CSV into base64
    json_payload = {
        'file': base64.b64encode(csv_data.encode()).decode(),
        'model': base64.b64encode(compressed_model).decode(),
        'method': 'PDP',
        'feature_index': '0'
    }

    # Send the POST request
    response = client.post('/feature_effect',
                           data=json.dumps(json_payload),
                           content_type='application/json')

    # Assert the response is an image
    assert response.status_code == 200
    assert response.content_type == 'image/png'
