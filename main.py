import base64
import bz2
import io
import pickle

import effector
import pandas as pd
from flask import Flask, request, jsonify, send_file
from matplotlib import pyplot as plt

app = Flask(__name__)

VALID_METHODS = ['PDP', 'dPDP', 'ALE', 'RHALE', 'SHAP_DP']


def validateEffectorMethod(method):
    if method not in VALID_METHODS:
        return False
    return True


def validateFile(file):
    if file is None:
        return False
    return True


@app.route('/feature_effect', methods=['POST'])
def feature_effect():

    data = request.get_json()

    # Decode and read CSV file
    csv_base64 = data.get('file')
    if csv_base64 is None:
        return jsonify({'error': 'CSV file is required'}), 400

    csv_data = base64.b64decode(csv_base64).decode()
    csv_buffer = io.StringIO(csv_data)
    X_df = pd.read_csv(csv_buffer)

    X_data = X_df.to_numpy()
    feature_names = X_df.columns.to_list()

    # Decode and decompress the model
    model_base64 = data.get('model')

    if model_base64 is None:
        return jsonify({'error': 'Model is required'}), 400

    compressed_model = base64.b64decode(model_base64)
    model = pickle.loads(bz2.decompress(compressed_model))

    # Ensure model is callable before proceeding
    if not callable(model):
        return jsonify({'error': 'The model must be callable'}), 400

    # Validate the requested method
    method = request.form.get('method')
    if not validateEffectorMethod(method):
        return jsonify({'error': f'Invalid method. Must be one of {VALID_METHODS}'}), 400

    # Validate the feature index
    feature_index = request.form.get('feature_index')
    if feature_index is None or not feature_index.isdigit():
        return jsonify({'error': 'Feature index must be a valid integer'}), 400
    feature_index = int(feature_index)

    # Call the appropriate Effector method
    if method == 'PDP':
        result = effector.PDP(data=X_data, model=model, feature_names=feature_names)
    elif method == 'dPDP':
        result = effector.RegionalPDP(data=X_data, model=model, feature_names=feature_names)
    elif method == 'ALE':
        result = effector.ALE(data=X_data, model=model, feature_names=feature_names)
    elif method == 'RHALE':
        result = effector.RegionalALE(data=X_data, model=model, feature_names=feature_names)
    elif method == 'SHAP_DP':
        result = effector.ShapDP(data=X_data, model=model, feature_names=feature_names)
    else:
        return jsonify({'error': 'Invalid method'}), 400

    result.plot(feature=feature_index, centering=True, show_avg_output=True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Return the plot as a response
    return send_file(img, mimetype='image/png')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
