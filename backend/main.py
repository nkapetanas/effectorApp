import json
import traceback

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import numpy as np
import tensorflow as tf
from effector import PDP, RHALE, RegionalRHALE, RegionalPDP, binning_methods, utils
import requests
import tempfile
import os
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd

from backend.DataModelFetcher import DataModelFetcher

app = Flask(__name__)
CORS(app, origins=['http://localhost:4200'], allow_headers=['Content-Type'])

# Ensure CORS headers are added to every response
@app.after_request
def after_request(response):
    # Allow specific origin
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:4200')
    # Allow specific methods
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    # Allow specific headers
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    # Allow credentials
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Handle preflight requests
@app.route('/analyze', methods=['OPTIONS'])
def handle_preflight():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:4200')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

def encode_plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return plot_base64


def compute_bin_effect(xs, df_dxs, limits):
    """Modified version of Effector's compute_bin_effect function using np.nan instead of np.NaN"""
    empty_symbol = np.nan  # Changed from np.NaN to np.nan

    # Prepare output arrays
    nof_bins = len(limits) - 1
    bin_effects = np.full([nof_bins], empty_symbol)
    points_per_bin = np.zeros([nof_bins], dtype=int)

    # Compute effect for each bin
    for k in range(nof_bins):
        # Get points inside bin k
        indices = np.logical_and(xs >= limits[k], xs < limits[k + 1])
        points_per_bin[k] = np.sum(indices)

        if points_per_bin[k] > 0:
            bin_effects[k] = np.mean(df_dxs[indices])

    return bin_effects, points_per_bin

# Override the utils function
utils.compute_bin_effect = compute_bin_effect

def model_jacobian(model, x):
    with tf.GradientTape() as tape:
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        tape.watch(x_tensor)
        prediction = model(x_tensor)
    return tape.gradient(prediction, x_tensor).numpy()

@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        data_fetcher = DataModelFetcher()

        if 'data' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No data file provided'
            }), 400

        try:
            # Get the data file
            data_file = request.files['data']

            # For CSV files, get column names before processing
            feature_names = None
            if data_file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(data_file.read()))
                # Remove timestamp if present
                if 'Timestamp' in df.columns:
                    df = df.drop('Timestamp', axis=1)
                feature_names = df.columns.tolist()
                # Reset file pointer for parse_data_file
                data_file.seek(0)

            # Parse the data
            X_train = data_fetcher.parse_data_file(data_file)

            # If feature names weren't obtained from CSV, generate them
            if not feature_names:
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

            # Handle model input
            if 'model' in request.files:
                model = data_fetcher.handle_model(model_file=request.files['model'])
            elif request.form.get('model_url'):
                model = data_fetcher.handle_model(model_url=request.form.get('model_url'))
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'No model provided'
                }), 400

            # Get analysis parameters
            method = request.form.get('method', 'pdp')
            feature_index = int(request.form.get('feature_index', 0))
            target_name = request.form.get('target_name', 'prediction')
            node_idx = int(request.form.get('node_idx', 1))

            # Generate feature names if not provided
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

            results = {}

            if method == 'pdp':
                pdp = PDP(
                    data=X_train,
                    model=model.predict,
                    feature_names=feature_names,
                    target_name=target_name,
                    nof_instances=300
                )
                pdp.plot(
                    feature=feature_index,
                    centering=True,
                    show_avg_output=True
                )
                results['pdp_plot'] = encode_plot_to_base64()

            elif method == 'rhale':
                def model_jac(x):
                    return model_jacobian(model, x)

                rhale = RHALE(
                    data=X_train,
                    model=model.predict,
                    model_jac=model_jac,
                    feature_names=feature_names,
                    target_name=target_name
                )
                rhale.plot(
                    feature=feature_index,
                    centering=True,
                    show_avg_output=True
                )
                results['rhale_plot'] = encode_plot_to_base64()

            elif method == 'regional_rhale':
                def model_jac(x):
                    return model_jacobian(model, x)

                regional_rhale = RegionalRHALE(
                    data=X_train,
                    model=model.predict,
                    model_jac=model_jac,
                    cat_limit=10,
                    feature_names=feature_names,
                    nof_instances="all"
                )

                regional_rhale.fit(
                    features=feature_index,
                    heter_small_enough=0.1,
                    heter_pcg_drop_thres=0.2,
                    binning_method=binning_methods.Greedy(
                        init_nof_bins=100,
                        min_points_per_bin=100,
                        discount=1.,
                        cat_limit=10
                    ),
                    max_depth=2,
                    nof_candidate_splits_for_numerical=10,
                    min_points_per_subregion=10,
                    candidate_conditioning_features="all",
                    split_categorical_features=True
                )

                # Capture partitioning info
                partitioning_buffer = io.StringIO()
                regional_rhale.show_partitioning(
                    features=feature_index,
                    only_important=True
                )
                results['partitioning_info'] = partitioning_buffer.getvalue()

                regional_rhale.plot(
                    feature=feature_index,
                    node_idx=node_idx,
                    heterogeneity=True,
                    centering=True
                )
                results['regional_rhale_plot'] = encode_plot_to_base64()

            elif method == 'regional_pdp':
                regional_pdp = RegionalPDP(
                    data=X_train,
                    model=model.predict,
                    cat_limit=10,
                    feature_names=feature_names,
                    nof_instances="all"
                )

                regional_pdp.fit(
                    features=feature_index,
                    heter_small_enough=0.1,
                    heter_pcg_drop_thres=0.1,
                    max_depth=2,
                    nof_candidate_splits_for_numerical=5,
                    min_points_per_subregion=10,
                    candidate_conditioning_features="all",
                    split_categorical_features=True,
                    nof_instances=1000
                )

                # Capture partitioning info
                partitioning_buffer = io.StringIO()
                regional_pdp.show_partitioning(
                    features=feature_index,
                    only_important=True,
                )
                results['partitioning_info'] = partitioning_buffer.getvalue()

                regional_pdp.plot(
                    feature=feature_index,
                    node_idx=node_idx,
                    centering=True
                )
                results['regional_pdp_plot'] = encode_plot_to_base64()

            return jsonify({
                'status': 'success',
                'results': results
            })

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 500

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)