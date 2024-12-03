import json
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import numpy as np
import tensorflow as tf
from effector import PDP, RHALE
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

@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        fetcher = DataModelFetcher()

        # Debug logging
        print("Request form data:", request.form)
        print("Request files:", request.files)

        try:
            # Handle data input
            if 'data' in request.files:
                print("Processing uploaded data file...")
                data_file = request.files['data']
                print(f"Data file name: {data_file.filename}")
                X_train = fetcher.parse_data_file(data_file)
            elif request.form.get('data_url'):
                print("Processing data from URL...")
                X_train = fetcher.fetch_from_url(request.form.get('data_url'))
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'No data provided'
                }), 400

            print(f"Data shape: {X_train.shape}")

            # Handle model input
            if 'model' in request.files:
                print("Processing uploaded model file...")
                model_file = request.files['model']
                print(f"Model file name: {model_file.filename}")
                model = fetcher.handle_model(model_file=model_file)
            elif request.form.get('model_url'):
                print("Processing model from URL...")
                model = fetcher.handle_model(model_url=request.form.get('model_url'))
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'No model provided'
                }), 400

        except Exception as e:
            import traceback
            print("Error processing input:", str(e))
            print("Traceback:", traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f'Error processing input: {str(e)}\n{traceback.format_exc()}'
            }), 400

        # Process request parameters
        try:
            method = request.form.get('method', 'pdp')
            feature_index = int(request.form.get('feature_index', 0))
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

            print(f"Analysis method: {method}")
            print(f"Feature index: {feature_index}")

            results = {}

            if method == 'pdp':
                print("Running PDP analysis...")
                pdp = PDP(
                    data=X_train,
                    model=model.predict,
                    feature_names=feature_names,
                    target_name="prediction",
                    nof_instances=300
                )
                pdp.plot(
                    feature=feature_index,
                    centering=True,
                    show_avg_output=True
                )
                results['pdp_plot'] = encode_plot_to_base64()

            elif method == 'rhale':
                print("Running RHALE analysis...")
                def model_jacobian(x):
                    with tf.GradientTape() as tape:
                        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
                        tape.watch(x_tensor)
                        prediction = model(x_tensor)
                    return tape.gradient(prediction, x_tensor).numpy()

                rhale = RHALE(
                    data=X_train,
                    model=model.predict,
                    model_jac=model_jacobian,
                    feature_names=feature_names,
                    target_name="prediction"
                )
                rhale.plot(
                    feature=feature_index,
                    centering=True,
                    show_avg_output=True
                )
                results['rhale_plot'] = encode_plot_to_base64()

            return jsonify({
                'status': 'success',
                'results': results
            })

        except Exception as e:
            import traceback
            print("Error during analysis:", str(e))
            print("Traceback:", traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f'Error during analysis: {str(e)}\n{traceback.format_exc()}'
            }), 500

    except Exception as e:
        import traceback
        print("Unexpected error:", str(e))
        print("Traceback:", traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Unexpected error: {str(e)}\n{traceback.format_exc()}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)