from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from effector import PDP, RHALE
import requests

import os
import io
import base64
import matplotlib.pyplot as plt
import json

from DataModelFetcher import DataModelFetcher

app = Flask(__name__)
CORS(app)


def encode_plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return plot_base64

def publish_results(url, results):
    if url:
        try:
            response = requests.post(url, json=results)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    return True

@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        data = request.get_json()
        fetcher = DataModelFetcher()

        # Fetch data and model from provided URLs
        X_train = fetcher.fetch_data(data['data_url'])
        model_path = fetcher.fetch_model(data['model_url'])

        feature_names = data.get('feature_names', [f'feature_{i}' for i in range(X_train.shape[1])])
        feature_index = data.get('feature_index', 0)
        method = data.get('method', 'pdp')
        publish_url = data.get('publish_url')

        # Load the model
        model = tf.keras.models.load_model(model_path)
        os.unlink(model_path)  # Clean up temporary file

        results = {}

        if method == 'pdp':
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

        # Publish results if URL provided
        publish_success = publish_results(publish_url, results)

        return jsonify({
            'status': 'success',
            'results': results,
            'published': publish_success
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)