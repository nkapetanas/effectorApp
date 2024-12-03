import io
import json
import os
import tempfile

import numpy as np
import pandas as pd
import requests
import tensorflow as tf


class DataModelFetcher:
    def fetch_from_url(self, url):
        """Fetch data from URL"""
        response = requests.get(url)
        content_type = response.headers.get('content-type', '')

        try:
            if 'application/octet-stream' in content_type:
                return np.load(io.BytesIO(response.content))
            elif 'application/json' in content_type:
                data = response.json()
                if isinstance(data, dict) and 'data' in data:
                    return np.array(data['data'])
                return np.array(data)
            elif 'text/csv' in content_type:
                df = pd.read_csv(io.StringIO(response.text))
                return df.to_numpy()
            else:
                return np.load(io.BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"Failed to parse data from URL: {str(e)}")

    def parse_data_file(self, file_data):
        """Parse data from uploaded file"""
        try:
            file_extension = file_data.filename.split('.')[-1].lower()
            file_content = file_data.read()

            if file_extension == 'npy':
                return np.load(io.BytesIO(file_content))
            elif file_extension == 'csv':
                return pd.read_csv(io.BytesIO(file_content)).to_numpy()
            elif file_extension == 'json':
                data = json.loads(file_content.decode('utf-8'))
                if isinstance(data, dict) and 'data' in data:
                    return np.array(data['data'])
                return np.array(data)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to parse file data: {str(e)}")

    def handle_model(self, model_file=None, model_url=None):
        """Handle model from either file or URL"""
        try:
            if model_file:
                # Check file extension
                file_extension = model_file.filename.split('.')[-1].lower()
                if file_extension not in ['h5', 'keras']:
                    raise ValueError(
                        f"Unsupported model format: {file_extension}. Only .h5 and .keras files are supported.")

                # Create temp file for uploaded model
                with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp:
                    tmp.write(model_file.read())
                    model_path = tmp.name
            elif model_url:
                # Download model from URL
                response = requests.get(model_url)
                # Try to determine extension from URL
                url_extension = model_url.split('.')[-1].lower()
                if url_extension not in ['h5', 'keras']:
                    raise ValueError(
                        f"Unsupported model format: {url_extension}. Only .h5 and .keras files are supported.")

                with tempfile.NamedTemporaryFile(suffix=f'.{url_extension}', delete=False) as tmp:
                    tmp.write(response.content)
                    model_path = tmp.name
            else:
                raise ValueError("No model provided")

            try:
                # Load the model with custom object scope for better compatibility
                with tf.keras.utils.custom_object_scope({}):
                    model = tf.keras.models.load_model(
                        model_path,
                        compile=False  # Don't require optimizer/loss function
                    )
                return model
            except Exception as e:
                raise ValueError(f"Failed to load model: {str(e)}")
            finally:
                # Clean up temp file
                if os.path.exists(model_path):
                    os.unlink(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
