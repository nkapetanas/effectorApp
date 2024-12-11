import io
import json
import os
import tempfile
import traceback

import numpy as np
import pandas as pd
import requests
import tensorflow as tf


class DataModelFetcher:
    def parse_data_file(self, file_data):
        """Parse data from uploaded file"""
        try:
            file_extension = file_data.filename.split('.')[-1].lower()
            file_content = file_data.read()

            print(f"Processing file with extension: {file_extension}")

            if file_extension == 'npy':
                data = np.load(io.BytesIO(file_content))
            elif file_extension == 'csv':
                data = pd.read_csv(io.BytesIO(file_content)).to_numpy()
            elif file_extension == 'json':
                json_data = json.loads(file_content.decode('utf-8'))

                # Debug print
                print("JSON data type:", type(json_data))
                print("JSON data sample:", json_data[:5] if isinstance(json_data, list) else json_data)

                # Handle different JSON structures
                if isinstance(json_data, dict):
                    if 'data' in json_data:
                        data = np.array(json_data['data'])
                    else:
                        # If it's a dict without 'data' key, try to convert values
                        data = np.array(list(json_data.values()))
                elif isinstance(json_data, list):
                    # If it's a list of lists, convert directly
                    if json_data and isinstance(json_data[0], list):
                        data = np.array(json_data)
                    # If it's a list of dicts, convert to DataFrame first
                    elif json_data and isinstance(json_data[0], dict):
                        df = pd.DataFrame(json_data)
                        data = df.to_numpy()
                    else:
                        data = np.array(json_data)
                else:
                    raise ValueError(f"Unsupported JSON structure")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Ensure data is 2D
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.ndim > 2:
                raise ValueError(f"Data has too many dimensions: {data.ndim}")

            print(f"Parsed data shape: {data.shape}")
            print(f"Data sample: {data[:2]}")  # Show first two rows

            return data

        except Exception as e:
            print(f"Error parsing file: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to parse file data: {str(e)}")

    def handle_model(self, model_file=None, model_url=None):
        """Handle model from either file or URL"""
        try:
            if model_file:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(model_file.read())
                    model_path = tmp.name
            elif model_url:
                response = requests.get(model_url)
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(response.content)
                    model_path = tmp.name
            else:
                raise ValueError("No model provided")

            try:
                model = tf.keras.models.load_model(model_path)
                return model
            finally:
                if os.path.exists(model_path):
                    os.unlink(model_path)

        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

    def parse_data_file(self, file_data):
        """Parse data from uploaded file"""
        try:
            file_extension = file_data.filename.split('.')[-1].lower()
            file_content = file_data.read()

            print(f"Processing file with extension: {file_extension}")

            if file_extension == 'npy':
                data = np.load(io.BytesIO(file_content))
            elif file_extension == 'csv':
                data = pd.read_csv(io.BytesIO(file_content)).to_numpy()
            elif file_extension == 'json':
                data = json.loads(file_content.decode('utf-8'))
                if isinstance(data, dict) and 'data' in data:
                    data = data['data']
                data = np.array(data)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Ensure data is 2D
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            print(f"Parsed data shape: {data.shape}")
            return data

        except Exception as e:
            print(f"Error parsing file: {str(e)}")
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
