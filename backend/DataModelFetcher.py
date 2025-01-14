import io
import json
import os
import pickle
import tempfile
import traceback

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

from backend.ModelWrapper import ModelWrapper


class DataModelFetcher:
    def parse_data_file(self, file_data):
        """Parse data from uploaded file"""
        try:
            file_extension = file_data.filename.split('.')[-1].lower()
            print(f"Processing file with extension: {file_extension}")

            if file_extension == 'csv':
                # Read CSV with pandas
                df = pd.read_csv(io.BytesIO(file_data.read()))
                print(f"CSV columns: {df.columns.tolist()}")

                # Handle timestamp column if present
                if 'Timestamp' in df.columns:
                    df = df.drop('Timestamp', axis=1)

                # Convert boolean columns to int
                bool_columns = df.select_dtypes(include=['bool']).columns
                for col in bool_columns:
                    df[col] = df[col].astype(int)

                # Ensure all data is numeric
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                if len(numeric_df.columns) != len(df.columns):
                    non_numeric = set(df.columns) - set(numeric_df.columns)
                    print(f"Dropped non-numeric columns: {non_numeric}")

                # Convert to numpy array
                data = numeric_df.to_numpy()

            elif file_extension == 'json':
                json_data = json.loads(file_data.read().decode('utf-8'))
                if isinstance(json_data, dict) and 'data' in json_data:
                    data = np.array(json_data['data'])
                else:
                    data = np.array(json_data)

            elif file_extension == 'npy':
                data = np.load(io.BytesIO(file_data.read()))
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Validate and reshape data
            if data is None or data.size == 0:
                raise ValueError("Data is empty")

            print(f"Data type: {type(data)}")
            print(f"Data shape before processing: {data.shape}")

            # Handle missing values
            if np.isnan(data).any():
                print("Found missing values, filling with mean")
                # Fill missing values with mean of each column
                for col in range(data.shape[1]):
                    col_data = data[:, col]
                    mean_val = np.nanmean(col_data)
                    data[:, col] = np.where(np.isnan(col_data), mean_val, col_data)

            # Ensure data is 2D
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            print(f"Final data shape: {data.shape}")
            print(f"Sample data:\n{data[:2]}")

            return data

        except Exception as e:
            print(f"Error parsing file: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to parse file data: {str(e)}")

    def handle_model(self, model_file=None, model_url=None):
        """Handle model from either file or URL"""
        try:
            if model_file:
                file_extension = model_file.filename.split('.')[-1].lower()
                with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp:
                    tmp.write(model_file.read())
                    model_path = tmp.name
            elif model_url:
                response = requests.get(model_url)
                file_extension = model_url.split('.')[-1].lower()
                with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp:
                    tmp.write(response.content)
                    model_path = tmp.name
            else:
                raise ValueError("No model provided")

            try:
                print(f"Loading model with extension: {file_extension}")

                if file_extension == 'pkl':
                    print("Loading pickle model")
                    with open(model_path, 'rb') as f:
                        original_model = pickle.load(f)
                    print(f"Loaded model type: {type(original_model).__name__}")

                    # Create wrapper
                    model = ModelWrapper(original_model)

                    # Test with minimal sample data
                    try:
                        print("\nTesting prediction...")
                        # Create test data with exact dimensions
                        n_samples = max(model.input_chunk_length, 48)

                        # Create sample data with random values
                        sample_data = np.random.rand(n_samples, model.n_features)
                        # Scale to reasonable values (between 0 and 1)
                        sample_data = (sample_data - sample_data.min()) / (sample_data.max() - sample_data.min())

                        print(f"Test input shape: {sample_data.shape}")
                        print("Testing prediction...")
                        pred = model.predict(sample_data)
                        print(f"Test prediction successful, shape: {pred.shape}")

                    except Exception as e:
                        print("Test prediction failed:")
                        print(f"Error: {str(e)}")
                        print(f"Traceback: {traceback.format_exc()}")
                        raise
                    return model

                elif file_extension in ['h5', 'keras']:
                    with tf.keras.utils.custom_object_scope({}):
                        model = tf.keras.models.load_model(model_path, compile=False)
                    return model
                else:
                    raise ValueError(f"Unsupported format: {file_extension}")

            finally:
                if os.path.exists(model_path):
                    os.unlink(model_path)
                    print("Cleaned up temporary files")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to load model: {str(e)}")