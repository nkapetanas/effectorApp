import traceback
from functools import reduce

import numpy as np


class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.model_type = type(model).__name__
        print(f"Initializing wrapper for model type: {self.model_type}")

        # Default configuration
        self.input_chunk_length = 24
        self.output_chunk_length = 1
        self.n_features = 11
        self.n_covariates = 11
        self.target_idx = -1

        # Try to get configuration from model if available
        if hasattr(model, 'input_chunk_length'):
            self.input_chunk_length = model.input_chunk_length
        if hasattr(model, 'output_chunk_length'):
            self.output_chunk_length = model.output_chunk_length

        print("Model configuration:")
        print(f"- Input chunk length: {self.input_chunk_length}")
        print(f"- Output chunk length: {self.output_chunk_length}")
        print(f"- Total features: {self.n_features}")
        print(f"- Expected covariates: {self.n_covariates}")
        print(f"- Target index: {self.target_idx}")

    def _create_time_series(self, X):
        """Create Darts TimeSeries with correct dimensions"""
        import pandas as pd
        from darts import TimeSeries
        import numpy as np

        print(f"Input data shape: {X.shape}")

        # Ensure we have enough data points
        if len(X) < self.input_chunk_length:
            raise ValueError(f"Need at least {self.input_chunk_length} data points, got {len(X)}")

        # Create dates aligned with the end
        dates = pd.date_range(
            end=pd.Timestamp.now(),
            periods=len(X),
            freq='h'
        )

        # Split features and target
        features = X[:, :self.n_covariates]  # Take only first n_covariates columns
        target = X[:, self.target_idx].reshape(-1, 1)  # Reshape target to 2D

        # Create target series
        target_df = pd.DataFrame(
            target,
            index=dates,
            columns=['target']
        )
        target_series = TimeSeries.from_dataframe(
            target_df,
            freq='h',
            fill_missing_dates=True
        )

        # Create separate series for each covariate
        covariate_series = []
        for i in range(self.n_covariates):
            df = pd.DataFrame(
                features[:, i].reshape(-1, 1),
                index=dates,
                columns=[f'feature_{i}']
            )
            series = TimeSeries.from_dataframe(
                df,
                freq='h',
                fill_missing_dates=True
            )
            covariate_series.append(series)

        # Combine all covariate series
        covariates = reduce(
            lambda x, y: x.concatenate(y, axis=1),
            covariate_series
        )

        print("Time series created:")
        print(f"- Target shape: {target_series.values().shape}")
        print(f"- Covariates shape: {covariates.values().shape}")
        print(f"- Date range: {dates[0]} to {dates[-1]}")
        print(f"- Number of covariate components: {len(covariate_series)}")

        return target_series, covariates

    def predict(self, X):
        print(f"Starting prediction with input shape: {X.shape}")
        try:
            if self.model_type == 'CatBoostModel':
                print("Creating time series for prediction")
                target_series, covariates = self._create_time_series(X)

                print("Making prediction")
                print(f"Target series shape: {target_series.values().shape}")
                print(f"Covariates shape: {covariates.values().shape}")

                # Use only the required window length
                window_start = -self.input_chunk_length if len(target_series) >= self.input_chunk_length else None

                predictions = self.model.predict(
                    n=1,
                    series=target_series[window_start:],
                    future_covariates=covariates[window_start:]
                )

                result = predictions.values().flatten()
                print(f"Prediction shape: {result.shape}")
                return result
            else:
                return self.model.predict(X)

        except Exception as e:
            print(f"Prediction error in {self.model_type}:")
            print(f"Error message: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise