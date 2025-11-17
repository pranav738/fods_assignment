"""
Temporal data preprocessing for high-frequency traffic forecasting

Handles resampling from daily to high-frequency (hourly/15-min) data,
feature engineering, and data splitting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler


class TemporalPreprocessor:
    """
    Preprocess temporal traffic data for GNN models

    This class addresses the key limitation identified in the research document:
    the need to move from daily aggregation to high-frequency data.
    """

    def __init__(self,
                 freq: str = '15T',
                 window_size: int = 12,
                 horizon: int = 12,
                 normalize: bool = True):
        """
        Initialize preprocessor

        Args:
            freq: Resampling frequency ('15T' = 15 min, '1H' = hourly, etc.)
            window_size: Number of historical time steps to use (input sequence length)
            horizon: Number of future time steps to predict (output sequence length)
            normalize: Whether to normalize features
        """
        self.freq = freq
        self.window_size = window_size
        self.horizon = horizon
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.feature_columns = None

    def resample_to_high_frequency(self,
                                   df: pd.DataFrame,
                                   timestamp_col: str = 'Timestamp',
                                   value_col: str = 'Traffic Volume',
                                   agg_method: str = 'mean') -> pd.DataFrame:
        """
        Resample daily data to high-frequency intervals

        Args:
            df: Input DataFrame with daily data
            timestamp_col: Name of timestamp column
            value_col: Name of value column to aggregate
            agg_method: Aggregation method ('mean', 'sum', 'max', etc.)

        Returns:
            Resampled DataFrame
        """
        # Ensure timestamp is datetime
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Set timestamp as index
        df = df.set_index(timestamp_col)

        # Resample to target frequency
        if agg_method == 'mean':
            df_resampled = df.resample(self.freq).mean()
        elif agg_method == 'sum':
            df_resampled = df.resample(self.freq).sum()
        elif agg_method == 'max':
            df_resampled = df.resample(self.freq).max()
        elif agg_method == 'interpolate':
            # Create regular time index
            df_resampled = df.resample(self.freq).asfreq()
            # Interpolate missing values
            df_resampled = df_resampled.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")

        return df_resampled.reset_index()

    def add_temporal_features(self, df: pd.DataFrame, timestamp_col: str = 'Timestamp') -> pd.DataFrame:
        """
        Add temporal features for high-frequency data

        Features:
        - hour_of_day (cyclical encoding)
        - day_of_week (cyclical encoding)
        - day_of_year (cyclical encoding)
        - is_weekend
        - is_peak_hour

        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()

        # Ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Extract temporal components
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_year'] = df[timestamp_col].dt.dayofyear
        df['month'] = df[timestamp_col].dt.month

        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Cyclical encoding for day of year
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Peak hours: 7-10 AM and 5-8 PM
        df['is_peak_hour'] = (
            ((df['hour'] >= 7) & (df['hour'] <= 10)) |
            ((df['hour'] >= 17) & (df['hour'] <= 20))
        ).astype(int)

        # Drop non-cyclical temporal columns
        df = df.drop(columns=['hour', 'day_of_week', 'day_of_year'], errors='ignore')

        return df

    def add_holiday_features(self,
                           df: pd.DataFrame,
                           timestamp_col: str = 'Timestamp',
                           country: str = 'IN') -> pd.DataFrame:
        """
        Add holiday indicators

        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            country: Country code for holidays

        Returns:
            DataFrame with holiday features
        """
        try:
            import holidays
        except ImportError:
            print("Warning: 'holidays' package not installed. Skipping holiday features.")
            return df

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Get holidays for India
        indian_holidays = holidays.country_holidays(country)

        # Add holiday indicator
        df['is_holiday'] = df[timestamp_col].dt.date.apply(
            lambda x: 1 if x in indian_holidays else 0
        )

        return df

    def add_event_features(self,
                         df: pd.DataFrame,
                         events_df: Optional[pd.DataFrame] = None,
                         timestamp_col: str = 'Timestamp',
                         location_col: Optional[str] = None) -> pd.DataFrame:
        """
        Add event-based features (cricket matches, concerts, etc.)

        Args:
            df: Input DataFrame
            events_df: DataFrame with events (columns: timestamp, event_type, location)
            timestamp_col: Name of timestamp column
            location_col: Name of location column

        Returns:
            DataFrame with event features
        """
        df = df.copy()

        if events_df is None:
            # Just add placeholder columns
            df['is_major_event'] = 0
            df['event_proximity'] = 0.0
            return df

        # Merge with events
        events_df = events_df.copy()
        events_df[timestamp_col] = pd.to_datetime(events_df[timestamp_col])

        # Add event indicators
        # (Implementation depends on events_df structure)

        return df

    def create_sequences(self,
                        data: np.ndarray,
                        window_size: Optional[int] = None,
                        horizon: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for seq2seq learning

        Args:
            data: Input data (T x N x F) where T=time, N=nodes, F=features
            window_size: Input sequence length (default: self.window_size)
            horizon: Output sequence length (default: self.horizon)

        Returns:
            Tuple of (X, y) where:
                X: (num_samples, window_size, num_nodes, num_features)
                y: (num_samples, horizon, num_nodes, 1)
        """
        if window_size is None:
            window_size = self.window_size
        if horizon is None:
            horizon = self.horizon

        num_timesteps, num_nodes, num_features = data.shape
        num_samples = num_timesteps - window_size - horizon + 1

        X = np.zeros((num_samples, window_size, num_nodes, num_features))
        y = np.zeros((num_samples, horizon, num_nodes, 1))

        for i in range(num_samples):
            X[i] = data[i:i + window_size, :, :]
            # Target is the first feature (traffic volume)
            y[i] = data[i + window_size:i + window_size + horizon, :, 0:1]

        return X, y

    def train_val_test_split(self,
                            df: pd.DataFrame,
                            timestamp_col: str = 'Timestamp',
                            train_ratio: float = 0.6,
                            val_ratio: float = 0.2,
                            test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronological train/validation/test split

        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            train_ratio: Proportion of data for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        # Sort by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def pivot_to_graph_format(self,
                              df: pd.DataFrame,
                              timestamp_col: str = 'Timestamp',
                              node_col: str = 'Road/Intersection Name',
                              feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """
        Pivot DataFrame to (T x N x F) format required by GNN models

        Args:
            df: Input DataFrame with multiple nodes
            timestamp_col: Name of timestamp column
            node_col: Name of node/location column
            feature_cols: List of feature columns (if None, use all numeric columns)

        Returns:
            Data tensor of shape (T, N, F)
        """
        if feature_cols is None:
            # Use all numeric columns except timestamp and node
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Get unique timestamps and nodes
        timestamps = sorted(df[timestamp_col].unique())
        nodes = sorted(df[node_col].unique())

        T = len(timestamps)
        N = len(nodes)
        F = len(feature_cols)

        # Initialize data tensor
        data = np.zeros((T, N, F))

        # Fill data tensor
        for t_idx, timestamp in enumerate(timestamps):
            for n_idx, node in enumerate(nodes):
                # Get data for this timestamp and node
                mask = (df[timestamp_col] == timestamp) & (df[node_col] == node)
                node_data = df.loc[mask, feature_cols].values

                if len(node_data) > 0:
                    data[t_idx, n_idx, :] = node_data[0]
                # else: remains zero (missing data)

        return data

    def normalize_features(self,
                          train_data: np.ndarray,
                          val_data: Optional[np.ndarray] = None,
                          test_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """
        Normalize features using StandardScaler

        Fits scaler on training data only to prevent data leakage.

        Args:
            train_data: Training data (T x N x F)
            val_data: Validation data (optional)
            test_data: Test data (optional)

        Returns:
            Tuple of normalized data arrays
        """
        if not self.normalize:
            if val_data is None and test_data is None:
                return train_data
            elif test_data is None:
                return train_data, val_data
            else:
                return train_data, val_data, test_data

        # Reshape to (T*N, F) for scaling
        T, N, F = train_data.shape
        train_reshaped = train_data.reshape(-1, F)

        # Fit scaler on training data
        self.scaler.fit(train_reshaped)

        # Transform training data
        train_normalized = self.scaler.transform(train_reshaped).reshape(T, N, F)

        results = [train_normalized]

        # Transform validation data if provided
        if val_data is not None:
            T_val, N_val, F_val = val_data.shape
            val_reshaped = val_data.reshape(-1, F_val)
            val_normalized = self.scaler.transform(val_reshaped).reshape(T_val, N_val, F_val)
            results.append(val_normalized)

        # Transform test data if provided
        if test_data is not None:
            T_test, N_test, F_test = test_data.shape
            test_reshaped = test_data.reshape(-1, F_test)
            test_normalized = self.scaler.transform(test_reshaped).reshape(T_test, N_test, F_test)
            results.append(test_normalized)

        return tuple(results) if len(results) > 1 else results[0]

    def inverse_transform(self, data: np.ndarray, feature_idx: int = 0) -> np.ndarray:
        """
        Inverse transform normalized predictions back to original scale

        Args:
            data: Normalized data
            feature_idx: Index of feature to inverse transform (default: 0 = traffic volume)

        Returns:
            Data in original scale
        """
        if not self.normalize or self.scaler is None:
            return data

        # Create dummy array with all features
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)

        # Create full feature array (all zeros except the target feature)
        num_features = len(self.scaler.mean_)
        full_data = np.zeros((len(data_flat), num_features))
        full_data[:, feature_idx] = data_flat[:, 0]

        # Inverse transform
        inverse_transformed = self.scaler.inverse_transform(full_data)

        # Extract target feature
        result = inverse_transformed[:, feature_idx].reshape(original_shape)

        return result

    def preprocess_pipeline(self,
                           df: pd.DataFrame,
                           timestamp_col: str = 'Timestamp',
                           node_col: str = 'Road/Intersection Name',
                           target_col: str = 'Traffic Volume') -> Dict:
        """
        Complete preprocessing pipeline

        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            node_col: Name of node column
            target_col: Name of target column

        Returns:
            Dictionary containing all processed data and metadata
        """
        print("Starting preprocessing pipeline...")

        # 1. Add temporal features
        print("Adding temporal features...")
        df = self.add_temporal_features(df, timestamp_col)

        # 2. Add holiday features
        print("Adding holiday features...")
        df = self.add_holiday_features(df, timestamp_col)

        # 3. Split data
        print("Splitting data...")
        train_df, val_df, test_df = self.train_val_test_split(df, timestamp_col)

        # 4. Convert to graph format
        print("Converting to graph format...")
        feature_cols = [col for col in df.columns
                       if col not in [timestamp_col, node_col] and pd.api.types.is_numeric_dtype(df[col])]

        train_data = self.pivot_to_graph_format(train_df, timestamp_col, node_col, feature_cols)
        val_data = self.pivot_to_graph_format(val_df, timestamp_col, node_col, feature_cols)
        test_data = self.pivot_to_graph_format(test_df, timestamp_col, node_col, feature_cols)

        # 5. Normalize
        print("Normalizing features...")
        train_data, val_data, test_data = self.normalize_features(train_data, val_data, test_data)

        # 6. Create sequences
        print("Creating sequences...")
        X_train, y_train = self.create_sequences(train_data)
        X_val, y_val = self.create_sequences(val_data)
        X_test, y_test = self.create_sequences(test_data)

        print(f"Train: X={X_train.shape}, y={y_train.shape}")
        print(f"Val: X={X_val.shape}, y={y_val.shape}")
        print(f"Test: X={X_test.shape}, y={y_test.shape}")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_columns': feature_cols,
            'scaler': self.scaler,
            'nodes': sorted(df[node_col].unique())
        }
