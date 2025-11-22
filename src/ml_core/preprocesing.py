"""
Data Preprocessing Module for Jakarta FloodNet
=============================================

This module contains functions for data cleaning, preprocessing,
and feature engineering for flood level prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and perform initial data exploration.
    
    Args:
        file_path: Path to the CSV data file
        
    Returns:
        Loaded dataframe
    """
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Drop unnamed index column if exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw data.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Cleaned dataframe
    """
    print("Starting data cleaning...")
    df_clean = df.copy()
    
    # Parse datetime - assuming first column is timestamp
    timestamp_col = df.columns[0]
    df_clean['timestamp'] = pd.to_datetime(df_clean[timestamp_col])
    
    # Rename columns to standard format
    if len(df.columns) >= 4:
        df_clean = df_clean.rename(columns={
            df.columns[1]: 'hujan_bogor',
            df.columns[2]: 'hujan_jakarta', 
            df.columns[3]: 'tma_manggarai'
        })
    
    # Keep only relevant columns
    df_clean = df_clean[['timestamp', 'hujan_bogor', 'hujan_jakarta', 'tma_manggarai']]
    
    # Sort by timestamp
    df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
    
    # Data type conversion
    numeric_cols = ['hujan_bogor', 'hujan_jakarta', 'tma_manggarai']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    print(f"Data cleaned. Shape: {df_clean.shape}")
    return df_clean

def handle_missing_values(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input dataframe
        method: Method to handle missing values ('forward_fill', 'interpolate', 'drop')
        
    Returns:
        Dataframe with missing values handled
    """
    print(f"Handling missing values using method: {method}")
    df_filled = df.copy()
    
    # Check for missing values
    missing_counts = df_filled.isnull().sum()
    print(f"Missing values before handling:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    
    if method == 'forward_fill':
        # Forward fill then backward fill for time series
        df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
    elif method == 'interpolate':
        # Linear interpolation for numeric columns
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method='linear')
    elif method == 'drop':
        # Drop rows with missing values
        df_filled = df_filled.dropna()
    
    # Final check
    remaining_missing = df_filled.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values remain")
        # Drop any remaining missing values
        df_filled = df_filled.dropna()
    
    print(f"Missing values handled. Final shape: {df_filled.shape}")
    return df_filled

def detect_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> Dict[str, np.ndarray]:
    """
    Detect outliers in specified columns.
    
    Args:
        df: Input dataframe
        columns: List of column names to check for outliers
        method: Method for outlier detection ('iqr', 'zscore')
        
    Returns:
        Dictionary with column names as keys and boolean arrays as values
    """
    outliers = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col]))
            outlier_mask = z_scores > 3
        
        outliers[col] = outlier_mask
        outlier_count = outlier_mask.sum()
        print(f"{col}: {outlier_count} outliers ({outlier_count/len(df)*100:.2f}%)")
    
    return outliers

def create_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Create time-based features from timestamp.
    
    Args:
        df: Input dataframe
        timestamp_col: Name of timestamp column
        
    Returns:
        Dataframe with additional time features
    """
    print("Creating time-based features...")
    df_time = df.copy()
    
    # Ensure timestamp is datetime
    df_time[timestamp_col] = pd.to_datetime(df_time[timestamp_col])
    
    # Extract time components
    df_time['hour'] = df_time[timestamp_col].dt.hour
    df_time['day_of_week'] = df_time[timestamp_col].dt.dayofweek
    df_time['day_of_month'] = df_time[timestamp_col].dt.day
    df_time['month'] = df_time[timestamp_col].dt.month
    df_time['is_weekend'] = (df_time['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for periodic features
    df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
    df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
    df_time['day_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
    df_time['day_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)
    df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
    df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
    
    print(f"Added {len(['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'])} time features")
    return df_time

def create_rainfall_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rainfall-based features.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with additional rainfall features
    """
    print("Creating rainfall combination features...")
    df_rain = df.copy()
    
    # Basic combinations
    df_rain['total_rainfall'] = df_rain['hujan_bogor'] + df_rain['hujan_jakarta']
    df_rain['rainfall_diff'] = df_rain['hujan_bogor'] - df_rain['hujan_jakarta']
    df_rain['rainfall_ratio'] = df_rain['hujan_bogor'] / (df_rain['hujan_jakarta'] + 1e-6)
    
    # Rainfall intensity categories
    df_rain['bogor_intensity'] = pd.cut(
        df_rain['hujan_bogor'], 
        bins=[0, 1, 5, 10, float('inf')], 
        labels=['ringan', 'sedang', 'lebat', 'sangat_lebat']
    ).astype(str)
    
    df_rain['jakarta_intensity'] = pd.cut(
        df_rain['hujan_jakarta'], 
        bins=[0, 1, 5, 10, float('inf')], 
        labels=['ringan', 'sedang', 'lebat', 'sangat_lebat']
    ).astype(str)
    
    # One-hot encode intensity categories
    bogor_dummies = pd.get_dummies(df_rain['bogor_intensity'], prefix='bogor')
    jakarta_dummies = pd.get_dummies(df_rain['jakarta_intensity'], prefix='jakarta')
    
    df_rain = pd.concat([df_rain, bogor_dummies, jakarta_dummies], axis=1)
    df_rain = df_rain.drop(['bogor_intensity', 'jakarta_intensity'], axis=1)
    
    print(f"Added rainfall combination features")
    return df_rain

def create_lag_features(
    df: pd.DataFrame, 
    columns: List[str], 
    lag_hours: List[int]
) -> pd.DataFrame:
    """
    Create lag features for specified columns.
    
    Args:
        df: Input dataframe
        columns: List of column names to create lags for
        lag_hours: List of lag periods (in hours)
        
    Returns:
        Dataframe with lag features
    """
    print(f"Creating lag features for {columns} with lags {lag_hours}")
    df_lag = df.copy()
    
    for col in columns:
        for lag in lag_hours:
            if lag < len(df_lag):
                df_lag[f'{col}_lag_{lag}h'] = df_lag[col].shift(lag)
    
    print(f"Added {len(columns) * len(lag_hours)} lag features")
    return df_lag

def create_rolling_features(
    df: pd.DataFrame, 
    columns: List[str], 
    windows: List[int],
    agg_funcs: List[str] = ['mean']
) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Args:
        df: Input dataframe
        columns: List of column names to create rolling features for
        windows: List of window sizes (in hours)
        agg_funcs: List of aggregation functions ('mean', 'std', 'max', 'min', 'sum')
        
    Returns:
        Dataframe with rolling features
    """
    print(f"Creating rolling features for {columns} with windows {windows}")
    df_rolling = df.copy()
    
    for col in columns:
        for window in windows:
            if window <= len(df_rolling):
                for agg_func in agg_funcs:
                    if agg_func == 'mean':
                        df_rolling[f'{col}_rolling_{window}h_mean'] = df_rolling[col].rolling(window, min_periods=1).mean()
                    elif agg_func == 'std':
                        df_rolling[f'{col}_rolling_{window}h_std'] = df_rolling[col].rolling(window, min_periods=1).std()
                    elif agg_func == 'max':
                        df_rolling[f'{col}_rolling_{window}h_max'] = df_rolling[col].rolling(window, min_periods=1).max()
                    elif agg_func == 'min':
                        df_rolling[f'{col}_rolling_{window}h_min'] = df_rolling[col].rolling(window, min_periods=1).min()
                    elif agg_func == 'sum':
                        df_rolling[f'{col}_rolling_{window}h_sum'] = df_rolling[col].rolling(window, min_periods=1).sum()
    
    print(f"Added {len(columns) * len(windows) * len(agg_funcs)} rolling features")
    return df_rolling

def engineer_features(
    df: pd.DataFrame,
    lag_hours: List[int] = [1, 2, 3, 6, 12, 24],
    rolling_windows: List[int] = [3, 6, 12, 24],
    tma_lags: List[int] = [1, 2, 3, 6]
) -> pd.DataFrame:
    """
    Comprehensive feature engineering pipeline.
    
    Args:
        df: Input dataframe (should be cleaned)
        lag_hours: List of lag periods for rainfall features
        rolling_windows: List of rolling window sizes
        tma_lags: List of lag periods for water level features
        
    Returns:
        Dataframe with engineered features
    """
    print("Starting comprehensive feature engineering...")
    df_features = df.copy()
    
    # Time-based features
    df_features = create_time_features(df_features)
    
    # Rainfall combination features
    df_features = create_rainfall_features(df_features)
    
    # Lag features for rainfall
    rainfall_cols = ['hujan_bogor', 'hujan_jakarta', 'total_rainfall']
    df_features = create_lag_features(df_features, rainfall_cols, lag_hours)
    
    # Rolling features for rainfall
    df_features = create_rolling_features(df_features, rainfall_cols, rolling_windows, ['mean'])
    
    # Lag features for water level (autoregressive)
    df_features = create_lag_features(df_features, ['tma_manggarai'], tma_lags)
    
    # Remove rows with NaN values created by lagging
    initial_rows = len(df_features)
    df_features = df_features.dropna().reset_index(drop=True)
    removed_rows = initial_rows - len(df_features)
    
    print(f"Feature engineering complete:")
    print(f"  - Total features: {len(df_features.columns)}")
    print(f"  - Removed {removed_rows} rows due to NaN values")
    print(f"  - Final shape: {df_features.shape}")
    
    return df_features

def preprocess_data(
    file_path: str,
    feature_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        file_path: Path to the data file
        feature_config: Configuration for feature engineering
        
    Returns:
        Fully preprocessed dataframe
    """
    print("="*60)
    print("COMPREHENSIVE DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Default feature configuration
    if feature_config is None:
        feature_config = {
            'lag_hours': [1, 2, 3, 6, 12, 24],
            'rolling_windows': [3, 6, 12, 24],
            'tma_lags': [1, 2, 3, 6]
        }
    
    # Load data
    df = load_data(file_path)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Handle missing values
    df_clean = handle_missing_values(df_clean, method='forward_fill')
    
    # Detect outliers (for information only)
    numeric_cols = ['hujan_bogor', 'hujan_jakarta', 'tma_manggarai']
    outliers = detect_outliers(df_clean, numeric_cols, method='iqr')
    
    # Feature engineering
    df_features = engineer_features(
        df_clean,
        lag_hours=feature_config['lag_hours'],
        rolling_windows=feature_config['rolling_windows'],
        tma_lags=feature_config['tma_lags']
    )
    
    print("="*60)
    print("PREPROCESSING COMPLETED")
    print("="*60)
    
    return df_features
