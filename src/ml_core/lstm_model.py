"""
LSTM Model Module for Jakarta FloodNet
====================================

This module contains the LSTM model class for water level prediction
based on rainfall data from Bogor and Jakarta.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import os
from typing import Tuple, List, Optional, Dict, Any

class FloodLevelLSTM:
    """
    LSTM model for predicting water level (tma_manggarai) based on rainfall data.
    """
    
    def __init__(
        self,
        sequence_length: int = 24,
        lstm_units: List[int] = [64, 32, 16],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps to use for prediction
            lstm_units: List of LSTM layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_cols = None
        self.is_trained = False
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
        """
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(
            self.lstm_units[0], 
            return_sequences=True, 
            input_shape=input_shape
        ))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate))
        
        # Second LSTM layer
        self.model.add(LSTM(
            self.lstm_units[1], 
            return_sequences=True if len(self.lstm_units) > 2 else False
        ))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate))
        
        # Third LSTM layer (if specified)
        if len(self.lstm_units) > 2:
            self.model.add(LSTM(self.lstm_units[2], return_sequences=False))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        self.model.add(Dense(32, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate / 2))
        
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
    def create_sequences(
        self, 
        data: pd.DataFrame, 
        target_col: str, 
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Input dataframe
            target_col: Name of target column
            feature_cols: List of feature column names
            
        Returns:
            X: Feature sequences
            y: Target values
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            # Get sequence of features
            X.append(data[feature_cols].iloc[i-self.sequence_length:i].values)
            # Get corresponding target
            y.append(data[target_col].iloc[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'tma_manggarai',
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Tuple[np.ndarray, ...]:
        """
        Prepare data for training including scaling and sequence creation.
        
        Args:
            df: Input dataframe with features and target
            target_col: Name of target column
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            
        Returns:
            Tuple of train/val/test sets
        """
        # Define feature columns (exclude timestamp and target)
        exclude_cols = ['timestamp', target_col]
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Initialize scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()
        
        # Scale features and target
        X_scaled = self.scaler_X.fit_transform(df[self.feature_cols])
        y_scaled = self.scaler_y.fit_transform(df[target_col].values.reshape(-1, 1)).flatten()
        
        # Update dataframe with scaled values
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = X_scaled
        df_scaled[target_col] = y_scaled
        
        # Create sequences
        X_sequences, y_sequences = self.create_sequences(
            df_scaled, target_col, self.feature_cols
        )
        
        # Split data (temporal split to avoid data leakage)
        split_idx_1 = int(len(X_sequences) * (1 - test_size - val_size))
        split_idx_2 = int(len(X_sequences) * (1 - test_size))
        
        X_train = X_sequences[:split_idx_1]
        y_train = y_sequences[:split_idx_1]
        X_val = X_sequences[split_idx_1:split_idx_2]
        y_val = y_sequences[split_idx_1:split_idx_2]
        X_test = X_sequences[split_idx_2:]
        y_test = y_sequences[split_idx_2:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        if self.model is None:
            # Build model with correct input shape
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if save_path:
            callbacks.append(
                ModelCheckpoint(
                    save_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        return history.history
    
    def predict(self, X: np.ndarray, inverse_transform: bool = True) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            inverse_transform: Whether to inverse transform predictions
            
        Returns:
            Predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X, verbose=0)
        
        if inverse_transform and self.scaler_y is not None:
            predictions = self.scaler_y.inverse_transform(predictions).flatten()
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        y_pred_scaled = self.model.predict(X_test, verbose=0).flatten()
        
        # Inverse transform
        y_test_actual = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        r2 = r2_score(y_test_actual, y_pred_actual)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred_actual,
            'actual': y_test_actual
        }
    
    def save_model(self, model_path: str, scaler_path: str = None) -> None:
        """
        Save the trained model and scalers.
        
        Args:
            model_path: Path to save the model
            scaler_path: Base path for saving scalers
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save model
        self.model.save(model_path)
        
        # Save scalers and feature columns
        if scaler_path:
            scaler_dir = os.path.dirname(scaler_path)
            os.makedirs(scaler_dir, exist_ok=True)
            
            joblib.dump(self.scaler_X, f"{scaler_path}_X.pkl")
            joblib.dump(self.scaler_y, f"{scaler_path}_y.pkl")
            joblib.dump(self.feature_cols, f"{scaler_path}_features.pkl")
    
    def load_model(self, model_path: str, scaler_path: str = None) -> None:
        """
        Load a trained model and scalers.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Base path for loading scalers
        """
        # Load model
        self.model = load_model(model_path)
        self.is_trained = True
        
        # Load scalers and feature columns
        if scaler_path:
            self.scaler_X = joblib.load(f"{scaler_path}_X.pkl")
            self.scaler_y = joblib.load(f"{scaler_path}_y.pkl")
            self.feature_cols = joblib.load(f"{scaler_path}_features.pkl")

def create_lstm_model(config: Dict[str, Any]) -> FloodLevelLSTM:
    """
    Factory function to create LSTM model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured LSTM model instance
    """
    return FloodLevelLSTM(
        sequence_length=config.get('sequence_length', 24),
        lstm_units=config.get('lstm_units', [64, 32, 16]),
        dropout_rate=config.get('dropout_rate', 0.2),
        learning_rate=config.get('learning_rate', 0.001)
    )
