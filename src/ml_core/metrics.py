"""
Evaluation Metrics Module for Jakarta FloodNet
=============================================

This module contains functions for evaluating model performance
including various regression metrics and visualization utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error,
    r2_score,
    explained_variance_score
)
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing various metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
    
    # MAPE (handle zero values)
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        metrics['mape'] = mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask])
    else:
        metrics['mape'] = float('inf')
    
    # Additional metrics
    residuals = y_true - y_pred
    metrics['mean_residual'] = np.mean(residuals)
    metrics['std_residual'] = np.std(residuals)
    metrics['max_error'] = np.max(np.abs(residuals))
    metrics['median_absolute_error'] = np.median(np.abs(residuals))
    
    # Normalized metrics
    y_range = np.max(y_true) - np.min(y_true)
    if y_range > 0:
        metrics['normalized_rmse'] = metrics['rmse'] / y_range
        metrics['normalized_mae'] = metrics['mae'] / y_range
    else:
        metrics['normalized_rmse'] = 0
        metrics['normalized_mae'] = 0
    
    return metrics

def calculate_flood_specific_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    flood_threshold: float = 80.0
) -> Dict[str, float]:
    """
    Calculate flood-specific evaluation metrics.
    
    Args:
        y_true: True water levels
        y_pred: Predicted water levels
        flood_threshold: Water level threshold to consider as flood
        
    Returns:
        Dictionary with flood-specific metrics
    """
    # Classify as flood/no-flood
    y_true_flood = (y_true >= flood_threshold).astype(int)
    y_pred_flood = (y_pred >= flood_threshold).astype(int)
    
    # Confusion matrix elements
    tp = np.sum((y_true_flood == 1) & (y_pred_flood == 1))
    tn = np.sum((y_true_flood == 0) & (y_pred_flood == 0))
    fp = np.sum((y_true_flood == 0) & (y_pred_flood == 1))
    fn = np.sum((y_true_flood == 1) & (y_pred_flood == 0))
    
    metrics = {}
    
    # Classification metrics
    if tp + fp > 0:
        metrics['precision'] = tp / (tp + fp)
    else:
        metrics['precision'] = 0
        
    if tp + fn > 0:
        metrics['recall'] = tp / (tp + fn)
    else:
        metrics['recall'] = 0
        
    if tp + tn + fp + fn > 0:
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    else:
        metrics['accuracy'] = 0
        
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0
    
    # Flood event statistics
    metrics['true_flood_events'] = np.sum(y_true_flood)
    metrics['pred_flood_events'] = np.sum(y_pred_flood)
    metrics['false_alarms'] = fp
    metrics['missed_floods'] = fn
    
    return metrics

def evaluate_model_performance(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    flood_threshold: float = 80.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        flood_threshold: Threshold for flood classification
        verbose: Whether to print results
        
    Returns:
        Dictionary with all evaluation metrics
    """
    results = {}
    
    # Regression metrics
    results['regression'] = calculate_regression_metrics(y_true, y_pred)
    
    # Flood-specific metrics
    results['flood_classification'] = calculate_flood_specific_metrics(
        y_true, y_pred, flood_threshold
    )
    
    if verbose:
        print("="*60)
        print("MODEL PERFORMANCE EVALUATION")
        print("="*60)
        
        print("\nRegression Metrics:")
        print(f"  MSE:                    {results['regression']['mse']:.4f}")
        print(f"  RMSE:                   {results['regression']['rmse']:.4f}")
        print(f"  MAE:                    {results['regression']['mae']:.4f}")
        print(f"  R²:                     {results['regression']['r2']:.4f}")
        print(f"  MAPE:                   {results['regression']['mape']:.4f}%")
        print(f"  Explained Variance:     {results['regression']['explained_variance']:.4f}")
        print(f"  Max Error:              {results['regression']['max_error']:.4f}")
        print(f"  Normalized RMSE:        {results['regression']['normalized_rmse']:.4f}")
        
        print(f"\nFlood Classification Metrics (threshold={flood_threshold}):")
        print(f"  Precision:              {results['flood_classification']['precision']:.4f}")
        print(f"  Recall:                 {results['flood_classification']['recall']:.4f}")
        print(f"  F1-Score:               {results['flood_classification']['f1_score']:.4f}")
        print(f"  Accuracy:               {results['flood_classification']['accuracy']:.4f}")
        print(f"  True Flood Events:      {results['flood_classification']['true_flood_events']}")
        print(f"  Predicted Flood Events: {results['flood_classification']['pred_flood_events']}")
        print(f"  False Alarms:           {results['flood_classification']['false_alarms']}")
        print(f"  Missed Floods:          {results['flood_classification']['missed_floods']}")
    
    return results

def plot_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    timestamps: Optional[pd.Series] = None,
    title: str = "Model Predictions",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive prediction visualization plots.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        timestamps: Optional timestamps for time series plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 1. Scatter plot: Predictions vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Water Level (cm)')
    axes[0, 0].set_ylabel('Predicted Water Level (cm)')
    axes[0, 0].set_title('Predictions vs Actual Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Water Level (cm)')
    axes[0, 1].set_ylabel('Residuals (cm)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Time series plot
    if timestamps is not None:
        # Plot subset of data if too large
        plot_size = min(200, len(y_true))
        indices = np.linspace(0, len(y_true)-1, plot_size).astype(int)
        
        axes[1, 0].plot(timestamps.iloc[indices], y_true[indices], 'b-', label='Actual', linewidth=1.5)
        axes[1, 0].plot(timestamps.iloc[indices], y_pred[indices], 'r-', label='Predicted', linewidth=1.5)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Water Level (cm)')
        axes[1, 0].set_title(f'Time Series Comparison (Sample of {plot_size} points)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    else:
        # Plot without timestamps
        plot_size = min(200, len(y_true))
        indices = range(plot_size)
        axes[1, 0].plot(indices, y_true[:plot_size], 'b-', label='Actual', linewidth=1.5)
        axes[1, 0].plot(indices, y_pred[:plot_size], 'r-', label='Predicted', linewidth=1.5)
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Water Level (cm)')
        axes[1, 0].set_title(f'Time Series Comparison (First {plot_size} points)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error distribution histogram
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Residuals (cm)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(residuals)
    std_error = np.std(residuals)
    axes[1, 1].text(0.05, 0.95, f'Mean: {mean_error:.2f}\nStd: {std_error:.2f}', 
                    transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        figsize: Figure size
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    axes[0].plot(history['loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss During Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    if 'mae' in history:
        axes[1].plot(history['mae'], label='Training MAE')
        axes[1].plot(history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE During Training')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()

def create_feature_importance_plot(
    feature_importance: Dict[str, float],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create feature importance plot.
    
    Args:
        feature_importance: Dictionary with feature names and importance scores
        top_n: Number of top features to display
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importance = zip(*top_features)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(features)), importance)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    plt.show()

def compare_models(
    model_results: Dict[str, Dict[str, Any]], 
    metric: str = 'r2',
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Compare multiple models performance.
    
    Args:
        model_results: Dictionary with model names and their evaluation results
        metric: Metric to compare
        figsize: Figure size
    """
    model_names = list(model_results.keys())
    values = [model_results[name]['regression'][metric] for name in model_names]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(model_names, values)
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_model_summary(
    model_results: Dict[str, Any],
    model_name: str = "LSTM Model"
) -> None:
    """
    Print a formatted summary of model performance.
    
    Args:
        model_results: Model evaluation results
        model_name: Name of the model
    """
    print("="*70)
    print(f"{model_name.upper()} - PERFORMANCE SUMMARY")
    print("="*70)
    
    reg_metrics = model_results['regression']
    flood_metrics = model_results['flood_classification']
    
    print(f"\n📊 REGRESSION PERFORMANCE:")
    print(f"   • R² Score:           {reg_metrics['r2']:.4f}")
    print(f"   • RMSE:               {reg_metrics['rmse']:.2f} cm")
    print(f"   • MAE:                {reg_metrics['mae']:.2f} cm")
    print(f"   • MAPE:               {reg_metrics['mape']:.2f}%")
    
    print(f"\n🌊 FLOOD PREDICTION PERFORMANCE:")
    print(f"   • Precision:          {flood_metrics['precision']:.4f}")
    print(f"   • Recall:             {flood_metrics['recall']:.4f}")
    print(f"   • F1-Score:           {flood_metrics['f1_score']:.4f}")
    print(f"   • Accuracy:           {flood_metrics['accuracy']:.4f}")
    
    print(f"\n📈 FLOOD EVENT STATISTICS:")
    print(f"   • True flood events:  {flood_metrics['true_flood_events']}")
    print(f"   • Detected floods:    {flood_metrics['pred_flood_events']}")
    print(f"   • False alarms:       {flood_metrics['false_alarms']}")
    print(f"   • Missed floods:      {flood_metrics['missed_floods']}")
    
    print("="*70)
