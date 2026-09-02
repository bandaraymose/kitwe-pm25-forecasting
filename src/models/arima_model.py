"""
ARIMA model implementation for PM2.5 forecasting.

This module provides:
- ARIMA model with automatic parameter selection
- Grid search for optimal (p,d,q) parameters
- 7-day and 30-day forecasting with confidence intervals
- Model evaluation and comparison
"""

import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Dict, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    print("⚠️  statsmodels not available")


def arima_grid_search(train_data: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[Tuple[int, int, int], float]:
    """
    Grid search for optimal ARIMA(p,d,q) parameters.
    
    Args:
        train_data: Training time series
        max_p: Maximum AR order
        max_d: Maximum differencing order
        max_q: Maximum MA order
        
    Returns:
        Tuple of ((best_p, best_d, best_q), best_aic)
    """
    if not STATS_AVAILABLE:
        return ((1, 0, 1), float('inf'))
    
    print("Searching for optimal ARIMA parameters...")
    
    best_aic = float('inf')
    best_order = (1, 0, 1)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    results = model.fit(method='lbfgs', maxiter=100, disp=False)
                    
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                        
                except Exception:
                    continue
    
    print(f"   Best ARIMA{best_order} with AIC: {best_aic:.2f}")
    return best_order, best_aic


def train_arima_model(train_data: pd.Series, order: Tuple[int, int, int]) -> object:
    """
    Train ARIMA model with given order.
    
    Args:
        train_data: Training time series
        order: ARIMA order (p, d, q)
        
    Returns:
        Trained ARIMA model
    """
    print("Training ARIMA model...")
    
    if not STATS_AVAILABLE:
        return None
    
    try:
        model = ARIMA(train_data, order=order)
        results = model.fit()
        
        print("   ARIMA model trained successfully")
        
        return results
    except Exception as e:
        print(f"   Error training ARIMA model: {e}")
        return None


def evaluate_arima_model(model: object, test_data: pd.Series) -> Dict[str, float]:
    """
    Evaluate ARIMA model on test data.
    
    Args:
        model: Trained ARIMA model
        test_data: Test time series
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating ARIMA model...")
    
    if not STATS_AVAILABLE or model is None:
        return {}
    
    # Make predictions
    predictions = model.forecast(steps=len(test_data))
    
    # Calculate metrics
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, predictions)
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
    
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    return metrics


def generate_arima_forecasts(model: object, steps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate forecasts with confidence intervals.
    
    Args:
        model: Trained ARIMA model
        steps: Number of forecast steps
        
    Returns:
        Tuple of (forecast, confidence_intervals)
    """
    print(f"Generating {steps}-day ARIMA forecast...")
    
    # Generate forecast
    forecast_result = model.forecast(steps=steps, alpha=0.05)
    
    if hasattr(forecast_result, 'predicted_mean'):
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
    else:
        forecast = forecast_result
        conf_int = None
    
    print(f"   Generated {len(forecast)} forecasts")
    
    return forecast, conf_int


def arima_pipeline(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """
    Complete ARIMA modeling pipeline.
    
    Args:
        train_df: Training data
        val_df: Validation data  
        test_df: Test data
        
    Returns:
        Dictionary with results
    """
    print("Starting ARIMA modeling pipeline...")
    
    # Prepare data
    train_series = train_df['pm25_ug_m3']
    test_series = test_df['pm25_ug_m3']
    
    # Grid search for best parameters
    best_order, best_aic = arima_grid_search(train_series)
    
    # Train final model
    final_model = train_arima_model(train_series, best_order)
    
    if final_model is None:
        return {}
    
    # Evaluate model
    metrics = evaluate_arima_model(final_model, test_series)
    predictions = final_model.forecast(steps=len(test_series))
    
    # Generate forecasts
    forecast_7d, conf_int_7d = generate_arima_forecasts(final_model, steps=7)
    forecast_30d, conf_int_30d = generate_arima_forecasts(final_model, steps=30)
    
    results = {
        'model': final_model,
        'order': best_order,
        'aic': best_aic,
        'metrics': metrics,
        'predictions': predictions,
        'forecast_7d': forecast_7d,
        'conf_int_7d': conf_int_7d,
        'forecast_30d': forecast_30d,
        'conf_int_30d': conf_int_30d
    }
    
    return results


if __name__ == "__main__":
    # Load data
    try:
        train_df = pd.read_csv("d:/SCHOOL PROJECT - Copy/kitwe-pm25-forecasting/data/processed/train_2015_2024.csv")
        val_df = pd.read_csv("d:/SCHOOL PROJECT - Copy/kitwe-pm25-forecasting/data/processed/validation_2015_2024.csv") 
        test_df = pd.read_csv("d:/SCHOOL PROJECT - Copy/kitwe-pm25-forecasting/data/processed/test_2015_2024.csv")
        
        # Run ARIMA pipeline
        results = arima_pipeline(train_df, val_df, test_df)
        
        print("ARIMA modeling complete!")
        print(f"Test RMSE: {results['metrics']['rmse']:.4f}")
        
        # Save results
        model_path = Path("d:/SCHOOL PROJECT - Copy/kitwe-pm25-forecasting/models/arima_model.pkl")
        model_path.parent.mkdir(exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"Error: {e}")
    else:
        print("Data files not found")
