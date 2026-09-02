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
from typing import Tuple, Dict, List
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
        return ((1, 1), float('inf'))
    
    print("🔍 Searching for optimal ARIMA parameters...")
    
    best_aic = float('inf')
    best_order = (1, 0, 1)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    results = model.fit()
                    
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                        
                except Exception:
                    continue
    
    print(f"   ✅ Best ARIMA{best_order} with AIC: {best_aic:.2f}")
    return best_order, best_aic


def train_arima_model(train_data: pd.Series, order: Tuple[int, int, int]) -> object:
    """
    Train ARIMA model with given order.
    
    Args:
        train_data: Training time series
        order: ARIMA order (p, d, q)
        
    Returns:
       print("🔧 Training ARIMA model...")
    
    if not STATS_AVAILABLE:
        return None
    
    try:
        model = ARIMA(train_data, order=order)
        results = model.fit()
        
        print("   ✅ ARIMA model trained successfully")
        
        return results
    except Exception as e:
        print(f"   ❌ Error training ARIMA model: {e}")
        return None


def evaluate_arma_model(model: object, test_data: pd.Series) -> Dict[str, float]:
    """
    Evaluate ARMA model on test data.
    
    Args:
        model: Trained ARMA model
        test_data: Test time series
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not STATS_AVAILABLE or model is None:
        return {}
    
    print("📈 Evaluating ARMA model...")
    
    # Make predictions
    predictions = model.forecast(steps=len(test_data))
    
    # Calculate metrics
    mse = mean_squared_error(test_data, predictions)
    mae = mean_absolute_error(test_data, predictions)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }
    
    print(f"   📊 RMSE: {rmse:.4f}")
    print(f"   📊 MAE: {mae:.4f}")
    print(f"   📊 MAPE: {mape:.2f}%")
    
    return metrics, predictions


def generate_arma_forecasts(model: object, steps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate forecasts with confidence intervals.
    
    Args:
        model: Trained ARMA model
        steps: Number of steps to forecast
        
    Returns:
        Tuple of (forecast, confidence_intervals)
    """
    if not STATS_AVAILABLE or model is None:
        return np.array([]), np.array([])
    
    print(f"🔮 Generating {steps}-day ARMA forecast...")
    
    # Generate forecast
    forecast_result = model.forecast(steps=steps, alpha=0.05)
    
    if hasattr(forecast_result, 'predicted_mean'):
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
    else:
        forecast = forecast_result
        conf_int = None
    
    print(f"   ✅ Generated {len(forecast)} forecasts")
    
    return forecast, conf_int


def arma_pipeline(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """
    Complete ARMA modeling pipeline.
    
    Args:
        train_df: Training data
        val_df: Validation data  
        test_df: Test data
        
    Returns:
        Dictionary with results
    """
    print("🚀 Starting ARMA modeling pipeline...")
    
    # Extract time series
    train_series = train_df.set_index('date')['pm25_ug_m3']
    val_series = val_df.set_index('date')['pm25_ug_m3']
    test_series = test_df.set_index('date')['pm25_ug_m3']
    
    # Combine train+val for final training
    full_train_series = pd.concat([train_series, val_series])
    
    # Grid search for best parameters
    best_order, best_aic = arma_grid_search(train_series, max_p=5, max_q=5)
    
    # Train model on full training data
    final_model = train_arma_model(full_train_series, best_order)
    
    # Evaluate on test set
    metrics, predictions = evaluate_arma_model(final_model, test_series)
    
    # Generate forecasts
    forecast_7d, conf_int_7d = generate_arma_forecasts(final_model, steps=7)
    forecast_30d, conf_int_30d = generate_arma_forecasts(final_model, steps=30)
    
    results = {
        'model': final_model,
        'order': best_order,
        'aic': best_aic,
        'metrics': metrics,
        'predictions': predictions,
        'forecast_7d': forecast_7d,
        'forecast_30d': forecast_30d,
        'conf_int_7d': conf_int_7d,
        'conf_int_30d': conf_int_30d
    }
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    
    # Load data splits
    train_path = Path("data/processed/train_2015_2024.csv")
    val_path = Path("data/processed/validation_2015_2024.csv")
    test_path = Path("data/processed/test_2015_2024.csv")
    
    if all(p.exists() for p in [train_path, val_path, test_path]):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # Run ARMA pipeline
        results = arma_pipeline(train_df, val_df, test_df)
        
        print("\n🎉 ARMA modeling complete!")
        print(f"📊 Best order: ARMA{results['order']}")
        print(f"📊 Test RMSE: {results['metrics']['rmse']:.4f}")
        
        # Save results
        import pickle
        model_path = Path("models/arma_model.pkl")
        model_path.parent.mkdir(exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 Model saved to: {model_path}")
    else:
        print("❌ Data files not found")
