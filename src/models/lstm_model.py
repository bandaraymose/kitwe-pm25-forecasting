"""
LSTM model implementation for PM2.5 forecasting.

This module provides:
- 2-layer LSTM architecture with 50 units each
- Sliding window approach (L=30)
- Dropout regularization (0.2)
- Adam optimizer with early stopping
- 7-day and 30-day forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not available")


def create_lstm_model(input_shape: Tuple[int, int], dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Create LSTM model with specified architecture.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled LSTM model
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    print("🧠 Creating LSTM model...")
    
    model = Sequential([
        # First LSTM layer
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        
        # Second LSTM layer
        LSTM(50, return_sequences=False),
        Dropout(dropout_rate),
        
        # Dense output layer
        Dense(1)
    ])
    
    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    print(f"   ✅ LSTM model created with input shape {input_shape}")
    print(f"   📊 Total parameters: {model.count_params():,}")
    
    return model


def prepare_lstm_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare and scale LSTM data.
    
    Args:
        X: Input sequences
        y: Target values
        
    Returns:
        Tuple of (scaled_X, scaled_y, scaler)
    """
    print("📊 Preparing LSTM data...")
    
    # Scale features
    scaler_X = StandardScaler()
    n_samples, n_timesteps, n_features = X.shape
    
    # Reshape for scaling
    X_reshaped = X.reshape(-1, n_features)
    X_scaled = scaler_X.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Scale target
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    print(f"   ✅ Data scaled - X: {X_scaled.shape}, y: {y_scaled.shape}")
    
    return X_scaled, y_scaled, scaler_X, scaler_y


def train_lstm_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100,
                   batch_size: int = 32) -> tf.keras.callbacks.History:
    """
    Train LSTM model with early stopping.
    
    Args:
        model: LSTM model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        epochs: Maximum number of epochs
        batch_size: Batch size
        
    Returns:
        Training history
    """
    if not TENSORFLOW_AVAILABLE or model is None:
        return None
    
    print("🚂 Training LSTM model...")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print(f"   ✅ Training completed - Best val loss: {min(history.history['val_loss']):.4f}")
    
    return history


def evaluate_lstm_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray,
                       scaler_y: StandardScaler) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate LSTM model on test data.
    
    Args:
        model: Trained LSTM model
        X_test: Test features
        y_test: Test targets
        scaler_y: Target scaler
        
    Returns:
        Tuple of (metrics, predictions)
    """
    if not TENSORFLOW_AVAILABLE or model is None:
        return {}, np.array([])
    
    print("📈 Evaluating LSTM model...")
    
    # Make predictions
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse scaling
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }
    
    print(f"   📊 RMSE: {rmse:.4f}")
    print(f"   📊 MAE: {mae:.4f}")
    print(f"   📊 MAPE: {mape:.2f}%")
    
    return metrics, y_pred


def generate_lstm_forecasts(model: tf.keras.Model, last_sequence: np.ndarray,
                           scaler_X: StandardScaler, scaler_y: StandardScaler,
                           steps: int = 30) -> np.ndarray:
    """
    Generate multi-step forecasts using LSTM.
    
    Args:
        model: Trained LSTM model
        last_sequence: Last known sequence
        scaler_X: Feature scaler
        scaler_y: Target scaler
        steps: Number of steps to forecast
        
    Returns:
        Forecast array
    """
    if not TENSORFLOW_AVAILABLE or model is None:
        return np.array([])
    
    print(f"🔮 Generating {steps}-day LSTM forecast...")
    
    forecasts = []
    current_sequence = last_sequence.copy()
    
    for step in range(steps):
        # Scale the sequence
        current_sequence_scaled = scaler_X.transform(current_sequence.reshape(-1, current_sequence.shape[-1]))
        current_sequence_scaled = current_sequence_scaled.reshape(1, current_sequence.shape[0], current_sequence.shape[-1])
        
        # Make prediction
        pred_scaled = model.predict(current_sequence_scaled, verbose=0)
        
        # Inverse scaling
        pred = scaler_y.inverse_transform(pred_scaled).flatten()[0]
        forecasts.append(pred)
        
        # Update sequence for next prediction
        # This is a simplified approach - in practice, you'd need to update all features
        # For now, we'll just shift and add the predicted value
        new_row = current_sequence[-1].copy()
        new_row[0] = pred  # Assuming PM2.5 is the first feature
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    forecasts = np.array(forecasts)
    print(f"   ✅ Generated {len(forecasts)} forecasts")
    
    return forecasts


def lstm_pipeline() -> Dict:
    """
    Complete LSTM modeling pipeline using pre-processed sequences.
    
    Returns:
        Dictionary with results
    """
    print("🚀 Starting LSTM modeling pipeline...")
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available")
        return {}
    
    from pathlib import Path
    
    # Load pre-processed sequences
    data_dir = Path("data/processed")
    
    try:
        X_train = np.load(data_dir / "X_train.npy")
        y_train = np.load(data_dir / "y_train.npy")
        X_val = np.load(data_dir / "X_validation.npy")
        y_val = np.load(data_dir / "y_validation.npy")
        X_test = np.load(data_dir / "X_test.npy")
        y_test = np.load(data_dir / "y_test.npy")
        
        print(f"   📊 Loaded sequences - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    except FileNotFoundError:
        print("❌ Sequence files not found")
        return {}
    
    # Prepare data
    X_train_scaled, y_train_scaled, scaler_X, scaler_y = prepare_lstm_data(X_train, y_train)
    X_val_scaled, y_val_scaled, _, _ = prepare_lstm_data(X_val, y_val)
    X_test_scaled, y_test_scaled, _, _ = prepare_lstm_data(X_test, y_test)
    
    # Create model
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    model = create_lstm_model(input_shape, dropout_rate=0.2)
    
    # Train model
    history = train_lstm_model(model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
    
    # Evaluate model
    metrics, predictions = evaluate_lstm_model(model, X_test_scaled, y_test_scaled, scaler_y)
    
    # Generate forecasts
    last_sequence = X_test_scaled[-1]  # Use last test sequence
    forecast_7d = generate_lstm_forecasts(model, last_sequence, scaler_X, scaler_y, steps=7)
    forecast_30d = generate_lstm_forecasts(model, last_sequence, scaler_X, scaler_y, steps=30)
    
    results = {
        'model': model,
        'history': history,
        'metrics': metrics,
        'predictions': predictions,
        'forecast_7d': forecast_7d,
        'forecast_30d': forecast_30d,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    import pickle
    
    # Run LSTM pipeline
    results = lstm_pipeline()
    
    if results:
        print("\n🎉 LSTM modeling complete!")
        print(f"📊 Test RMSE: {results['metrics']['rmse']:.4f}")
        
        # Save results
        model_path = Path("models/lstm_model.h5")
        results_path = Path("models/lstm_results.pkl")
        
        model_path.parent.mkdir(exist_ok=True)
        
        # Save model
        results['model'].save(model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # Save other results (excluding the model itself)
        results_to_save = {k: v for k, v in results.items() if k != 'model'}
        with open(results_path, 'wb') as f:
            pickle.dump(results_to_save, f)
        print(f"💾 Results saved to: {results_path}")
    else:
        print("❌ LSTM modeling failed")
