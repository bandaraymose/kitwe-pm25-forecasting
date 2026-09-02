"""
Feature engineering for PM2.5 time series forecasting.

This module provides functions for:
- Seasonal feature extraction (day of year, month, season)
- Temporal features (trend, lag features)
- Cyclical encoding for seasonal patterns
- Stationarity testing and transformations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import adfuller
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    print("⚠️  statsmodels not available - stationarity tests disabled")


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create seasonal features for time series forecasting.
    
    Args:
        df: DataFrame with date column
        
    Returns:
        DataFrame with seasonal features added
    """
    print("🌸 Creating seasonal features...")
    
    df_features = df.copy()
    df_features['date'] = pd.to_datetime(df_features['date'])
    
    # Basic temporal features
    df_features['year'] = df_features['date'].dt.year
    df_features['month'] = df_features['date'].dt.month
    df_features['day'] = df_features['date'].dt.day
    df_features['dayofweek'] = df_features['date'].dt.dayofweek
    df_features['dayofyear'] = df_features['date'].dt.dayofyear
    df_features['week'] = df_features['date'].dt.isocalendar().week
    df_features['quarter'] = df_features['date'].dt.quarter
    
    # Season mapping (Southern Hemisphere - Zambia)
    def get_southern_season(month):
        if month in [12, 1, 2]:
            return 'summer'  # Hot, rainy season
        elif month in [3, 4, 5]:
            return 'autumn'  # Transition to dry season
        elif month in [6, 7, 8]:
            return 'winter'  # Dry, cool season
        else:
            return 'spring'  # Transition to rainy season
    
    df_features['season'] = df_features['month'].apply(get_southern_season)
    
    # Cyclical encoding for seasonal patterns
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day'] / 31)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day'] / 31)
    df_features['dayofyear_sin'] = np.sin(2 * np.pi * df_features['dayofyear'] / 365.25)
    df_features['dayofyear_cos'] = np.cos(2 * np.pi * df_features['dayofyear'] / 365.25)
    
    # Season indicators (one-hot encoding)
    seasons = ['summer', 'autumn', 'winter', 'spring']
    for season in seasons:
        df_features[f'season_{season}'] = (df_features['season'] == season).astype(int)
    
    print(f"   ✅ Added {len([col for col in df_features.columns if col not in df.columns])} seasonal features")
    
    return df_features


def create_temporal_features(df: pd.DataFrame, target_col: str = 'pm25_ug_m3') -> pd.DataFrame:
    """
    Create temporal features including trends and lag features.
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        
    Returns:
        DataFrame with temporal features added
    """
    print("⏰ Creating temporal features...")
    
    df_features = df.copy()
    df_features = df_features.sort_values('date')
    
    # Trend features
    df_features['time_index'] = np.arange(len(df_features))
    df_features['time_index_sqrt'] = np.sqrt(df_features['time_index'])
    df_features['time_index_log'] = np.log1p(df_features['time_index'])
    
    # Lag features (past values)
    lag_periods = [1, 7, 14, 30, 90]  # 1 day, 1 week, 2 weeks, 1 month, 3 months
    
    for lag in lag_periods:
        df_features[f'pm25_lag_{lag}'] = df_features[target_col].shift(lag)
    
    # Rolling statistics
    rolling_windows = [7, 14, 30, 90]  # 1 week, 2 weeks, 1 month, 3 months
    
    for window in rolling_windows:
        df_features[f'pm25_rolling_mean_{window}'] = df_features[target_col].rolling(window=window).mean()
        df_features[f'pm25_rolling_std_{window}'] = df_features[target_col].rolling(window=window).std()
        df_features[f'pm25_rolling_min_{window}'] = df_features[target_col].rolling(window=window).min()
        df_features[f'pm25_rolling_max_{window}'] = df_features[target_col].rolling(window=window).max()
    
    # Difference features (change over time)
    df_features['pm25_diff_1'] = df_features[target_col].diff(1)
    df_features['pm25_diff_7'] = df_features[target_col].diff(7)
    df_features['pm25_diff_30'] = df_features[target_col].diff(30)
    
    # Percentage change
    df_features['pm25_pct_change_1'] = df_features[target_col].pct_change(1)
    df_features['pm25_pct_change_7'] = df_features[target_col].pct_change(7)
    
    print(f"   ✅ Added temporal features including lags and rolling statistics")
    
    return df_features


def test_stationarity(df: pd.DataFrame, target_col: str = 'pm25_ug_m3') -> Dict[str, any]:
    """
    Test stationarity using Augmented Dickey-Fuller test.
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        
    Returns:
        Dictionary with stationarity test results
    """
    if not STATS_AVAILABLE:
        print("⚠️  Stationarity test skipped (statsmodels not available)")
        return {}
    
    print("📊 Testing stationarity with ADF test...")
    
    # Remove NaN values for the test
    series = df[target_col].dropna()
    
    # Perform ADF test
    result = adfuller(series)
    
    adf_stats = {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05,
        'interpretation': 'Stationary' if result[1] <= 0.05 else 'Non-stationary'
    }
    
    print(f"   📈 ADF Statistic: {adf_stats['adf_statistic']:.4f}")
    print(f"   📊 p-value: {adf_stats['p_value']:.4f}")
    print(f"   🎯 Result: {adf_stats['interpretation']}")
    
    return adf_stats


def apply_differencing(df: pd.DataFrame, target_col: str = 'pm25_ug_m3') -> pd.DataFrame:
    """
    Apply differencing to make series stationary.
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        
    Returns:
        DataFrame with differenced series
    """
    print("🔄 Applying differencing to achieve stationarity...")
    
    df_diff = df.copy()
    df_diff = df_diff.sort_values('date')
    
    # First difference
    df_diff[f'{target_col}_diff1'] = df_diff[target_col].diff()
    
    # Second difference if needed
    df_diff[f'{target_col}_diff2'] = df_diff[f'{target_col}_diff1'].diff()
    
    # Seasonal differencing (7-day and 30-day)
    df_diff[f'{target_col}_seasonal_diff_7'] = df_diff[target_col].diff(7)
    df_diff[f'{target_col}_seasonal_diff_30'] = df_diff[target_col].diff(30)
    
    print("   ✅ Applied differencing transformations")
    
    return df_diff


def create_feature_engineering_pipeline(df: pd.DataFrame, target_col: str = 'pm25_ug_m3') -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Cleaned DataFrame
        target_col: Target column name
        
    Returns:
        DataFrame with all features engineered
    """
    print("🚀 Starting feature engineering pipeline...")
    
    # Step 1: Create seasonal features
    df_features = create_seasonal_features(df)
    
    # Step 2: Create temporal features
    df_features = create_temporal_features(df_features, target_col)
    
    # Step 3: Test stationarity
    stationarity_results = test_stationarity(df_features, target_col)
    
    # Step 4: Apply differencing if needed
    if stationarity_results and not stationarity_results.get('is_stationary', True):
        df_features = apply_differencing(df_features, target_col)
    
    # Step 5: Remove rows with NaN values (created by lag features)
    initial_rows = len(df_features)
    df_features = df_features.dropna()
    final_rows = len(df_features)
    
    print(f"\n🎉 Feature engineering complete!")
    print(f"   📊 Initial features: {len(df.columns)}")
    print(f"   📊 Final features: {len(df_features.columns)}")
    print(f"   📊 Data retention: {final_rows/initial_rows*100:.1f}%")
    print(f"   📊 Final records: {final_rows:,}")
    
    return df_features


if __name__ == "__main__":
    # Test the feature engineering pipeline
    from pathlib import Path
    
    # Load cleaned data
    data_path = Path("data/processed/pm25_cleaned.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Run feature engineering
        features_df = create_feature_engineering_pipeline(df)
        
        # Save engineered features
        output_path = Path("data/processed/pm25_features.csv")
        features_df.to_csv(output_path, index=False)
        
        print(f"\n💾 Feature-engineered data saved to: {output_path}")
        
        # Show feature list
        print(f"\n📋 Engineered features:")
        for i, col in enumerate(features_df.columns):
            if col not in df.columns:
                print(f"   {i+1}. {col}")
    else:
        print(f"❌ Cleaned data file not found: {data_path}")
