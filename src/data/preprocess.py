"""
Data quality control and preprocessing pipeline for PM2.5 time series.

This module provides functions for:
- Data validation and quality checks
- Outlier detection (Z-score and IQR methods)
- Missing value handling
- Basic statistics and summary
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks.
    
    Args:
        df: DataFrame with PM2.5 data
        
    Returns:
        Dictionary with quality metrics
    """
    print("🔍 Performing data quality validation...")
    
    quality_report = {
        'total_records': len(df),
        'date_range': (df['date'].min(), df['date'].max()),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_dates': df['date'].duplicated().sum(),
        'pm25_stats': {
            'mean': df['pm25_ug_m3'].mean(),
            'std': df['pm25_ug_m3'].std(),
            'min': df['pm25_ug_m3'].min(),
            'max': df['pm25_ug_m3'].max(),
            'negative_values': (df['pm25_ug_m3'] < 0).sum(),
            'zero_values': (df['pm25_ug_m3'] == 0).sum()
        },
        'source_breakdown': df['source'].value_counts().to_dict(),
        'frequency_breakdown': df['frequency'].value_counts().to_dict()
    }
    
    # Check for date gaps
    df_sorted = df.sort_values('date')
    expected_dates = pd.date_range(start=df_sorted['date'].min(), 
                                  end=df_sorted['date'].max(), freq='D')
    missing_dates = set(expected_dates) - set(df_sorted['date'])
    quality_report['missing_dates'] = len(missing_dates)
    
    # Print summary
    print(f"   📊 Total records: {quality_report['total_records']:,}")
    print(f"   📅 Date range: {quality_report['date_range'][0]} to {quality_report['date_range'][1]}")
    print(f"   ❌ Missing dates: {quality_report['missing_dates']}")
    print(f"   🔄 Duplicate dates: {quality_report['duplicate_dates']}")
    print(f"   📉 PM2.5 range: {quality_report['pm25_stats']['min']:.2f} to {quality_report['pm25_stats']['max']:.2f} µg/m³")
    print(f"   ⚠️  Negative PM2.5 values: {quality_report['pm25_stats']['negative_values']}")
    
    return quality_report


def detect_zscore_outliers(df: pd.DataFrame, threshold: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outliers using Z-score method.
    
    Args:
        df: DataFrame with PM2.5 data
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Tuple of (clean_data, outliers)
    """
    print(f"🎯 Detecting outliers with Z-score (threshold={threshold})...")
    
    # Calculate Z-scores
    df_clean = df.copy()
    df_clean['zscore'] = np.abs((df_clean['pm25_ug_m3'] - df_clean['pm25_ug_m3'].mean()) / df_clean['pm25_ug_m3'].std())
    
    # Identify outliers
    outliers = df_clean[df_clean['zscore'] > threshold].copy()
    clean_data = df_clean[df_clean['zscore'] <= threshold].copy()
    
    print(f"   🚨 Found {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}% of data)")
    print(f"   ✅ Clean data: {len(clean_data)} records")
    
    return clean_data.drop('zscore', axis=1), outliers.drop('zscore', axis=1)


def detect_iqr_outliers(df: pd.DataFrame, multiplier: float = 1.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        df: DataFrame with PM2.5 data
        multiplier: IQR multiplier for outlier detection
        
    Returns:
        Tuple of (clean_data, outliers)
    """
    print(f"📏 Detecting outliers with IQR method (multiplier={multiplier})...")
    
    # Calculate IQR
    Q1 = df['pm25_ug_m3'].quantile(0.25)
    Q3 = df['pm25_ug_m3'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Identify outliers
    outliers = df[(df['pm25_ug_m3'] < lower_bound) | (df['pm25_ug_m3'] > upper_bound)].copy()
    clean_data = df[(df['pm25_ug_m3'] >= lower_bound) & (df['pm25_ug_m3'] <= upper_bound)].copy()
    
    print(f"   📊 IQR bounds: {lower_bound:.2f} to {upper_bound:.2f} µg/m³")
    print(f"   🚨 Found {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}% of data)")
    print(f"   ✅ Clean data: {len(clean_data)} records")
    
    return clean_data, outliers


def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Handle missing values in PM2.5 data.
    
    Args:
        df: DataFrame with PM2.5 data
        method: Method for handling missing values ('interpolate', 'forward_fill', 'backward_fill')
        
    Returns:
        DataFrame with missing values handled
    """
    print(f"🔧 Handling missing values using {method} method...")
    
    missing_before = df['pm25_ug_m3'].isnull().sum()
    
    if missing_before == 0:
        print("   ✅ No missing values found")
        return df
    
    df_clean = df.copy()
    df_clean = df_clean.sort_values('date')
    
    if method == 'interpolate':
        df_clean['pm25_ug_m3'] = df_clean['pm25_ug_m3'].interpolate(method='linear')
    elif method == 'forward_fill':
        df_clean['pm25_ug_m3'] = df_clean['pm25_ug_m3'].fillna(method='ffill')
    elif method == 'backward_fill':
        df_clean['pm25_ug_m3'] = df_clean['pm25_ug_m3'].fillna(method='bfill')
    
    missing_after = df_clean['pm25_ug_m3'].isnull().sum()
    
    print(f"   📝 Missing values before: {missing_before}")
    print(f"   ✅ Missing values after: {missing_after}")
    
    return df_clean


def remove_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove or correct negative PM2.5 values.
    
    Args:
        df: DataFrame with PM2.5 data
        
    Returns:
        DataFrame with negative values handled
    """
    print("🚫 Removing negative PM2.5 values...")
    
    negative_count = (df['pm25_ug_m3'] < 0).sum()
    
    if negative_count == 0:
        print("   ✅ No negative values found")
        return df
    
    df_clean = df.copy()
    
    # Set negative values to small positive number (0.1)
    df_clean.loc[df_clean['pm25_ug_m3'] < 0, 'pm25_ug_m3'] = 0.1
    
    print(f"   🔄 Corrected {negative_count} negative values to 0.1 µg/m³")
    
    return df_clean


def basic_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run basic preprocessing pipeline on PM2.5 data.
    
    Args:
        df: Raw PM2.5 DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    print("🚀 Starting basic preprocessing pipeline...")
    
    # Step 1: Data quality validation
    quality_report = validate_data_quality(df)
    
    # Step 2: Remove negative values
    df_clean = remove_negative_values(df)
    
    # Step 3: Handle missing values
    df_clean = handle_missing_values(df_clean, method='interpolate')
    
    # Step 4: Outlier detection (using Z-score)
    df_clean, zscore_outliers = detect_zscore_outliers(df_clean, threshold=3.0)
    
    # Step 5: Outlier detection (using IQR)
    df_clean, iqr_outliers = detect_iqr_outliers(df_clean, multiplier=1.5)
    
    print(f"\n🎉 Preprocessing complete!")
    print(f"   📊 Original records: {len(df):,}")
    print(f"   📊 Final records: {len(df_clean):,}")
    print(f"   📊 Data retention: {len(df_clean)/len(df)*100:.1f}%")
    
    return df_clean


if __name__ == "__main__":
    # Test the preprocessing pipeline
    from pathlib import Path
    
    # Load the complete dataset
    data_path = Path("data/raw/pm25_kitwe_2015_2024_daily_complete.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Run preprocessing
        processed_df = basic_preprocessing_pipeline(df)
        
        # Save processed data
        output_path = Path("data/processed/pm25_cleaned.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        
        print(f"\n💾 Preprocessed data saved to: {output_path}")
    else:
        print(f"❌ Data file not found: {data_path}")
