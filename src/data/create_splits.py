"""
Data splitting for PM2.5 time series forecasting.

This module creates train/validation/test splits according to the project specification:
- Training: 2015-2022 (8 years)
- Validation: 2023 (1 year)  
- Test: 2024 (1 year)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def create_time_series_splits(df: pd.DataFrame, target_col: str = 'pm25_ug_m3') -> Dict[str, pd.DataFrame]:
    """
    Create time-based train/validation/test splits.
    
    Args:
        df: Feature-engineered DataFrame
        target_col: Target column name
        
    Returns:
        Dictionary with train, validation, and test DataFrames
    """
    print("📅 Creating time-based train/validation/test splits...")
    
    # Ensure date column is datetime
    df_sorted = df.copy()
    df_sorted['date'] = pd.to_datetime(df_sorted['date'])
    df_sorted = df_sorted.sort_values('date')
    
    # Define split dates
    train_end = '2022-12-31'
    val_end = '2023-12-31'
    test_end = '2024-12-31'
    
    # Create splits
    train_df = df_sorted[df_sorted['date'] <= train_end].copy()
    val_df = df_sorted[(df_sorted['date'] > train_end) & (df_sorted['date'] <= val_end)].copy()
    test_df = df_sorted[(df_sorted['date'] > val_end) & (df_sorted['date'] <= test_end)].copy()
    
    # Print split information
    print(f"   🚂 Training set: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} records)")
    print(f"   ✅ Validation set: {val_df['date'].min()} to {val_df['date'].max()} ({len(val_df)} records)")
    print(f"   🧪 Test set: {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} records)")
    
    # Check for data gaps
    total_records = len(train_df) + len(val_df) + len(test_df)
    if total_records != len(df_sorted):
        print(f"   ⚠️  Warning: {len(df_sorted) - total_records} records excluded due to missing dates")
    
    # Target variable statistics for each split
    print(f"\n📊 Target variable (PM2.5) statistics:")
    for name, split_df in [('Training', train_df), ('Validation', val_df), ('Test', test_df)]:
        stats = split_df[target_col].describe()
        print(f"   {name}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}")
    
    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df
    }


def save_data_splits(splits: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """
    Save data splits to separate CSV files.
    
    Args:
        splits: Dictionary with train, validation, test DataFrames
        output_dir: Directory to save the splits
    """
    print(f"💾 Saving data splits to {output_dir}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each split
    for split_name, split_df in splits.items():
        output_path = output_dir / f"{split_name}_2015_2024.csv"
        split_df.to_csv(output_path, index=False)
        print(f"   ✅ {split_name.capitalize()} set: {output_path}")
    
    # Save split information
    info_path = output_dir / "data_split_info.txt"
    with open(info_path, 'w') as f:
        f.write("PM2.5 Time Series Data Split Information\n")
        f.write("=" * 50 + "\n\n")
        
        for split_name, split_df in splits.items():
            f.write(f"{split_name.upper()} SET:\n")
            f.write(f"  Records: {len(split_df):,}\n")
            f.write(f"  Date range: {split_df['date'].min()} to {split_df['date'].max()}\n")
            f.write(f"  PM2.5 mean: {split_df['pm25_ug_m3'].mean():.2f} µg/m³\n")
            f.write(f"  PM2.5 std: {split_df['pm25_ug_m3'].std():.2f} µg/m³\n")
            f.write(f"  PM2.5 range: {split_df['pm25_ug_m3'].min():.2f} - {split_df['pm25_ug_m3'].max():.2f} µg/m³\n\n")
        
        f.write("SPLIT RATIO:\n")
        total = sum(len(df) for df in splits.values())
        for split_name, split_df in splits.items():
            percentage = len(split_df) / total * 100
            f.write(f"  {split_name.capitalize()}: {percentage:.1f}% ({len(split_df):,} records)\n")
    
    print(f"   📋 Split information saved to: {info_path}")


def create_sequences_for_lstm(df: pd.DataFrame, target_col: str = 'pm25_ug_m3', 
                            sequence_length: int = 30, feature_cols: list = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        df: DataFrame with features
        target_col: Target column name
        sequence_length: Length of input sequences
        feature_cols: List of feature columns to use
        
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    print(f"🔄 Creating LSTM sequences (length={sequence_length})...")
    
    if feature_cols is None:
        # Use all numeric columns except date and target
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != target_col]
    
    # Sort by date
    df_sorted = df.sort_values('date')
    
    # Extract features and target
    features = df_sorted[feature_cols].values
    target = df_sorted[target_col].values
    
    # Create sequences
    X, y = [], []
    
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(target[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   ✅ Created {len(X)} sequences")
    print(f"   📊 Input shape: {X.shape}")
    print(f"   📊 Target shape: {y.shape}")
    
    return X, y


def create_all_splits_and_sequences() -> None:
    """
    Main function to create all data splits and sequences.
    """
    print("🚀 Starting data split and sequence creation...")
    
    # Load feature-engineered data
    data_path = Path("data/processed/pm25_features.csv")
    if not data_path.exists():
        print(f"❌ Feature data not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create time-based splits
    splits = create_time_series_splits(df)
    
    # Save splits
    output_dir = Path("data/processed")
    save_data_splits(splits, output_dir)
    
    # Create LSTM sequences for each split
    print(f"\n🧠 Creating LSTM sequences...")
    
    # Define feature columns (exclude date and target)
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['pm25_ug_m3', 'longitude', 'latitude']]
    
    for split_name, split_df in splits.items():
        print(f"\n   📊 Creating sequences for {split_name} set...")
        X, y = create_sequences_for_lstm(split_df, feature_cols=feature_cols)
        
        # Save sequences as numpy arrays
        np.save(output_dir / f"X_{split_name}.npy", X)
        np.save(output_dir / f"y_{split_name}.npy", y)
        
        print(f"   💾 Saved sequences for {split_name} set")
    
    print(f"\n🎉 All data splits and sequences created successfully!")
    print(f"📁 Files saved in: {output_dir}")


if __name__ == "__main__":
    create_all_splits_and_sequences()
