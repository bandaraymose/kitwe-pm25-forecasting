#!/usr/bin/env python3
"""
Combine yearly daily PM2.5 files into one complete dataset.
"""

import pandas as pd
from pathlib import Path

def combine_daily_files():
    raw_dir = Path("data/raw")
    
    # List of daily files
    daily_files = [
        "pm25_kitwe_2017_daily.csv",
        "pm25_kitwe_2018_daily.csv", 
        "pm25_kitwe_2019_daily.csv",
        "pm25_kitwe_2020_daily.csv",
        "pm25_kitwe_2021_daily.csv",
        "pm25_kitwe_2022_daily.csv"
    ]
    
    # Read and combine all files
    all_data = []
    for file in daily_files:
        file_path = raw_dir / file
        if file_path.exists():
            print(f"Reading {file}...")
            df = pd.read_csv(file_path)
            all_data.append(df)
        else:
            print(f"Warning: {file} not found")
    
    if not all_data:
        print("No daily files found!")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df = combined_df.sort_values('date')
    
    # Save combined file
    output_file = raw_dir / "pm25_kitwe_2017_2022_daily_combined.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Combined dataset saved to: {output_file}")
    print(f"📊 Total records: {len(combined_df)}")
    print(f"📅 Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"📈 PM2.5 range: {combined_df['pm25_ug_m3'].min():.2f} to {combined_df['pm25_ug_m3'].max():.2f} µg/m³")
    
    # Show sample data
    print(f"\n📋 Sample data:")
    print(combined_df.head())

if __name__ == "__main__":
    combine_daily_files()
