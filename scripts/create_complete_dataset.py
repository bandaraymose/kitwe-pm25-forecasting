#!/usr/bin/env python3
"""
Create complete PM2.5 dataset combining daily (2017-2022) with monthly (2015-2016) data.
For 2023-2024, we'll use temporal extrapolation from available data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_complete_dataset():
    raw_dir = Path("data/raw")
    
    # Load daily data (2017-2022)
    daily_file = raw_dir / "pm25_kitwe_2017_2022_daily_combined.csv"
    if daily_file.exists():
        print(" Loading daily data (2017-2022)...")
        daily_df = pd.read_csv(daily_file)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.sort_values('date')
        print(f"    {len(daily_df)} daily records loaded")
    else:
        print("❌ Daily file not found!")
        return
    
    # Load monthly data (2015-2016)
    monthly_file = raw_dir / "pm25_kitwe_2015_2016_monthly.csv"
    if monthly_file.exists():
        print(" Loading monthly data (2015-2016)...")
        monthly_df = pd.read_csv(monthly_file)
        monthly_df['date'] = pd.to_datetime(monthly_df['date'])
        monthly_df = monthly_df.sort_values('date')
        print(f"    {len(monthly_df)} monthly records loaded")
    else:
        print(" Monthly file not found!")
        return
    
    # Convert monthly to daily by interpolation
    print("Converting monthly data to daily...")
    daily_2015_2016 = []
    
    for _, month_row in monthly_df.iterrows():
        month_start = month_row['date']
        year, month = month_start.year, month_start.month
        
        # Get days in month
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        
        days_in_month = (next_month - month_start).days
        
        # Create daily values with seasonal variation
        base_pm25 = month_row['pm25_ug_m3']
        
        for day in range(1, days_in_month + 1):
            current_date = datetime(year, month, day)
            
            # Add some daily variation (±20% around monthly mean)
            daily_variation = 1 + 0.2 * np.sin(2 * np.pi * day / days_in_month)
            daily_pm25 = base_pm25 * daily_variation
            
            # Ensure non-negative
            daily_pm25 = max(0.1, daily_pm25)
            
            daily_2015_2016.append({
                'date': current_date,
                'pm25_ug_m3': daily_pm25,
                'longitude': month_row['longitude'],
                'latitude': month_row['latitude'],
                'source': month_row['source'] + '_interpolated',
                'frequency': 'daily'
            })
    
    monthly_daily_df = pd.DataFrame(daily_2015_2016)
    print(f"    Created {len(monthly_daily_df)} daily records from monthly data")
    
    # Combine all data
    print(🔗 Combining datasets...")
    all_data = pd.concat([monthly_daily_df, daily_df], ignore_index=True)
    all_data = all_data.sort_values('date')
    
    # Create temporal extrapolation for 2023-2024
    print(" Extrapolating data for 2023-2024...")
    future_data = []
    
    # Use last 2 years of data for trend analysis
    recent_data = all_data[all_data['date'] >= '2021-01-01'].copy()
    
    # Create features for trend modeling
    recent_data['days_since_start'] = (recent_data['date'] - recent_data['date'].min()).dt.days
    recent_data['day_of_year'] = recent_data['date'].dt.dayofyear
    recent_data['year'] = recent_data['date'].dt.year
    
    # Simple trend + seasonal model
    X = recent_data[['days_since_start', 'day_of_year']].values
    y = recent_data['pm25_ug_m3'].values
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future dates
    last_date = all_data['date'].max()
    future_start = datetime(2023, 1, 1)
    future_end = datetime(2024, 12, 31)
    
    current_date = future_start
    while current_date <= future_end:
        days_since_start = (current_date - recent_data['date'].min()).days
        day_of_year = current_date.timetuple().tm_yday
        
        # Predict PM2.5
        predicted_pm25 = model.predict([[days_since_start, day_of_year]])[0]
        
        # Add some realistic variation
        noise = np.random.normal(0, 0.5)  # Small random variation
        predicted_pm25 = max(0.1, predicted_pm25 + noise)
        
        future_data.append({
            'date': current_date,
            'pm25_ug_m3': predicted_pm25,
            'longitude': 28.213,
            'latitude': -12.8167,
            'source': 'temporal_extrapolation',
            'frequency': 'daily'
        })
        
        current_date += timedelta(days=1)
    
    future_df = pd.DataFrame(future_data)
    print(f"   ✅ Generated {len(future_df)} future records")
    
    # Final combined dataset
    final_df = pd.concat([all_data, future_df], ignore_index=True)
    final_df = final_df.sort_values('date')
    
    # Save complete dataset
    output_file = raw_dir / "pm25_kitwe_2015_2024_daily_complete.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\n🎉 Complete dataset saved to: {output_file}")
    print(f"📊 Total records: {len(final_df)}")
    print(f"📅 Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"📈 PM2.5 range: {final_df['pm25_ug_m3'].min():.2f} to {final_df['pm25_ug_m3'].max():.2f} µg/m³")
    
    # Summary by data source
    print(f"\n📋 Data source summary:")
    source_summary = final_df['source'].value_counts()
    for source, count in source_summary.items():
        print(f"   {source}: {count} records")
    
    return final_df

if __name__ == "__main__":
    create_complete_dataset()
