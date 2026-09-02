"""
Model comparison and evaluation summary for PM2.5 forecasting.

This module provides:
- Performance comparison across all models
- Model ranking and recommendations
- Summary statistics and visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_model_results() -> dict:
    """
    Load results from all trained models.
    
    Returns:
        Dictionary with model results
    """
    print("Loading model results...")
    
    results = {}
    models_dir = Path("d:/SCHOOL PROJECT - Copy/kitwe-pm25-forecasting/models")
    
    # Load ARIMA results
    arima_path = models_dir / "arima_model.pkl"
    if arima_path.exists():
        with open(arima_path, 'rb') as f:
            results['arima'] = pickle.load(f)
        print("   ARIMA results loaded")
    
    # Load Prophet results
    prophet_path = models_dir / "prophet_model.pkl"
    if prophet_path.exists():
        with open(prophet_path, 'rb') as f:
            results['prophet'] = pickle.load(f)
        print("   Prophet results loaded")
    
    # Load LSTM results
    lstm_path = models_dir / "lstm_results.pkl"
    if lstm_path.exists():
        with open(lstm_path, 'rb') as f:
            results['lstm'] = pickle.load(f)
        print("   LSTM results loaded")
    
    return results


def create_model_comparison_table(results: dict) -> pd.DataFrame:
    """
    Create comparison table of all models.
    
    Args:
        results: Dictionary with model results
        
    Returns:
        DataFrame with model comparison
    """
    print("Creating model comparison table...")
    
    comparison_data = []
    
    for model_name, model_results in results.items():
        if 'metrics' in model_results:
            metrics = model_results['metrics']
            
            row = {
                'Model': model_name.upper().replace('ARMA', 'ARIMA'),
                'RMSE': metrics.get('rmse', np.nan),
                'MAE': metrics.get('mae', np.nan),
                'MAPE': metrics.get('mape', np.nan),
                'MSE': metrics.get('mse', np.nan)
            }
            
            # Add model-specific details
            if model_name == 'arima':
                row['Parameters'] = f"ARIMA{model_results.get('order', 'N/A')}"
                row['AIC'] = model_results.get('aic', np.nan)
            elif model_name == 'prophet':
                row['Parameters'] = "Yearly+Weekly+Holidays"
                row['Holidays'] = len(model_results.get('holidays', pd.DataFrame()))
            elif model_name == 'lstm':
                row['Parameters'] = "2×50 LSTM, Dropout=0.2"
                row['Total_Params'] = "39,651"
            
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Rank models by RMSE (lower is better)
    comparison_df['Rank_RMSE'] = comparison_df['RMSE'].rank()
    comparison_df['Rank_MAE'] = comparison_df['MAE'].rank()
    comparison_df['Rank_MAPE'] = comparison_df['MAPE'].rank()
    comparison_df['Overall_Rank'] = (comparison_df['Rank_RMSE'] + comparison_df['Rank_MAE'] + comparison_df['Rank_MAPE']) / 3
    
    # Sort by overall rank
    comparison_df = comparison_df.sort_values('Overall_Rank')
    
    return comparison_df


def generate_forecast_summary(results: dict) -> pd.DataFrame:
    """
    Generate summary of forecasts for all models.
    
    Args:
        results: Dictionary with model results
        
    Returns:
        DataFrame with forecast summary
    """
    print("Generating forecast summary...")
    
    forecast_data = []
    
    for model_name, model_results in results.items():
        # 7-day forecast
        if f'forecast_7d' in model_results:
            forecast_7d = model_results['forecast_7d']
            
            if isinstance(forecast_7d, np.ndarray):
                mean_7d = np.mean(forecast_7d)
                std_7d = np.std(forecast_7d)
                min_7d = np.min(forecast_7d)
                max_7d = np.max(forecast_7d)
            elif hasattr(forecast_7d, 'yhat'):
                # Prophet DataFrame
                mean_7d = forecast_7d['yhat'].mean()
                std_7d = forecast_7d['yhat'].std()
                min_7d = forecast_7d['yhat'].min()
                max_7d = forecast_7d['yhat'].max()
            else:
                # Fallback for other structures (like Prophet DataFrame)
                if hasattr(forecast_7d, 'columns') and len(forecast_7d.columns) > 0:
                    # It's a DataFrame, use the first numeric column
                    numeric_cols = forecast_7d.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        col = numeric_cols[0]
                        mean_7d = forecast_7d[col].mean()
                        std_7d = forecast_7d[col].std()
                        min_7d = forecast_7d[col].min()
                        max_7d = forecast_7d[col].max()
                    else:
                        mean_7d = std_7d = min_7d = max_7d = 0
                elif hasattr(forecast_7d, 'mean'):
                    # It's a Series
                    mean_7d = forecast_7d.mean()
                    std_7d = forecast_7d.std()
                    min_7d = forecast_7d.min()
                    max_7d = forecast_7d.max()
                else:
                    mean_7d = std_7d = min_7d = max_7d = 0
            
            forecast_data.append({
                'Model': model_name.upper().replace('ARMA', 'ARIMA'),
                'Forecast_Period': '7 days',
                'Mean_PM25': mean_7d,
                'Std_PM25': std_7d,
                'Min_PM25': min_7d,
                'Max_PM25': max_7d
            })
        
        # 30-day forecast
        if f'forecast_30d' in model_results:
            forecast_30d = model_results['forecast_30d']
            
            if isinstance(forecast_30d, np.ndarray):
                mean_30d = np.mean(forecast_30d)
                std_30d = np.std(forecast_30d)
                min_30d = np.min(forecast_30d)
                max_30d = np.max(forecast_30d)
            elif hasattr(forecast_30d, 'yhat'):
                # Prophet DataFrame
                mean_30d = forecast_30d['yhat'].mean()
                std_30d = forecast_30d['yhat'].std()
                min_30d = forecast_30d['yhat'].min()
                max_30d = forecast_30d['yhat'].max()
            else:
                # Fallback for other structures (like Prophet DataFrame)
                if hasattr(forecast_30d, 'columns') and len(forecast_30d.columns) > 0:
                    # It's a DataFrame, use the first numeric column
                    numeric_cols = forecast_30d.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        col = numeric_cols[0]
                        mean_30d = forecast_30d[col].mean()
                        std_30d = forecast_30d[col].std()
                        min_30d = forecast_30d[col].min()
                        max_30d = forecast_30d[col].max()
                    else:
                        mean_30d = std_30d = min_30d = max_30d = 0
                elif hasattr(forecast_30d, 'mean'):
                    # It's a Series
                    mean_30d = forecast_30d.mean()
                    std_30d = forecast_30d.std()
                    min_30d = forecast_30d.min()
                    max_30d = forecast_30d.max()
                else:
                    mean_30d = std_30d = min_30d = max_30d = 0
            
            forecast_data.append({
                'Model': model_name.upper().replace('ARMA', 'ARIMA'),
                'Forecast_Period': '30 days',
                'Mean_PM25': mean_30d,
                'Std_PM25': std_30d,
                'Min_PM25': min_30d,
                'Max_PM25': max_30d
            })
    
    return pd.DataFrame(forecast_data)


def create_model_recommendations(comparison_df: pd.DataFrame) -> str:
    """
    Generate model recommendations based on performance.
    
    Returns:
        String with recommendations
    """
    print("Generating model recommendations...")
    
    best_model = comparison_df.iloc[0]['Model']
    best_rmse = comparison_df.iloc[0]['RMSE']
    
    recommendations = f"""
MODEL RECOMMENDATIONS

BEST OVERALL PERFORMER: {best_model}
   • RMSE: {best_rmse:.4f}
   • Best for: Overall accuracy and reliability

MODEL STRENGTHS:
"""
    
    # Add specific recommendations for each model
    for _, row in comparison_df.iterrows():
        model = row['Model']
        rmse = row['RMSE']
        
        if model == 'LSTM':
            recommendations += f"\n   {model}: Deep learning with temporal patterns"
            recommendations += f"\n          • Best for complex non-linear relationships"
            recommendations += f"\n          • Captures long-term dependencies"
        elif model == 'PROPHET':
            recommendations += f"\n   {model}: Business-ready forecasting"
            recommendations += f"\n          • Best for interpretability and holidays"
            recommendations += f"\n          • Handles missing data well"
        elif model == 'ARMA':
            recommendations += f"\n   {model}: Classical time series"
            recommendations += f"\n          • Best for baseline comparisons"
            recommendations += f"\n          • Fast and lightweight"
    
    recommendations += f"\n\nUSAGE RECOMMENDATIONS:"
    recommendations += f"\n   • Production use: {best_model}"
    recommendations += f"\n   • Quick forecasts: ARMA"
    recommendations += f"\n   • Business reporting: Prophet"
    recommendations += f"\n   • Research/Complex patterns: LSTM"
    
    return recommendations


def save_comprehensive_report(results: dict, comparison_df: pd.DataFrame, 
                            forecast_df: pd.DataFrame) -> None:
    """
    Save comprehensive model comparison report.
    
    Args:
        results: Model results
        comparison_df: Comparison table
        forecast_df: Forecast summary
    """
    print("Saving comprehensive report...")
    
    output_dir = Path("d:/SCHOOL PROJECT - Copy/kitwe-pm25-forecasting/reports")
    output_dir.mkdir(exist_ok=True)
    
    # Save comparison table
    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    # Save forecast summary
    forecast_path = output_dir / "forecast_summary.csv"
    forecast_df.to_csv(forecast_path, index=False)
    
    # Generate text report
    report_path = output_dir / "model_evaluation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("PM2.5 FORECASTING MODEL EVALUATION REPORT")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODEL PERFORMANCE COMPARISON\n")
        f.write("-" * 30 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("FORECAST SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(forecast_df.to_string(index=False))
        f.write("\n\n")
        
        f.write(create_model_recommendations(comparison_df))
        f.write("\n\n")
        
        f.write("DATA SUMMARY\n")
        f.write("-" * 15 + "\n")
        f.write("• Training period: 2015-2022 (8 years)\n")
        f.write("• Validation period: 2023 (1 year)\n")
        f.write("• Test period: 2024 (1 year)\n")
        f.write("• Total observations: 3,554 (after preprocessing)\n")
        f.write("• Features engineered: 53 total\n")
        f.write("• LSTM sequence length: 30 days\n")
    
    print(f"   Report saved to: {report_path}")
    print(f"   Comparison table: {comparison_path}")


def main():
    """Main function to run complete model comparison."""
    print("Starting comprehensive model evaluation...")
    
    # Load all model results
    results = load_model_results()
    
    if not results:
        print("No model results found")
        return
    
    # Create comparison table
    comparison_df = create_model_comparison_table(results)
    
    # Generate forecast summary
    forecast_df = generate_forecast_summary(results)
    
    # Generate recommendations
    recommendations = create_model_recommendations(comparison_df)
    print(recommendations)
    
    # Save comprehensive report
    save_comprehensive_report(results, comparison_df, forecast_df)
    
    print("\nModel evaluation complete!")
    print(f"Best performing model: {comparison_df.iloc[0]['Model']}")
    print(f"Best RMSE: {comparison_df.iloc[0]['RMSE']:.4f}")


if __name__ == "__main__":
    main()
