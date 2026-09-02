"""
PM2.5 Forecasting Dashboard - Streamlit Web Application

This dashboard provides interactive visualization of PM2.5 data and model forecasts
for Kitwe, Zambia (2015-2024).

Features:
- Interactive time series plots
- Model forecast comparisons
- Seasonal decomposition
- WHO guideline classifications
- Performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PM2.5 Forecasting Dashboard - Kitwe, Zambia",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .model-comparison {
        background: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load processed PM2.5 data and model results."""
    data = {}
    
    # Load complete dataset
    data_path = Path("../data/processed/pm25_features.csv")
    if data_path.exists():
        data['complete'] = pd.read_csv(data_path)
        data['complete']['date'] = pd.to_datetime(data['complete']['date'])
    
    # Load model comparison
    comparison_path = Path("../reports/model_comparison.csv")
    if comparison_path.exists():
        data['comparison'] = pd.read_csv(comparison_path)
    
    # Load model results
    models_dir = Path("../models")
    data['models'] = {}
    
    # ARMA results
    arma_path = models_dir / "arma_model.pkl"
    if arma_path.exists():
        with open(arma_path, 'rb') as f:
            data['models']['arma'] = pickle.load(f)
    
    # Prophet results
    prophet_path = models_dir / "prophet_model.pkl"
    if prophet_path.exists():
        with open(prophet_path, 'rb') as f:
            data['models']['prophet'] = pickle.load(f)
    
    # LSTM results
    lstm_path = models_dir / "lstm_results.pkl"
    if lstm_path.exists():
        with open(lstm_path, 'rb') as f:
            data['models']['lstm'] = pickle.load(f)
    
    return data

def classify_pm25_level(pm25_value):
    """Classify PM2.5 levels according to WHO guidelines."""
    if pm25_value <= 15:
        return "Good", "#00E400"
    elif pm25_value <= 35:
        return "Moderate", "#FFFF00"
    elif pm25_value <= 55:
        return "Unhealthy for Sensitive", "#FF7E00"
    elif pm25_value <= 150:
        return "Unhealthy", "#FF0000"
    else:
        return "Very Unhealthy", "#8F3F97"

def create_time_series_plot(df, start_date=None, end_date=None):
    """Create interactive time series plot."""
    if start_date and end_date:
        # Convert date objects to pandas Timestamp for proper comparison
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        df_filtered = df[(df['date'] >= start_ts) & (df['date'] <= end_ts)]
    else:
        df_filtered = df
    
    fig = go.Figure()
    
    # Add PM2.5 line
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['pm25_ug_m3'],
        mode='lines',
        name='PM2.5 Levels',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>PM2.5: %{y:.2f} µg/m³<extra></extra>'
    ))
    
    # Add WHO guideline line
    fig.add_hline(y=15, line_dash="dash", line_color="red", 
                 annotation_text="WHO Guideline (15 µg/m³)")
    
    fig.update_layout(
        title="PM2.5 Concentration Levels - Kitwe, Zambia",
        xaxis_title="Date",
        yaxis_title="PM2.5 (µg/m³)",
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_forecast_comparison_plot(models_data):
    """Create forecast comparison plot for all models showing future predictions."""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    model_names = ['ARMA', 'Prophet', 'LSTM']
    
    # Get current date for future predictions
    current_date = pd.Timestamp.now().normalize()
    
    for i, (model_name, model_data) in enumerate(models_data.items()):
        if f'forecast_7d' in model_data:
            forecast = model_data['forecast_7d']
            
            if isinstance(forecast, pd.DataFrame):
                # Prophet forecast
                if 'yhat' in forecast.columns:
                    y_values = forecast['yhat'].values
                else:
                    # Use first numeric column
                    numeric_cols = forecast.select_dtypes(include=[np.number]).columns
                    y_values = forecast[numeric_cols[0]].values if len(numeric_cols) > 0 else []
            else:
                # ARMA/LSTM numpy array
                y_values = forecast
            
            # Create future dates starting from today
            future_dates = [current_date + pd.Timedelta(days=i+1) for i in range(len(y_values))]
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=y_values,
                mode='lines+markers',
                name=f'{model_name.upper()} 7-day Forecast',
                line=dict(color=colors[i], width=2),
                marker=dict(size=6),
                hovertemplate=f'Date: %{{x}}<br>{model_name.upper()} PM2.5: %{{y:.2f}} µg/m³<extra></extra>'
            ))
    
    # Add current date reference line
    fig.add_vline(x=current_date.isoformat(), line_dash="dash", line_color="gray")
    
    # Add WHO guideline line
    fig.add_hline(y=15, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=f"Future PM2.5 Forecasts - Starting {current_date.strftime('%Y-%m-%d')}",
        xaxis_title="Future Dates",
        yaxis_title="Predicted PM2.5 (µg/m³)",
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_30day_forecast_plot(models_data):
    """Create 30-day forecast plot with confidence intervals."""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Get current date for future predictions
    current_date = pd.Timestamp.now().normalize()
    
    # Show only Prophet with confidence intervals for clarity
    if 'prophet' in models_data and 'forecast_30d' in models_data['prophet']:
        forecast = models_data['prophet']['forecast_30d']
        
        if isinstance(forecast, pd.DataFrame) and 'yhat' in forecast.columns:
            # Create future dates
            future_dates = [current_date + pd.Timedelta(days=i+1) for i in range(len(forecast))]
            
            # Main forecast line
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast['yhat'],
                mode='lines',
                name='Prophet 30-day Forecast',
                line=dict(color='#ff7f0e', width=2),
                hovertemplate='Date: %{x}<br>PM2.5: %{y:.2f} µg/m³<extra></extra>'
            ))
            
            # Confidence intervals
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,127,14,0.2)',
                    name='80% Confidence Interval',
                    hoverinfo='skip'
                ))
    
    # Add WHO guideline line
    fig.add_hline(y=15, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=f"30-Day PM2.5 Forecast with Confidence Intervals - Starting {current_date.strftime('%Y-%m-%d')}",
        xaxis_title="Future Dates",
        yaxis_title="Predicted PM2.5 (µg/m³)",
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_seasonal_decomposition_plot(df):
    """Create seasonal decomposition visualization."""
    # Simple seasonal visualization (monthly averages)
    df['month'] = df['date'].dt.month
    monthly_avg = df.groupby('month')['pm25_ug_m3'].mean().reset_index()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: month_names[x-1])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_avg['month_name'],
        y=monthly_avg['pm25_ug_m3'],
        marker_color='#1f77b4',
        name='Monthly Average PM2.5'
    ))
    
    fig.update_layout(
        title="Seasonal Pattern - Monthly Average PM2.5 Levels",
        xaxis_title="Month",
        yaxis_title="Average PM2.5 (µg/m³)",
        height=400
    )
    
    return fig

def create_who_heatmap(df):
    """Create WHO guideline classification heatmap."""
    # Daily classification
    df['classification'], df['color'] = zip(*df['pm25_ug_m3'].apply(classify_pm25_level))
    
    # Create heatmap data
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Pivot for heatmap
    heatmap_data = df.pivot_table(
        values='pm25_ug_m3',
        index='day_of_year',
        columns='year',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlBu_r',
        colorbar=dict(title="PM2.5 (µg/m³)")
    ))
    
    fig.update_layout(
        title="PM2.5 Levels Heatmap - WHO Guideline Compliance",
        xaxis_title="Year",
        yaxis_title="Day of Year",
        height=400
    )
    
    return fig

def create_performance_metrics_table(comparison_df):
    """Create performance metrics display."""
    # Format the comparison data
    display_df = comparison_df[['Model', 'RMSE', 'MAE', 'MAPE']].copy()
    display_df['RMSE'] = display_df['RMSE'].round(4)
    display_df['MAE'] = display_df['MAE'].round(4)
    display_df['MAPE'] = display_df['MAPE'].round(2).astype(str) + '%'
    
    # Highlight best performers
    best_rmse_idx = display_df['RMSE'].idxmin()
    best_mae_idx = display_df['MAE'].idxmin()
    best_mape_idx = display_df['MAPE'].str.rstrip('%').astype(float).idxmin()
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Model', 'RMSE', 'MAE', 'MAPE'],
            fill_color='#1f77b4',
            align='left',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[
                display_df['Model'],
                display_df['RMSE'],
                display_df['MAE'],
                display_df['MAPE']
            ],
            fill_color=[
                ['#f0f2f6'] * len(display_df),
                ['#d4edda' if i == best_rmse_idx else '#f0f2f6' for i in range(len(display_df))],
                ['#d4edda' if i == best_mae_idx else '#f0f2f6' for i in range(len(display_df))],
                ['#d4edda' if i == best_mape_idx else '#f0f2f6' for i in range(len(display_df))]
            ],
            align='left',
            font=dict(color='black', size=11)
        )
    )])
    
    fig.update_layout(
        title="Model Performance Comparison",
        height=300
    )
    
    return fig

def main():
    """Main dashboard application."""
    # Load data
    data = load_data()
    
    if 'complete' not in data or data['complete'].empty:
        st.error("❌ Data files not found. Please ensure data processing is complete.")
        return
    
    df = data['complete']
    
    # Header
    st.markdown('<h1 class="main-header">🌫️ PM2.5 Forecasting Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Kitwe, Zambia • 2015-2024 • Real-time Air Quality Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("📊 Dashboard Controls")
    
    # Date range selector
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Model selection
    st.sidebar.header("🤖 Model Selection")
    show_arma = st.sidebar.checkbox("ARMA", value=True)
    show_prophet = st.sidebar.checkbox("Prophet", value=True)
    show_lstm = st.sidebar.checkbox("LSTM", value=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Time Series", "🔮 Forecasts", "🌤️ Seasonal Analysis", 
        "🏥 WHO Guidelines", "📊 Performance"
    ])
    
    with tab1:
        st.header("PM2.5 Time Series Analysis")
        
        # Time series plot
        fig_ts = create_time_series_plot(df, start_date, end_date)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        filtered_df = df[(df['date'] >= pd.Timestamp(start_date)) & 
                        (df['date'] <= pd.Timestamp(end_date))]
        
        with col1:
            st.metric("Average PM2.5", f"{filtered_df['pm25_ug_m3'].mean():.2f} µg/m³")
        
        with col2:
            st.metric("Max PM2.5", f"{filtered_df['pm25_ug_m3'].max():.2f} µg/m³")
        
        with col3:
            st.metric("Min PM2.5", f"{filtered_df['pm25_ug_m3'].min():.2f} µg/m³")
        
        with col4:
            good_days = (filtered_df['pm25_ug_m3'] <= 15).sum()
            st.metric("Good Days (≤15 µg/m³)", f"{good_days} ({good_days/len(filtered_df)*100:.1f}%)")
    
    with tab2:
        st.header("🔮 Future PM2.5 Forecasts")
        st.info("📅 **Note:** These are genuine future predictions starting from today's date, not historical data analysis.")
        
        models_to_show = {}
        if show_arma and 'arma' in data.get('models', {}):
            models_to_show['arma'] = data['models']['arma']
        if show_prophet and 'prophet' in data.get('models', {}):
            models_to_show['prophet'] = data['models']['prophet']
        if show_lstm and 'lstm' in data.get('models', {}):
            models_to_show['lstm'] = data['models']['lstm']
        
        if models_to_show:
            # 7-day forecast comparison
            st.subheader("7-Day Forecast Comparison")
            fig_forecast = create_forecast_comparison_plot(models_to_show)
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # 30-day forecast with confidence intervals
            if 'prophet' in models_to_show:
                st.subheader("30-Day Forecast with Confidence Intervals")
                fig_30d = create_30day_forecast_plot(models_to_show)
                st.plotly_chart(fig_30d, use_container_width=True)
            
            # Forecast summary table
            st.subheader("Forecast Summary")
            current_date = pd.Timestamp.now().normalize()
            
            summary_data = []
            for model_name, model_data in models_to_show.items():
                if f'forecast_7d' in model_data:
                    forecast = model_data['forecast_7d']
                    
                    if isinstance(forecast, pd.DataFrame) and 'yhat' in forecast.columns:
                        avg_7d = forecast['yhat'].mean()
                        max_7d = forecast['yhat'].max()
                        min_7d = forecast['yhat'].min()
                    else:
                        avg_7d = np.mean(forecast) if isinstance(forecast, np.ndarray) else 0
                        max_7d = np.max(forecast) if isinstance(forecast, np.ndarray) else 0
                        min_7d = np.min(forecast) if isinstance(forecast, np.ndarray) else 0
                    
                    summary_data.append({
                        'Model': model_name.upper(),
                        '7-Day Avg': f"{avg_7d:.2f}",
                        '7-Day Min': f"{min_7d:.2f}",
                        '7-Day Max': f"{max_7d:.2f}",
                        'WHO Compliance': "✅" if avg_7d <= 15 else "⚠️"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Health recommendations based on forecasts
            st.subheader("🏥 Health Recommendations")
            
            # Get the average forecast from the best model (LSTM if available)
            if 'lstm' in models_to_show and 'forecast_7d' in models_to_show['lstm']:
                lstm_forecast = models_to_show['lstm']['forecast_7d']
                avg_forecast = np.mean(lstm_forecast) if isinstance(lstm_forecast, np.ndarray) else 0
                
                if avg_forecast <= 15:
                    st.success("🟢 **Good Air Quality Expected** - Normal outdoor activities recommended")
                elif avg_forecast <= 35:
                    st.warning("🟡 **Moderate Air Quality Expected** - Sensitive individuals should limit prolonged outdoor exertion")
                elif avg_forecast <= 55:
                    st.error("🟠 **Unhealthy for Sensitive Groups Expected** - People with respiratory conditions should avoid outdoor activities")
                else:
                    st.error("🔴 **Unhealthy Air Quality Expected** - Everyone should avoid prolonged outdoor exertion")
        else:
            st.warning("⚠️ No models selected. Please select models from the sidebar.")
    
    with tab3:
        st.header("Seasonal Analysis")
        
        # Seasonal decomposition
        fig_seasonal = create_seasonal_decomposition_plot(df)
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Seasonal statistics
        st.subheader("Seasonal Statistics")
        df['season'] = df['date'].dt.month.apply(lambda x: 
            'Summer' if x in [12, 1, 2] else
            'Autumn' if x in [3, 4, 5] else
            'Winter' if x in [6, 7, 8] else 'Spring'
        )
        
        seasonal_stats = df.groupby('season')['pm25_ug_m3'].agg(['mean', 'std', 'count']).round(2)
        st.dataframe(seasonal_stats, use_container_width=True)
    
    with tab4:
        st.header("WHO Guideline Compliance")
        
        # WHO heatmap
        fig_who = create_who_heatmap(df)
        st.plotly_chart(fig_who, use_container_width=True)
        
        # WHO classification summary
        st.subheader("Air Quality Classification")
        
        classification_counts = df['pm25_ug_m3'].apply(classify_pm25_level).apply(lambda x: x[0]).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            for classification, count in classification_counts.items():
                percentage = count / len(df) * 100
                st.metric(classification, f"{count} days ({percentage:.1f}%)")
        
        with col2:
            # WHO guideline info
            st.info("""
            **WHO Air Quality Guidelines for PM2.5:**
            - **Good:** ≤15 µg/m³
            - **Moderate:** 16-35 µg/m³
            - **Unhealthy for Sensitive:** 36-55 µg/m³
            - **Unhealthy:** 56-150 µg/m³
            - **Very Unhealthy:** >150 µg/m³
            """)
    
    with tab5:
        st.header("Model Performance Metrics")
        
        if 'comparison' in data:
            fig_metrics = create_performance_metrics_table(data['comparison'])
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Additional metrics
            st.subheader("Model Details")
            
            for _, row in data['comparison'].iterrows():
                with st.expander(f"{row['Model']} Details"):
                    st.write(f"**Configuration:** {row['Parameters']}")
                    if not pd.isna(row.get('AIC')):
                        st.write(f"**AIC:** {row['AIC']:.2f}")
                    if not pd.isna(row.get('Total_Params')):
                        st.write(f"**Total Parameters:** {row['Total_Params']}")
        else:
            st.warning("⚠️ Model comparison data not available.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    PM2.5 Forecasting Dashboard | Kitwe, Zambia | Data Source: Satellite Observations (2015-2024)
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
