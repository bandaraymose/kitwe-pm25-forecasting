# PM2.5 Forecasting Dashboard

## Overview
Interactive web dashboard for PM2.5 air quality forecasting in Kitwe, Zambia (2015-2024).

## Features

### 🌫️ Real-time Air Quality Monitoring
- Interactive PM2.5 time series plots (2015-2024)
- WHO guideline compliance visualization
- Date range filtering for custom analysis

### 🔮 Model Forecast Comparison
- ARMA (ARIMA) model forecasts
- Prophet model with seasonality and holidays
- LSTM deep learning predictions
- Side-by-side forecast comparisons

### 🌤️ Seasonal Analysis
- Monthly average PM2.5 patterns
- Seasonal decomposition (Southern Hemisphere)
- Season-specific statistics

### 🏥 Health Guidelines
- WHO air quality classification
- Color-coded air quality levels
- Compliance percentage tracking

### 📊 Performance Metrics
- Model comparison table (RMSE, MAE, MAPE)
- Best performer highlighting
- Detailed model configurations

## Quick Start

### Prerequisites
- Python 3.8+
- All model files generated from previous steps

### Installation
```bash
# Navigate to dashboard directory
cd dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

### Access
- Local: http://localhost:8501
- Network: http://YOUR_IP:8501

## Dashboard Sections

### 1. 📈 Time Series Analysis
- Interactive plot of PM2.5 levels
- WHO guideline reference line
- Date range filtering
- Real-time statistics (average, min, max, good days)

### 2. 🔮 Forecasts
- 7-day forecast comparison
- Model selection toggle
- Visual forecast comparison
- Interactive hover information

### 3. 🌤️ Seasonal Analysis
- Monthly average bar chart
- Seasonal statistics table
- Southern hemisphere seasons
- Pattern identification

### 4. 🏥 WHO Guidelines
- Heatmap visualization
- Air quality classification
- Compliance tracking
- Health impact information

### 5. 📊 Performance Metrics
- Model comparison table
- Performance highlighting
- Detailed model information
- Configuration parameters

## Data Requirements

The dashboard expects the following files:

### Data Files
- `data/processed/pm25_features.csv` - Complete dataset with features
- `reports/model_comparison.csv` - Model performance comparison

### Model Files
- `models/arma_model.pkl` - ARMA model and results
- `models/prophet_model.pkl` - Prophet model and results  
- `models/lstm_results.pkl` - LSTM training results

## Technical Architecture

### Frontend
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **HTML/CSS**: Custom styling

### Backend
- **Python**: Data processing and model loading
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Pickle**: Model serialization

### Visualization Components
- Time series plots with hover information
- Forecast comparison charts
- Seasonal decomposition visualizations
- WHO compliance heatmaps
- Performance metrics tables

## Configuration

### Customization Options
- Date range limits
- Model selection defaults
- Color schemes
- Layout preferences

### Environment Variables
```bash
# Optional: Set custom port
STREAMLIT_SERVER_PORT=8501

# Optional: Enable debug mode
STREAMLIT_LOGGER_LEVEL=debug
```

## Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Network Access
```bash
streamlit run app.py --server.address 0.0.0.0
```

### Streamlit Community Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Troubleshooting

### Common Issues

1. **Data Files Not Found**
   - Ensure all preprocessing steps are complete
   - Check file paths in the code
   - Verify data directory structure

2. **Model Loading Errors**
   - Confirm model files exist in `models/` directory
   - Check Python version compatibility
   - Verify all dependencies installed

3. **Visualization Issues**
   - Update Plotly to latest version
   - Check browser compatibility
   - Clear browser cache

4. **Performance Issues**
   - Reduce data size for testing
   - Optimize data loading with caching
   - Use sampling for large datasets

### Performance Optimization

- Use `@st.cache_data` for expensive operations
- Implement data sampling for large datasets
- Optimize Plotly figure rendering
- Use lazy loading for model results

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Security Considerations

- No sensitive data exposure
- Input validation for date ranges
- Secure model file handling
- Safe deployment practices

## Future Enhancements

### Planned Features
- Real-time data integration
- Additional forecast models
- Export functionality
- Mobile responsiveness
- Multi-city support

### Advanced Analytics
- Anomaly detection
- Trend analysis
- Correlation studies
- Health impact assessment

## Support

For issues and questions:
1. Check troubleshooting section
2. Verify data and model files
3. Review error logs
4. Check dependency versions

## License

This dashboard is part of the PM2.5 forecasting project for Kitwe, Zambia.
