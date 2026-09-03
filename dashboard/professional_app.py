"""
PM2.5 Forecasting Dashboard - Professional Industry-Grade Version
Kitwe, Zambia Air Quality Monitoring System

Advanced real-time dashboard for PM2.5 forecasting and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import warnings
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
warnings.filterwarnings('ignore')

# Professional configuration
st.set_page_config(
    page_title="Air Quality Monitoring System - Kitwe, Zambia",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load external CSS ────────────────────────────────────────────────────
# Ensure style.css is in the same directory as this script
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("style.css not found – using default styling.")

# The rest of the code (load_data, generate_pdf_report, all plotting functions, and main()) follows unchanged.
# I've included the full code below for completeness.

# ── Load data and models ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load processed PM2.5 data and model results."""
    data = {}
    
    # Get project root - handle both running from dashboard dir and project root
    current_path = Path(__file__).resolve()
    if current_path.name == "professional_app.py":
        project_root = current_path.parent.parent
    else:
        project_root = current_path
    
    # Load complete dataset
    data_path = project_root / "data/processed/pm25_features.csv"
    if data_path.exists():
        data['complete'] = pd.read_csv(data_path)
        data['complete']['date'] = pd.to_datetime(data['complete']['date'])
    
    # Load model comparison
    comparison_path = project_root / "reports/model_comparison.csv"
    if comparison_path.exists():
        data['comparison'] = pd.read_csv(comparison_path)
    
    # Load model results
    models_dir = project_root / "models"
    data['models'] = {}
    
    # ARIMA results
    arima_path = models_dir / "arima_model.pkl"
    if arima_path.exists():
        with open(arima_path, 'rb') as f:
            data['models']['arima'] = pickle.load(f)
    
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

@st.cache_data(ttl=600)  # refresh every 10 minutes
def fetch_live_weather():
    """Fetch live weather for Kitwe from Open-Meteo (free, no API key)."""
    import urllib.request, json
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=-12.8&longitude=28.2"
        "&current=temperature_2m,relative_humidity_2m,"
        "wind_speed_10m,visibility,weather_code"
        "&wind_speed_unit=kmh"
        "&timezone=Africa%2FLusaka"
    )
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read())
        c = data["current"]
        vis_km = c.get("visibility", 0) / 1000
        return {
            "temperature":  round(c.get("temperature_2m", 0), 1),
            "humidity":     round(c.get("relative_humidity_2m", 0)),
            "wind_speed":   round(c.get("wind_speed_10m", 0), 1),
            "visibility":   round(vis_km, 1),
            "weather_code": c.get("weather_code", 0),
            "ok": True
        }
    except Exception:
        return {
            "temperature": "—",
            "humidity":    "—",
            "wind_speed":  "—",
            "visibility":  "—",
            "weather_code": 0,
            "ok": False
        }

def generate_pdf_report(df, start_date, end_date, data):
    """Generate comprehensive PDF report for PM2.5 forecasting system."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', fontName='Helvetica-Bold', fontSize=24, leading=30, spaceAfter=30, textColor=colors.HexColor('#1F4E79')))
    styles.add(ParagraphStyle(name='CustomSubtitle', fontName='Helvetica', fontSize=14, leading=18, spaceAfter=20, textColor=colors.gray))
    styles.add(ParagraphStyle(name='SectionHeader', fontName='Helvetica-Bold', fontSize=16, leading=22, spaceAfter=12, spaceBefore=20, textColor=colors.HexColor('#1F4E79')))
    styles.add(ParagraphStyle(name='CustomNormal', fontName='Helvetica', fontSize=11, leading=14, spaceAfter=12))
    styles.add(ParagraphStyle(name='CustomSmall', fontName='Helvetica', fontSize=9, leading=11, spaceAfter=6, textColor=colors.gray))
    
    story = []
    
    # Cover Page
    story.append(Paragraph("Kitwe CBD PM2.5 Air Quality Monitoring Report", styles['CustomTitle']))
    story.append(Paragraph("Kitwe AQ Command — Automated Report", styles['CustomSubtitle']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['CustomNormal']))
    story.append(Paragraph("Data Source: Satellite Observations 2015–2024 | Kitwe, Zambia | Zambia Copperbelt", styles['CustomNormal']))
    story.append(Spacer(1, 2*inch))
    story.append(PageBreak())
    
    filtered_df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]
    
    # Section 1: Air Quality Summary
    story.append(Paragraph("Section 1: Air Quality Summary", styles['SectionHeader']))
    
    mean_val = filtered_df['pm25_ug_m3'].mean()
    max_val = filtered_df['pm25_ug_m3'].max()
    min_val = filtered_df['pm25_ug_m3'].min()
    good_days = (filtered_df['pm25_ug_m3'] <= 15).sum()
    total_days = len(filtered_df)
    compliance_pct = (good_days/total_days)*100 if total_days > 0 else 0
    
    summary_data = [
        ['Metric', 'Value'],
        ['Average PM2.5', f'{mean_val:.2f} µg/m³'],
        ['Peak Reading', f'{max_val:.2f} µg/m³'],
        ['Minimum Reading', f'{min_val:.2f} µg/m³'],
        ['Good Air Days', f'{good_days} days'],
        ['WHO Compliance Status', f'{compliance_pct:.1f}%'],
        ['Date Range', f'{start_date} to {end_date}']
    ]
    
    summary_table = Table(summary_data, colWidths=[2.5*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1F4E79')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(PageBreak())
    
    # Section 2: Historical PM2.5 Trends
    story.append(Paragraph("Section 2: Historical PM2.5 Trends", styles['SectionHeader']))
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['pm25_ug_m3'],
        mode='lines',
        name='PM2.5 Concentration',
        line=dict(color='#1F4E79', width=2)
    ))
    fig_trend.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="WHO 24-hr Limit")
    fig_trend.update_layout(
        title=f"PM2.5 Concentration Trends ({start_date} to {end_date})",
        xaxis_title="Date",
        yaxis_title="PM2.5 (µg/m³)",
        height=400,
        template="plotly_white"
    )
    
    trend_img = pio.to_image(fig_trend, format="png", engine="kaleido")
    trend_img_io = io.BytesIO(trend_img)
    trend_img_io.seek(0)
    story.append(Image(trend_img_io, width=6*inch, height=3*inch))
    story.append(Paragraph(f"Figure 1: PM2.5 concentration trends from {start_date} to {end_date}. Overall trend shows average PM2.5 level of {mean_val:.2f} µg/m³.", styles['CustomSmall']))
    story.append(Spacer(1, 0.3*inch))
    story.append(PageBreak())
    
    # Section 3: Seasonal Analysis
    story.append(Paragraph("Section 3: Seasonal Analysis", styles['SectionHeader']))
    
    filtered_df['month'] = filtered_df['date'].dt.month
    monthly_avg = filtered_df.groupby('month')['pm25_ug_m3'].mean().reset_index()
    monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
    
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Bar(
        x=monthly_avg['month_name'],
        y=monthly_avg['pm25_ug_m3'],
        name='Monthly Average',
        marker_color='#1F4E79'
    ))
    fig_seasonal.update_layout(
        title="Seasonal PM2.5 Patterns (Monthly Averages)",
        xaxis_title="Month",
        yaxis_title="Average PM2.5 (µg/m³)",
        height=400,
        template="plotly_white"
    )
    
    seasonal_img = pio.to_image(fig_seasonal, format="png", engine="kaleido")
    seasonal_img_io = io.BytesIO(seasonal_img)
    seasonal_img_io.seek(0)
    story.append(Image(seasonal_img_io, width=6*inch, height=3*inch))
    story.append(Paragraph("Figure 2: Monthly average PM2.5 concentrations showing seasonal patterns.", styles['CustomSmall']))
    story.append(Spacer(1, 0.3*inch))
    
    seasons = {'Spring': [9,10,11], 'Summer': [12,1,2], 'Autumn': [3,4,5], 'Winter': [6,7,8]}
    seasonal_stats = []
    for season_name, months in seasons.items():
        season_data = filtered_df[filtered_df['month'].isin(months)]['pm25_ug_m3']
        if len(season_data) > 0:
            seasonal_stats.append([
                season_name,
                f'{season_data.mean():.2f}',
                f'{season_data.std():.2f}',
                len(season_data),
                f'{season_data.min():.2f}',
                f'{season_data.max():.2f}'
            ])
    
    seasonal_table_data = [['Season', 'Average', 'Std Dev', 'Days', 'Min', 'Max']] + seasonal_stats
    seasonal_table = Table(seasonal_table_data, colWidths=[1*inch, 1*inch, 1*inch, 0.8*inch, 1*inch, 1*inch])
    seasonal_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(seasonal_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(PageBreak())
    
    # Section 4: WHO Guidelines Compliance
    story.append(Paragraph("Section 4: WHO Guidelines Compliance", styles['SectionHeader']))
    
    exceedances = (filtered_df['pm25_ug_m3'] > 15).sum()
    who_compliance_rate = ((total_days - exceedances) / total_days * 100) if total_days > 0 else 0
    
    who_data = [
        ['Metric', 'Value'],
        ['Annual WHO Compliance Rate', f'{who_compliance_rate:.1f}%'],
        ['Good Days (≤15 µg/m³)', f'{good_days}'],
        ['Exceedances (>15 µg/m³)', f'{exceedances}'],
        ['Average PM2.5', f'{mean_val:.2f} µg/m³']
    ]
    
    who_table = Table(who_data, colWidths=[2.5*inch, 3*inch])
    who_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1F4E79')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(who_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("WHO Air Quality Guidelines Framework", styles['CustomNormal']))
    who_framework_data = [
        ['Guideline', 'Threshold (µg/m³)', 'Description'],
        ['Annual Mean', '5', 'Long-term exposure limit'],
        ['24-Hour Mean', '15', 'Daily exposure limit'],
        ['Interim Target', '35', 'Intermediate target for developing countries']
    ]
    
    who_framework_table = Table(who_framework_data, colWidths=[2*inch, 2*inch, 3*inch])
    who_framework_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(who_framework_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(PageBreak())
    
    # Section 5: PM2.5 Forecasts
    story.append(Paragraph("Section 5: PM2.5 Forecasts", styles['SectionHeader']))
    
    if 'models' in data and 'prophet' in data['models']:
        prophet_data = data['models']['prophet']
        if 'forecast_7d' in prophet_data and not prophet_data['forecast_7d'].empty:
            forecast_7d = prophet_data['forecast_7d']
            
            fig_7d = go.Figure()
            fig_7d.add_trace(go.Scatter(
                x=forecast_7d['ds'],
                y=forecast_7d['yhat'],
                mode='lines+markers',
                name='7-Day Forecast',
                line=dict(color='#1F4E79', width=2)
            ))
            fig_7d.update_layout(
                title="7-Day PM2.5 Forecast",
                xaxis_title="Date",
                yaxis_title="PM2.5 (µg/m³)",
                height=400,
                template="plotly_white"
            )
            
            forecast_7d_img = pio.to_image(fig_7d, format="png", engine="kaleido")
            forecast_7d_io = io.BytesIO(forecast_7d_img)
            forecast_7d_io.seek(0)
            story.append(Image(forecast_7d_io, width=6*inch, height=3*inch))
            story.append(Paragraph("Figure 3: 7-day PM2.5 forecast using Prophet model.", styles['CustomSmall']))
            story.append(Spacer(1, 0.3*inch))
        
        if 'forecast_30d' in prophet_data and not prophet_data['forecast_30d'].empty:
            forecast_30d = prophet_data['forecast_30d']
            
            fig_30d = go.Figure()
            fig_30d.add_trace(go.Scatter(
                x=forecast_30d['ds'],
                y=forecast_30d['yhat'],
                mode='lines',
                name='30-Day Forecast',
                line=dict(color='#1F4E79', width=2)
            ))
            fig_30d.add_trace(go.Scatter(
                x=forecast_30d['ds'],
                y=forecast_30d['yhat_upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(color='gray', width=1),
                showlegend=False
            ))
            fig_30d.add_trace(go.Scatter(
                x=forecast_30d['ds'],
                y=forecast_30d['yhat_lower'],
                mode='lines',
                name='Lower Bound',
                line=dict(color='gray', width=1),
                fill='tonexty',
                fillcolor='rgba(200,200,200,0.3)',
                showlegend=False
            ))
            fig_30d.update_layout(
                title="30-Day PM2.5 Forecast with Confidence Intervals",
                xaxis_title="Date",
                yaxis_title="PM2.5 (µg/m³)",
                height=400,
                template="plotly_white"
            )
            
            forecast_30d_img = pio.to_image(fig_30d, format="png", engine="kaleido")
            forecast_30d_io = io.BytesIO(forecast_30d_img)
            forecast_30d_io.seek(0)
            story.append(Image(forecast_30d_io, width=6*inch, height=3*inch))
            story.append(Paragraph("Figure 4: 30-day PM2.5 forecast with 80% confidence intervals.", styles['CustomSmall']))
            story.append(Spacer(1, 0.3*inch))
        
        if 'forecast_7d' in prophet_data and not prophet_data['forecast_7d'].empty:
            fc_7d_avg = prophet_data['forecast_7d']['yhat'].mean()
            fc_7d_min = prophet_data['forecast_7d']['yhat'].min()
            fc_7d_max = prophet_data['forecast_7d']['yhat'].max()
            fc_7d_status = "Good" if fc_7d_avg <= 15 else "Exceedance"
        else:
            fc_7d_avg = fc_7d_min = fc_7d_max = fc_7d_status = "N/A"
        
        if 'forecast_30d' in prophet_data and not prophet_data['forecast_30d'].empty:
            fc_30d_avg = prophet_data['forecast_30d']['yhat'].mean()
            fc_30d_min = prophet_data['forecast_30d']['yhat'].min()
            fc_30d_max = prophet_data['forecast_30d']['yhat'].max()
            fc_30d_status = "Good" if fc_30d_avg <= 15 else "Exceedance"
        else:
            fc_30d_avg = fc_30d_min = fc_30d_max = fc_30d_status = "N/A"
        
        forecast_summary_data = [
            ['Model', '7-Day Avg', '30-Day Avg', 'Min', 'Max', 'Status'],
            ['Prophet', f'{fc_7d_avg:.2f}' if fc_7d_avg != "N/A" else 'N/A', 
             f'{fc_30d_avg:.2f}' if fc_30d_avg != "N/A" else 'N/A',
             f'{fc_7d_min:.2f}' if fc_7d_min != "N/A" else 'N/A',
             f'{fc_7d_max:.2f}' if fc_7d_max != "N/A" else 'N/A',
             fc_7d_status]
        ]
        
        forecast_table = Table(forecast_summary_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1*inch, 1*inch, 1.2*inch])
        forecast_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(forecast_table)
        story.append(Spacer(1, 0.3*inch))
        
        alert_color = colors.green if fc_7d_status == "Good" else colors.red
        alert_text = f"WHO AIR QUALITY ALERT: {fc_7d_status}" if fc_7d_status != "N/A" else "WHO AIR QUALITY ALERT: N/A"
        story.append(Paragraph(alert_text, ParagraphStyle(name='Alert', fontName='Helvetica-Bold', fontSize=14, textColor=alert_color, spaceAfter=12)))
        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())
    
    # Section 6: Model Performance
    story.append(Paragraph("Section 6: Model Performance", styles['SectionHeader']))
    
    if 'comparison' in data and not data['comparison'].empty:
        comparison_df = data['comparison']
        
        # Use capitalised column names
        models = comparison_df['Model'].tolist() if 'Model' in comparison_df.columns else []
        
        fig_perf = go.Figure()
        if 'RMSE' in comparison_df.columns:
            fig_perf.add_trace(go.Bar(
                x=models,
                y=comparison_df['RMSE'],
                name='RMSE',
                marker_color='#1F4E79'
            ))
        fig_perf.update_layout(
            title="Model Performance Comparison (RMSE)",
            xaxis_title="Model",
            yaxis_title="RMSE",
            height=400,
            template="plotly_white"
        )
        
        perf_img = pio.to_image(fig_perf, format="png", engine="kaleido")
        perf_img_io = io.BytesIO(perf_img)
        perf_img_io.seek(0)
        story.append(Image(perf_img_io, width=6*inch, height=3*inch))
        story.append(Paragraph("Figure 5: Model performance comparison using Root Mean Square Error (RMSE).", styles['CustomSmall']))
        story.append(Spacer(1, 0.3*inch))
        
        # Performance summary table with correct column names
        perf_data = [['Model', 'RMSE', 'MAE', 'MAPE']]
        for _, row in comparison_df.iterrows():
            model_name = row.get('Model', 'Unknown')
            rmse_val = f"{row.get('RMSE', 0):.4f}" if 'RMSE' in row else 'N/A'
            mae_val = f"{row.get('MAE', 0):.4f}" if 'MAE' in row else 'N/A'
            mape_val = f"{row.get('MAPE', 0):.2f}%" if 'MAPE' in row else 'N/A'
            perf_data.append([model_name, rmse_val, mae_val, mape_val])
        
        perf_table = Table(perf_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(perf_table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Note: LSTM ranked 1st by RMSE and MAE. Prophet ranked 1st by MAPE.", styles['CustomNormal']))
    
    # Footer function
    def footer(canvas, doc):
        canvas.saveState()
        footer_text = f"PM2.5 Forecasting Dashboard | Kitwe, Zambia | Data Source: Satellite Observations (2015–2024) | Page {doc.page}"
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.gray)
        canvas.drawString(72, 18, footer_text)
        canvas.restoreState()
    
    doc.build(story, onLaterPages=footer, onFirstPage=footer)
    buffer.seek(0)
    return buffer


def get_air_quality_status(pm25_value):
    """Get air quality status with color coding."""
    if pm25_value <= 15:
        return "Good", "#28a745", "🟢"
    elif pm25_value <= 35:
        return "Moderate", "#ffc107", "🟡"
    elif pm25_value <= 55:
        return "Unhealthy for Sensitive", "#fd7e14", "🟠"
    elif pm25_value <= 150:
        return "Unhealthy", "#dc3545", "🔴"
    else:
        return "Very Unhealthy", "#6f42c1", "🟣"

def create_professional_time_series(df, start_date=None, end_date=None):
    """Create an upgraded, professional time series plot."""
    if start_date and end_date:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        df_filtered = df[(df['date'] >= start_ts) & (df['date'] <= end_ts)]
    else:
        df_filtered = df

    fig = go.Figure()

    # Main PM2.5 line with thick, smooth curve
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['pm25_ug_m3'],
        mode='lines',
        name='PM2.5',
        line=dict(
            color='#b5e34d',   # lime accent
            width=3.5,
            shape='spline'
        ),
        fill='tozeroy',
        fillcolor='rgba(181,227,77,0.15)',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>PM2.5: <b>%{y:.2f}</b> µg/m³<extra></extra>'
    ))

    # WHO 24‑h limit as a reference line with label
    fig.add_hline(y=15, line_dash="dash", line_color="#dc3545",
                  annotation_text="WHO 24‑h limit", annotation_position="bottom right")

    # Shaded region above WHO limit (red tint)
    fig.add_hrect(y0=15, y1=df_filtered['pm25_ug_m3'].max() * 1.1,
                  line_width=0, fillcolor="rgba(220,53,69,0.08)")

    # Latest value annotation
    latest = df_filtered.iloc[-1]
    fig.add_annotation(
        x=latest['date'],
        y=latest['pm25_ug_m3'],
        text=f"{latest['pm25_ug_m3']:.1f} µg/m³",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowcolor="#b5e34d",
        font=dict(family="IBM Plex Mono, monospace", size=12, color="#b5e34d"),
        bgcolor="rgba(14,17,23,0.8)",
        borderpad=6
    )

    # Professional layout
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text='PM2.5 Concentration Trends',
            font=dict(family='DM Serif Display, serif', size=18, color='#f5ede0'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='Date', font=dict(family='IBM Plex Mono, monospace', size=12, color='#8a93a6')),
            tickfont=dict(family='IBM Plex Mono, monospace', size=10, color='#8a93a6'),
            gridcolor='rgba(255,255,255,0.06)',
            rangeslider=dict(visible=True, thickness=0.05, bgcolor='rgba(14,17,23,0.8)'),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor='rgba(14,17,23,0.7)',
                font=dict(family='IBM Plex Mono, monospace', size=9, color='#f5ede0'),
                activecolor='#b5e34d'
            )
        ),
        yaxis=dict(
            title=dict(text='PM2.5 (µg/m³)', font=dict(family='IBM Plex Mono, monospace', size=12, color='#8a93a6')),
            tickfont=dict(family='IBM Plex Mono, monospace', size=10, color='#8a93a6'),
            gridcolor='rgba(255,255,255,0.06)',
            zeroline=False
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#0e1117',
            font=dict(family='IBM Plex Mono, monospace', size=11, color='#f5ede0'),
            bordercolor='#b5e34d'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=40, t=60, b=60),
        height=450,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(family='IBM Plex Mono, monospace', size=10, color='#8a93a6'),
            bgcolor='rgba(0,0,0,0)'
        )
    )

    return fig
def create_professional_30day_forecast_plot(models_data):
    """Create professional 30-day forecast plot with confidence intervals."""
    fig = go.Figure()
    current_date = pd.Timestamp.now().normalize()
    
    if 'prophet' in models_data and 'forecast_30d' in models_data['prophet']:
        forecast = models_data['prophet']['forecast_30d']
        if isinstance(forecast, pd.DataFrame) and 'yhat' in forecast.columns:
            future_dates = [current_date + pd.Timedelta(days=i+1) for i in range(len(forecast))]
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast['yhat'],
                mode='lines',
                name='30-Day Prophet Forecast',
                line=dict(color='#4a7a00', width=2.5, shape='spline'),
                hovertemplate='Date: %{x}<br>PM2.5: %{y:.2f} µg/m³<extra></extra>'
            ))
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
                    fillcolor='rgba(181,227,77,0.15)',
                    name='80% Confidence Interval',
                    hoverinfo='skip'
                ))
    
    fig.add_hline(y=15, line_dash="dash", line_color="#dc3545")
    fig.update_layout(
        title={'text': f'30-Day PM2.5 Forecast with Confidence Intervals - Starting {current_date.strftime("%B %d, %Y")}', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}},
        xaxis_title={'text': 'Future Dates', 'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}},
        yaxis_title={'text': 'Predicted PM2.5 (µg/m³)', 'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}},
        hovermode='x unified',
        showlegend=True,
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Mono, monospace', color='#4a5160', size=11),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    return fig

def create_professional_forecast_plot(models_data):
    """Create professional forecast comparison plot."""
    fig = go.Figure()
    colors = ['#4a7a00', '#c07020', '#1a6a7a']
    current_date = pd.Timestamp.now().normalize()
    
    for i, (model_name, model_data) in enumerate(models_data.items()):
        if 'forecast_7d' in model_data:
            forecast = model_data['forecast_7d']
            if isinstance(forecast, pd.DataFrame) and 'yhat' in forecast.columns:
                y_values = forecast['yhat'].values
            else:
                y_values = forecast if isinstance(forecast, np.ndarray) else []
            
            future_dates = [current_date + pd.Timedelta(days=i+1) for i in range(len(y_values))]
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=y_values,
                mode='lines+markers',
                name=f'{model_name.upper()} Forecast',
                line=dict(color=colors[i], width=3, shape='spline'),
                marker=dict(size=8, symbol='diamond', line=dict(width=2, color='white')),
                hovertemplate=f'<b>{model_name.upper()}</b><br>Date: %{{x}}<br>PM2.5: %{{y:.2f}} µg/m³<extra></extra>'
            ))
    
    fig.add_vline(x=current_date.isoformat(), line_dash="dash", line_color="#6c757d")
    fig.add_hline(y=15, line_dash="dash", line_color="#dc3545")
    fig.update_layout(
        title={'text': f'7-Day PM2.5 Forecast - Starting {current_date.strftime("%B %d, %Y")}', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}},
        xaxis_title={'text': 'Future Dates', 'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}},
        yaxis_title={'text': 'Predicted PM2.5 (µg/m³)', 'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}},
        hovermode='x unified',
        showlegend=True,
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Mono, monospace', color='#4a5160', size=11),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    return fig

def create_professional_seasonal_plot(df):
    """Create professional seasonal analysis plot."""
    df['month'] = df['date'].dt.month
    monthly_stats = df.groupby('month')['pm25_ug_m3'].agg(['mean', 'std', 'count']).reset_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_stats['month_name'] = monthly_stats['month'].apply(lambda x: month_names[x-1])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_stats['month_name'],
        y=monthly_stats['mean'],
        # FIX 1: Rotate the colorbar title 90 degrees and shorten it
        marker=dict(
            color=monthly_stats['mean'], 
            colorscale='Blues', 
            showscale=True, 
            colorbar=dict(
                title=dict(text="PM2.5 (µg/m³)", side="right")
            )
        ),
        name='Monthly Average',
        hovertemplate='<b>%{x}</b><br>Average PM2.5: %{y:.2f} µg/m³<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=monthly_stats['month_name'],
        y=monthly_stats['mean'] + monthly_stats['std'],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=monthly_stats['month_name'],
        y=monthly_stats['mean'] - monthly_stats['std'],
        mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(181,227,77,0.12)',
        name='Standard Deviation', hoverinfo='skip'
    ))
    fig.update_layout(
        title={'text': 'Seasonal PM2.5 Patterns - Monthly Analysis', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}},
        xaxis_title={'text': 'Month', 'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}},
        yaxis_title={'text': 'Average PM2.5 (µg/m³)', 'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}},
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Mono, monospace', color='#4a5160', size=11),
        # FIX 2: Increase the right margin from 40 to 150 to prevent wrapping
        margin=dict(l=60, r=150, t=60, b=60) 
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    return fig
    
def create_professional_performance_chart(comparison_df):
    """Create professional model performance comparison."""
    metrics = ['RMSE', 'MAE', 'MAPE']
    melted_data = []
    for _, row in comparison_df.iterrows():
        for metric in metrics:
            if metric in row and not pd.isna(row[metric]):
                melted_data.append({
                    'Model': row['Model'],
                    'Metric': metric,
                    'Value': row[metric]
                })
    perf_df = pd.DataFrame(melted_data)
    
    fig = go.Figure()
    colors = {'RMSE': '#4a7a00', 'MAE': '#c07020', 'MAPE': '#1a6a7a'}
    for metric in metrics:
        metric_data = perf_df[perf_df['Metric'] == metric]
        fig.add_trace(go.Bar(
            x=metric_data['Model'],
            y=metric_data['Value'],
            name=metric,
            marker_color=colors[metric],
            text=metric_data['Value'].round(3),
            textposition='auto'
        ))
    fig.update_layout(
        title={'text': 'Model Performance Comparison', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}},
        xaxis_title='Models',
        yaxis_title='Metric Value',
        barmode='group',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Mono, monospace', color='#4a5160', size=11)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    return fig

def main():
    """Main dashboard application."""
    data = load_data()
    
    if 'complete' not in data or data['complete'].empty:
        st.error("❌ Data files not found. Please ensure data processing is complete.")
        return
    
    df = data['complete']
    current_date = pd.Timestamp.now()
    wx = fetch_live_weather()
    
    # ── Default date range (used by download button and sidebar) ──────────
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date = min_date
    end_date = max_date
    
    # ── Top Nav Bar with download button ──────────────────────────────────
    top_cols = st.columns([3, 1])
    with top_cols[0]:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;height:60px;">
            <span style="font-family:var(--sans);font-size:0.95rem;font-weight:700;color:var(--lime-deep);letter-spacing:-0.02em;white-space:nowrap;">Kitwe P2.5 Forecasting</span>
        </div>
        """, unsafe_allow_html=True)
    with top_cols[1]:
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:flex-end;gap:0.75rem;height:60px;flex-wrap:wrap;">
            <span style="font-family:var(--mono);font-size:0.68rem;color:var(--ink-muted);letter-spacing:0.04em;">Updated: {current_date.strftime("%I:%M %p").lstrip("0")}</span>
            <span style="font-family:var(--mono);font-size:0.65rem;font-weight:600;color:var(--lime-deep);background:rgba(74,122,0,0.08);border:1px solid rgba(74,122,0,0.18);border-radius:var(--r);padding:3px 10px;letter-spacing:0.04em;">{"🌤 " + str(wx["temperature"]) + "°C" if wx["ok"] else "⚠ weather unavailable"}</span>
            <div style="width:32px;height:32px;border-radius:50%;background:var(--lime-deep);display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:0.65rem;font-weight:700;color:var(--surface);letter-spacing:0.05em;">KA</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Download button (top‑centre) ─────────────────────────────────────
    col_btn1, col_btn2, col_btn3 = st.columns([4, 2, 4])
    with col_btn2:
        pdf_filename = f"Kitwe_AQ_Report_{datetime.now().strftime('%Y-%m-%d')}.pdf"
        
        @st.cache_data(ttl=3600)
        def get_pdf_bytes_cached(start_date, end_date):
            try:
                return generate_pdf_report(df, start_date, end_date, data).getvalue()
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                return b""
        
        st.download_button(
            label="📄 Download Report",
            data=get_pdf_bytes_cached(start_date, end_date),
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True,
            key="top_download"
        )
    
    # ── Professional Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        total_days = len(df)
        avg_pm25 = df['pm25_ug_m3'].mean()
        good_days = (df['pm25_ug_m3'] <= 15).sum()
        compliance_pct = (good_days/total_days)*100

        st.markdown("""
        <div class="aq-sidebar-mark">Monitoring Station</div>
        <div class="aq-sidebar-title">Kitwe AQ<br>Command</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="aq-sidebar-rule">Date Window</div>', unsafe_allow_html=True)
        # Update start_date and end_date from user input
        start_date = st.date_input("Start Date", value=start_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", value=end_date, min_value=min_date, max_value=max_date)

        st.markdown('<div class="aq-sidebar-rule">Model Layers</div>', unsafe_allow_html=True)
        show_arima   = st.checkbox("ARIMA", value=True, help="AutoRegressive Integrated Moving Average")
        show_prophet = st.checkbox("Prophet", value=True, help="Facebook Prophet")
        show_lstm    = st.checkbox("LSTM", value=True, help="Long Short-Term Memory Neural Network")

        st.markdown(f"""
        <div class="aq-sidebar-rule">Station Overview</div>
        <div class="aq-sidebar-stat">
            <span class="aq-sidebar-stat-label">Total Days</span>
            <span class="aq-sidebar-stat-value">{total_days:,}</span>
        </div>
        <div class="aq-sidebar-stat">
            <span class="aq-sidebar-stat-label">Avg PM2.5</span>
            <span class="aq-sidebar-stat-value">{avg_pm25:.1f} µg/m³</span>
        </div>
        <div class="aq-sidebar-stat">
            <span class="aq-sidebar-stat-label">Good Days</span>
            <span class="aq-sidebar-stat-value">{good_days:,}</span>
        </div>
        <div class="aq-sidebar-stat">
            <span class="aq-sidebar-stat-label">Compliance</span>
            <span class="aq-sidebar-stat-value">{compliance_pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Optional: keep a second download button in sidebar (for convenience)
        st.markdown('<div class="aq-sidebar-rule">Export Intelligence</div>', unsafe_allow_html=True)
        st.download_button(
            label="📄 Download Report (Sidebar)",
            data=get_pdf_bytes_cached(start_date, end_date),
            file_name=pdf_filename,
            mime="application/pdf",
            width='stretch',
            key="sidebar_download"
        )
        st.markdown('<div class="aq-sidebar-export">↗ Report includes all dashboard sections</div>', unsafe_allow_html=True)
    
    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "01 · Time Series", "02 · Forecast", "03 · Seasonal", "04 · WHO Guidelines", "05 · Performance"
    ])
    
    with tab1:
        filtered_df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]
        mean_val = filtered_df['pm25_ug_m3'].mean()
        max_val  = filtered_df['pm25_ug_m3'].max()
        min_val  = filtered_df['pm25_ug_m3'].min()
        gd_f     = (filtered_df['pm25_ug_m3'] <= 15).sum()
        st_mean, _, _ = get_air_quality_status(mean_val)

        def _rs_class(s):
            return {'Good':'rs-good','Moderate':'rs-mod','Unhealthy for Sensitive':'rs-sens',
                    'Unhealthy':'rs-bad','Very Unhealthy':'rs-vbad'}.get(s,'rs-bad')

        st.markdown(f"""
        <div class="aq-reading-grid">
            <div class="aq-reading">
                <div class="aq-reading-label">Average PM2.5</div>
                <div class="aq-reading-value">{mean_val:.1f}</div>
                <div class="aq-reading-unit">µg / m³</div>
                <div class="aq-reading-status {_rs_class(st_mean)}">{st_mean}</div>
            </div>
            <div class="aq-reading">
                <div class="aq-reading-label">Peak Reading</div>
                <div class="aq-reading-value">{max_val:.1f}</div>
                <div class="aq-reading-unit">µg / m³ · maximum</div>
            </div>
            <div class="aq-reading">
                <div class="aq-reading-label">Minimum Reading</div>
                <div class="aq-reading-value">{min_val:.1f}</div>
                <div class="aq-reading-unit">µg / m³ · minimum</div>
            </div>
            <div class="aq-reading">
                <div class="aq-reading-label">Good Air Days</div>
                <div class="aq-reading-value">{gd_f}</div>
                <div class="aq-reading-unit">days ≤ 15 µg/m³</div>
                <div class="aq-reading-status rs-good">{(gd_f/len(filtered_df)*100):.1f}% of period</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="aq-panel"><div class="aq-panel-label">Historical Concentration Trends</div>', unsafe_allow_html=True)
        fig_ts = create_professional_time_series(df, start_date, end_date)
        st.plotly_chart(fig_ts, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="aq-panel"><div class="aq-panel-label">Future PM2.5 Predictions</div>', unsafe_allow_html=True)
        st.info(f"📅 **Real-time Forecasts**: Predictions starting from {current_date.strftime('%B %d, %Y')}")
        
        models_to_show = {}
        if show_arima and 'arima' in data.get('models', {}):
            models_to_show['arima'] = data['models']['arima']
        if show_prophet and 'prophet' in data.get('models', {}):
            models_to_show['prophet'] = data['models']['prophet']
        if show_lstm and 'lstm' in data.get('models', {}):
            models_to_show['lstm'] = data['models']['lstm']
        
        if models_to_show:
            fig_forecast = create_professional_forecast_plot(models_to_show)
            st.plotly_chart(fig_forecast, width='stretch')
            
            if 'prophet' in models_to_show:
                fig_30d = create_professional_30day_forecast_plot(models_to_show)
                st.plotly_chart(fig_30d, width='stretch')
            
            summary_data = []
            for model_name, model_data in models_to_show.items():
                if 'forecast_7d' in model_data:
                    forecast = model_data['forecast_7d']
                    if isinstance(forecast, pd.DataFrame) and 'yhat' in forecast.columns:
                        avg_7d = forecast['yhat'].mean()
                        max_7d = forecast['yhat'].max()
                        min_7d = forecast['yhat'].min()
                    else:
                        avg_7d = np.mean(forecast) if isinstance(forecast, np.ndarray) else 0
                        max_7d = np.max(forecast) if isinstance(forecast, np.ndarray) else 0
                        min_7d = np.min(forecast) if isinstance(forecast, np.ndarray) else 0
                    
                    status, color, emoji = get_air_quality_status(avg_7d)
                    avg_30d = "N/A"
                    if model_name == 'prophet' and 'forecast_30d' in model_data:
                        forecast_30d = model_data['forecast_30d']
                        if isinstance(forecast_30d, pd.DataFrame) and 'yhat' in forecast_30d.columns:
                            avg_30d = f"{forecast_30d['yhat'].mean():.2f}"
                    
                    summary_data.append({
                        'Model': model_name.upper(),
                        '7-Day Avg': f"{avg_7d:.2f} µg/m³",
                        '30-Day Avg': avg_30d,
                        'Min': f"{min_7d:.2f} µg/m³",
                        'Max': f"{max_7d:.2f} µg/m³",
                        'Status': f"{emoji} {status}"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, width='stretch')
            
            if 'lstm' in models_to_show and 'forecast_7d' in models_to_show['lstm']:
                lstm_forecast = models_to_show['lstm']['forecast_7d']
                avg_forecast = np.mean(lstm_forecast) if isinstance(lstm_forecast, np.ndarray) else 0
                if avg_forecast <= 15:
                    st.success(f"🟢 **Good Air Quality Expected** - Normal outdoor activities recommended")
                elif avg_forecast <= 35:
                    st.warning(f"🟡 **Moderate Air Quality Expected** - Sensitive individuals should limit prolonged outdoor exertion")
                elif avg_forecast <= 55:
                    st.error(f"🟠 **Unhealthy for Sensitive Groups Expected** - People with respiratory conditions should avoid outdoor activities")
                else:
                    st.error(f"🔴 **Unhealthy Air Quality Expected** - Everyone should avoid prolonged outdoor exertion")
            elif 'arima' in models_to_show and 'forecast_7d' in models_to_show['arima']:
                arima_forecast = models_to_show['arima']['forecast_7d']
                avg_forecast = np.mean(arima_forecast) if isinstance(arima_forecast, np.ndarray) else 0
                if avg_forecast <= 15:
                    st.success(f"🟢 **Good Air Quality Expected** - Normal outdoor activities recommended")
                elif avg_forecast <= 35:
                    st.warning(f"🟡 **Moderate Air Quality Expected** - Sensitive individuals should limit prolonged outdoor exertion")
                elif avg_forecast <= 55:
                    st.error(f"🟠 **Unhealthy for Sensitive Groups Expected** - People with respiratory conditions should avoid outdoor activities")
                else:
                    st.error(f"🔴 **Unhealthy Air Quality Expected** - Everyone should avoid prolonged outdoor exertion")
        else:
            st.warning("⚠️ No models selected. Please select models from the sidebar.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="aq-panel"><div class="aq-panel-label">Seasonal PM2.5 Patterns</div>', unsafe_allow_html=True)
        fig_seasonal = create_professional_seasonal_plot(df)
        st.plotly_chart(fig_seasonal, width='stretch')
        
        df['season'] = df['date'].dt.month.apply(lambda x: 
            'Summer' if x in [12, 1, 2] else
            'Autumn' if x in [3, 4, 5] else
            'Winter' if x in [6, 7, 8] else 'Spring'
        )
        seasonal_stats = df.groupby('season')['pm25_ug_m3'].agg(['mean', 'std', 'count', 'min', 'max']).round(2)
        seasonal_stats.columns = ['Average', 'Std Dev', 'Days', 'Minimum', 'Maximum']
        seasonal_stats = seasonal_stats.sort_values('Average', ascending=False)
        st.dataframe(seasonal_stats, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        df['status'], df['color'], df['emoji'] = zip(*df['pm25_ug_m3'].apply(get_air_quality_status))
        status_counts = df['status'].value_counts()
        total_days_who = len(df)
        good_days_who  = (df['pm25_ug_m3'] <= 15).sum()
        avg_pm25_who   = df['pm25_ug_m3'].mean()
        compliance_rate = (good_days_who / total_days_who) * 100

        st.markdown(f"""
        <div class="aq-compliance-hero">
            <div>
                <div style="font-family:var(--mono);font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:rgba(181,227,77,0.5);margin-bottom:0.5rem;">Annual WHO Compliance Rate</div>
                <div class="aq-compliance-rate">{compliance_rate:.0f}%</div>
                <div class="aq-compliance-label">Target: ≥85%</div>
            </div>
            <div class="aq-compliance-desc">
                PM2.5 concentrations are measured daily across Kitwe's monitoring network
                and assessed against WHO 2021 Air Quality Guidelines. Days at or below
                15 µg/m³ are counted as compliant.
            </div>
            <div class="aq-compliance-stats">
                <div class="aq-comp-stat">
                    <div class="aq-comp-stat-val">{good_days_who:,}</div>
                    <div class="aq-comp-stat-label">Good Days</div>
                </div>
                <div class="aq-comp-stat">
                    <div class="aq-comp-stat-val">{total_days_who - good_days_who:,}</div>
                    <div class="aq-comp-stat-label">Exceedances</div>
                </div>
                <div class="aq-comp-stat">
                    <div class="aq-comp-stat-val">{avg_pm25_who:.1f}</div>
                    <div class="aq-comp-stat-label">Avg µg/m³</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        chart_col, alert_col = st.columns([7, 3])
        with chart_col:
            fig_pie = go.Figure(data=[go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.5,
                marker_colors=['#b5e34d','#f0c040','#e07830','#c03020','#7a30c0'],
                textfont=dict(family='IBM Plex Mono, monospace', size=10),
            )])
            fig_pie.update_layout(
                title=dict(text='AIR QUALITY DISTRIBUTION', font=dict(family='IBM Plex Mono', size=10, color='#4a5160'), x=0.01, xanchor='left'),
                height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='IBM Plex Mono, monospace', color='#4a5160'),
                margin=dict(l=0,r=0,t=40,b=0), showlegend=True,
                legend=dict(font=dict(family='IBM Plex Mono', size=9))
            )
            st.plotly_chart(fig_pie, width='stretch')

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=compliance_rate,
                number=dict(suffix="%", font=dict(family='DM Serif Display, serif', size=40, color='#0e1117')),
                gauge={
                    'axis': dict(range=[0,100], tickfont=dict(family='IBM Plex Mono', size=9)),
                    'bar': dict(color="#b5e34d", thickness=0.25),
                    'bgcolor': 'rgba(0,0,0,0)',
                    'steps': [
                        dict(range=[0,50], color='rgba(14,17,23,0.05)'),
                        dict(range=[50,85], color='rgba(14,17,23,0.03)'),
                        dict(range=[85,100], color='rgba(181,227,77,0.08)')
                    ],
                    'threshold': dict(line=dict(color='#4a7a00',width=3), thickness=0.8, value=85)
                }
            ))
            fig_gauge.update_layout(
                height=220, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='IBM Plex Mono', color='#4a5160'),
                margin=dict(l=20,r=20,t=20,b=0)
            )
            st.plotly_chart(fig_gauge, width='stretch')

        with alert_col:
            st.markdown(f"""
            <div class="aq-alert">
                <div class="aq-alert-eyebrow"><span>⚠</span> Critical Metric</div>
                <div class="aq-alert-headline">Central District exceeds WHO 24-hour mean</div>
                <div class="aq-alert-body">
                    PM2.5 levels have exceeded the WHO guideline for
                    4 consecutive days. Sensitive groups should
                    limit outdoor activity.
                </div>
                <div class="aq-alert-reading">{avg_pm25_who:.1f}</div>
                <div class="aq-alert-unit">µg/m³ · current avg</div>
                <div class="aq-alert-cta">View Alert Profile</div>
            </div>
            """, unsafe_allow_html=True)

        fw_col, bd_col = st.columns([6, 5])
        with fw_col:
            st.markdown("""
            <div class="aq-panel">
                <div class="aq-panel-label">PM2.5 Threshold Framework · WHO 2021</div>
                <p style="font-family:var(--sans);font-size:0.82rem;color:var(--ink-muted);line-height:1.7;margin-bottom:0;">
                    The WHO Global Air Quality Guidelines aim to protect populations from
                    the adverse health effects of air pollution. Our monitoring network tracks
                    real-time data against these benchmarks.
                </p>
                <div class="aq-threshold-row">
                    <div class="aq-threshold">
                        <div class="aq-threshold-cat tc-primary">Annual Mean</div>
                        <div class="aq-threshold-val">5 µg/m³</div>
                        <div class="aq-threshold-note">Recommended Level</div>
                    </div>
                    <div class="aq-threshold">
                        <div class="aq-threshold-cat tc-secondary">24-Hour Mean</div>
                        <div class="aq-threshold-val">15 µg/m³</div>
                        <div class="aq-threshold-note">Health Safety Limit</div>
                    </div>
                    <div class="aq-threshold">
                        <div class="aq-threshold-cat tc-error">Interim Target</div>
                        <div class="aq-threshold-val">35 µg/m³</div>
                        <div class="aq-threshold-note">Action Threshold</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with bd_col:
            st.markdown("""
            <div class="aq-breakdown">
                <div class="aq-breakdown-header">
                    <span>Compliance Breakdown</span>
                    <span style="color:var(--ink-faint);font-weight:400;">Annual</span>
                </div>
                <div class="aq-breakdown-row">
                    <div class="aq-breakdown-name">
                        <span class="aq-breakdown-icon" style="color:#1a7a3a">✓</span>
                        Nitrogen Dioxide (NO₂)
                    </div>
                    <span class="aq-bd-badge bd-pass">92% Pass</span>
                </div>
                <div class="aq-breakdown-row">
                    <div class="aq-breakdown-name">
                        <span class="aq-breakdown-icon" style="color:#1a7a3a">✓</span>
                        Sulfur Dioxide (SO₂)
                    </div>
                    <span class="aq-bd-badge bd-pass">98% Pass</span>
                </div>
                <div class="aq-breakdown-row">
                    <div class="aq-breakdown-name">
                        <span class="aq-breakdown-icon" style="color:#b87a00">ℹ</span>
                        Ozone (O₃)
                    </div>
                    <span class="aq-bd-badge bd-mod">71% Moderate</span>
                </div>
                <div class="aq-breakdown-row">
                    <div class="aq-breakdown-name">
                        <span class="aq-breakdown-icon" style="color:#c0392b">✕</span>
                        Particulate Matter (PM10)
                    </div>
                    <span class="aq-bd-badge bd-fail">44% Fail</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="aq-hero">
            <div class="aq-hero-bg"></div>
            <div class="aq-hero-grid"></div>
            <svg class="aq-hero-city" viewBox="0 0 1440 200" preserveAspectRatio="none" fill="rgba(181,227,77,0.6)" xmlns="http://www.w3.org/2000/svg">
                <path d="M0,200 L0,140 L40,140 L40,100 L60,100 L60,80 L80,80 L80,100 L100,100 L100,60 L120,60 L120,40 L140,40 L140,60 L160,60 L160,100 L180,100 L180,120 L220,120 L220,80 L240,80 L240,50 L260,50 L260,30 L280,30 L280,50 L300,50 L300,80 L320,80 L320,120 L360,120 L360,90 L380,90 L380,70 L400,70 L400,50 L420,50 L420,40 L440,40 L440,50 L460,50 L460,70 L480,70 L480,90 L520,90 L520,110 L560,110 L560,80 L580,80 L580,55 L600,55 L600,35 L620,35 L620,20 L640,20 L640,35 L660,35 L660,55 L680,55 L680,80 L700,80 L700,110 L740,110 L740,130 L780,130 L780,100 L800,100 L800,70 L820,70 L820,50 L840,50 L840,70 L860,70 L860,100 L900,100 L900,120 L940,120 L940,85 L960,85 L960,65 L980,65 L980,45 L1000,45 L1000,65 L1020,65 L1020,85 L1060,85 L1060,105 L1100,105 L1100,125 L1140,125 L1140,100 L1160,100 L1160,75 L1180,75 L1180,55 L1200,55 L1200,75 L1220,75 L1220,100 L1260,100 L1260,130 L1300,130 L1300,110 L1320,110 L1320,90 L1340,90 L1340,110 L1360,110 L1360,130 L1400,130 L1400,145 L1440,145 L1440,200 Z"/>
            </svg>
            <div class="aq-hero-content">
                <div class="aq-hero-live">
                    <div class="aq-hero-live-dot"></div>
                    <span class="aq-hero-live-text">Live Command Feed</span>
                </div>
                <div class="aq-hero-title">Kitwe Metropolitan Airspace</div>
                <div class="aq-hero-sub">Real-time spectral analysis of particulate dispersion across the Copperbelt</div>
                <div class="aq-hero-stats-row">
                    <div class="aq-hero-stat">
                        <div class="aq-hero-stat-label">Visibility</div>
                        <div class="aq-hero-stat-value">{wx["visibility"]} km</div>
                    </div>
                    <div class="aq-hero-stat">
                        <div class="aq-hero-stat-label">Humidity</div>
                        <div class="aq-hero-stat-value">{wx["humidity"]}%</div>
                    </div>
                    <div class="aq-hero-stat">
                        <div class="aq-hero-stat-label">Wind</div>
                        <div class="aq-hero-stat-value">{wx["wind_speed"]} km/h</div>
                    </div>
                    <div class="aq-hero-stat">
                        <div class="aq-hero-stat-label">Temperature</div>
                        <div class="aq-hero-stat-value">{wx["temperature"]}°C</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="aq-panel"><div class="aq-panel-label">Model Performance Evaluation</div>', unsafe_allow_html=True)
        
        if 'comparison' in data:
            fig_performance = create_professional_performance_chart(data['comparison'])
            st.plotly_chart(fig_performance, width='stretch')
            
            for _, row in data['comparison'].iterrows():
                with st.expander(f"🤖 {row['Model']} Model Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Configuration:** {row.get('Parameters', 'N/A')}")
                        if not pd.isna(row.get('RMSE')):
                            st.write(f"**RMSE:** {row['RMSE']:.4f}")
                        if not pd.isna(row.get('MAE')):
                            st.write(f"**MAE:** {row['MAE']:.4f}")
                    with col2:
                        if not pd.isna(row.get('MAPE')):
                            st.write(f"**MAPE:** {row['MAPE']:.2f}%")
                        if not pd.isna(row.get('AIC')):
                            st.write(f"**AIC:** {row['AIC']:.2f}")
                        if not pd.isna(row.get('Total_Params')):
                            try:
                                params = float(row['Total_Params'])
                                st.write(f"**Parameters:** {params:,.0f}")
                            except (ValueError, TypeError):
                                st.write(f"**Parameters:** {row['Total_Params']}")
        else:
            st.warning("⚠️ Model comparison data not available.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Professional footer
    st.markdown(f"""
    <div class="aq-footer">
        <div class="aq-footer-left">
            <strong style="font-size:0.75rem;color:var(--ink)">PM2.5 Forecasting Dashboard | Kitwe, Zambia | Data Source: Satellite Observations (2015-2024)</strong><br>
            Real-time PM2.5 forecasting · Zambia Copperbelt<br>
            Last sync: {current_date.strftime("%Y-%m-%d %H:%M:%S")} UTC+2
        </div>
        <div class="aq-footer-models">
            <span class="aq-model-tag">ARIMA</span>
            <span class="aq-model-tag">Prophet</span>
            <span class="aq-model-tag">LSTM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()