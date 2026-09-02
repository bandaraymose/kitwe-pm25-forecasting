"""
Prophet model implementation for PM2.5 forecasting.

This module provides:
- Prophet model with yearly and weekly seasonality
- Zambian holidays as special events (15 holidays × 10 years = 150 events)
- Probabilistic forecasting with uncertainty intervals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet not available")


def get_zambian_holidays() -> pd.DataFrame:
    """
    Get Zambian public holidays for Prophet modeling.
    Covers 15 distinct public holidays across 10 years (2015-2024) = 150 events.

    Holidays included:
        1.  New Year's Day (1 January)
        2.  International Women's Day (8 March)
        3.  Youth Day (12 March)
        4.  Good Friday (variable)
        5.  Holy Saturday (variable-day before Easter Sunday)
        6.  Easter Monday (variable)
        7.  Labour Day (1 May)
        8.  African Freedom Day (25 May)
        9.  Heroes Day (first Monday in July)
        10. Unity Day (Tuesday after Heroes Day)
        11. Farmers Day (first Monday in August)
        12. National Prayer Day (18 October)
        13. Independence Day (24 October)
        14. Christmas Day (25 December)
        15. Boxing Day (26 December)

    Returns:
        DataFrame with holidays formatted for Prophet
    """
    holidays_data = [
        # 1. New Year's Day (1 January)
        {'holiday': 'new_years_day', 'ds': pd.to_datetime('2015-01-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'new_years_day', 'ds': pd.to_datetime('2016-01-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'new_years_day', 'ds': pd.to_datetime('2017-01-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'new_years_day', 'ds': pd.to_datetime('2018-01-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'new_years_day', 'ds': pd.to_datetime('2019-01-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'new_years_day', 'ds': pd.to_datetime('2020-01-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'new_years_day', 'ds': pd.to_datetime('2021-01-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'new_years_day', 'ds': pd.to_datetime('2022-01-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'new_years_day', 'ds': pd.to_datetime('2023-01-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'new_years_day', 'ds': pd.to_datetime('2024-01-01'), 'lower_window': 0, 'upper_window': 1},

        # 2. International Women's Day (8 March)
        {'holiday': 'womens_day', 'ds': pd.to_datetime('2015-03-08'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'womens_day', 'ds': pd.to_datetime('2016-03-08'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'womens_day', 'ds': pd.to_datetime('2017-03-08'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'womens_day', 'ds': pd.to_datetime('2018-03-08'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'womens_day', 'ds': pd.to_datetime('2019-03-08'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'womens_day', 'ds': pd.to_datetime('2020-03-08'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'womens_day', 'ds': pd.to_datetime('2021-03-08'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'womens_day', 'ds': pd.to_datetime('2022-03-08'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'womens_day', 'ds': pd.to_datetime('2023-03-08'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'womens_day', 'ds': pd.to_datetime('2024-03-08'), 'lower_window': 0, 'upper_window': 0},

        # 3. Youth Day (12 March)
        {'holiday': 'youth_day', 'ds': pd.to_datetime('2015-03-12'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'youth_day', 'ds': pd.to_datetime('2016-03-12'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'youth_day', 'ds': pd.to_datetime('2017-03-12'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'youth_day', 'ds': pd.to_datetime('2018-03-12'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'youth_day', 'ds': pd.to_datetime('2019-03-12'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'youth_day', 'ds': pd.to_datetime('2020-03-12'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'youth_day', 'ds': pd.to_datetime('2021-03-12'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'youth_day', 'ds': pd.to_datetime('2022-03-12'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'youth_day', 'ds': pd.to_datetime('2023-03-12'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'youth_day', 'ds': pd.to_datetime('2024-03-12'), 'lower_window': 0, 'upper_window': 0},

        # 4. Good Friday (variable dates)
        {'holiday': 'good_friday', 'ds': pd.to_datetime('2015-04-03'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'good_friday', 'ds': pd.to_datetime('2016-03-25'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'good_friday', 'ds': pd.to_datetime('2017-04-14'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'good_friday', 'ds': pd.to_datetime('2018-03-30'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'good_friday', 'ds': pd.to_datetime('2019-04-19'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'good_friday', 'ds': pd.to_datetime('2020-04-10'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'good_friday', 'ds': pd.to_datetime('2021-04-02'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'good_friday', 'ds': pd.to_datetime('2022-04-15'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'good_friday', 'ds': pd.to_datetime('2023-04-07'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'good_friday', 'ds': pd.to_datetime('2024-03-29'), 'lower_window': 0, 'upper_window': 0},

        # 5. Holy Saturday (day before Easter Sunday)
        {'holiday': 'holy_saturday', 'ds': pd.to_datetime('2015-04-04'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'holy_saturday', 'ds': pd.to_datetime('2016-03-26'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'holy_saturday', 'ds': pd.to_datetime('2017-04-15'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'holy_saturday', 'ds': pd.to_datetime('2018-03-31'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'holy_saturday', 'ds': pd.to_datetime('2019-04-20'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'holy_saturday', 'ds': pd.to_datetime('2020-04-11'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'holy_saturday', 'ds': pd.to_datetime('2021-04-03'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'holy_saturday', 'ds': pd.to_datetime('2022-04-16'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'holy_saturday', 'ds': pd.to_datetime('2023-04-08'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'holy_saturday', 'ds': pd.to_datetime('2024-03-30'), 'lower_window': 0, 'upper_window': 0},

        # 6. Easter Monday (variable dates)
        {'holiday': 'easter_monday', 'ds': pd.to_datetime('2015-04-06'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'easter_monday', 'ds': pd.to_datetime('2016-03-28'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'easter_monday', 'ds': pd.to_datetime('2017-04-17'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'easter_monday', 'ds': pd.to_datetime('2018-04-02'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'easter_monday', 'ds': pd.to_datetime('2019-04-22'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'easter_monday', 'ds': pd.to_datetime('2020-04-13'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'easter_monday', 'ds': pd.to_datetime('2021-04-05'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'easter_monday', 'ds': pd.to_datetime('2022-04-18'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'easter_monday', 'ds': pd.to_datetime('2023-04-10'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'easter_monday', 'ds': pd.to_datetime('2024-04-01'), 'lower_window': 0, 'upper_window': 1},

        # 7. Labour Day (1 May)
        {'holiday': 'labour_day', 'ds': pd.to_datetime('2015-05-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'labour_day', 'ds': pd.to_datetime('2016-05-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'labour_day', 'ds': pd.to_datetime('2017-05-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'labour_day', 'ds': pd.to_datetime('2018-05-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'labour_day', 'ds': pd.to_datetime('2019-05-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'labour_day', 'ds': pd.to_datetime('2020-05-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'labour_day', 'ds': pd.to_datetime('2021-05-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'labour_day', 'ds': pd.to_datetime('2022-05-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'labour_day', 'ds': pd.to_datetime('2023-05-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'labour_day', 'ds': pd.to_datetime('2024-05-01'), 'lower_window': 0, 'upper_window': 1},

        # 8. African Freedom Day (25 May)
        {'holiday': 'african_freedom_day', 'ds': pd.to_datetime('2015-05-25'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'african_freedom_day', 'ds': pd.to_datetime('2016-05-25'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'african_freedom_day', 'ds': pd.to_datetime('2017-05-25'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'african_freedom_day', 'ds': pd.to_datetime('2018-05-25'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'african_freedom_day', 'ds': pd.to_datetime('2019-05-25'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'african_freedom_day', 'ds': pd.to_datetime('2020-05-25'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'african_freedom_day', 'ds': pd.to_datetime('2021-05-25'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'african_freedom_day', 'ds': pd.to_datetime('2022-05-25'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'african_freedom_day', 'ds': pd.to_datetime('2023-05-25'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'african_freedom_day', 'ds': pd.to_datetime('2024-05-25'), 'lower_window': 0, 'upper_window': 1},

        # 9. Heroes Day (first Monday in July)
        {'holiday': 'heroes_day', 'ds': pd.to_datetime('2015-07-06'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'heroes_day', 'ds': pd.to_datetime('2016-07-04'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'heroes_day', 'ds': pd.to_datetime('2017-07-03'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'heroes_day', 'ds': pd.to_datetime('2018-07-02'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'heroes_day', 'ds': pd.to_datetime('2019-07-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'heroes_day', 'ds': pd.to_datetime('2020-07-06'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'heroes_day', 'ds': pd.to_datetime('2021-07-05'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'heroes_day', 'ds': pd.to_datetime('2022-07-04'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'heroes_day', 'ds': pd.to_datetime('2023-07-03'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'heroes_day', 'ds': pd.to_datetime('2024-07-01'), 'lower_window': 0, 'upper_window': 1},

        # 10. Unity Day (Tuesday after Heroes Day)
        {'holiday': 'unity_day', 'ds': pd.to_datetime('2015-07-07'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'unity_day', 'ds': pd.to_datetime('2016-07-05'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'unity_day', 'ds': pd.to_datetime('2017-07-04'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'unity_day', 'ds': pd.to_datetime('2018-07-03'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'unity_day', 'ds': pd.to_datetime('2019-07-02'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'unity_day', 'ds': pd.to_datetime('2020-07-07'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'unity_day', 'ds': pd.to_datetime('2021-07-06'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'unity_day', 'ds': pd.to_datetime('2022-07-05'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'unity_day', 'ds': pd.to_datetime('2023-07-04'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'unity_day', 'ds': pd.to_datetime('2024-07-02'), 'lower_window': 0, 'upper_window': 1},

        # 11. Farmers Day (first Monday in August)
        {'holiday': 'farmers_day', 'ds': pd.to_datetime('2015-08-03'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'farmers_day', 'ds': pd.to_datetime('2016-08-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'farmers_day', 'ds': pd.to_datetime('2017-08-07'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'farmers_day', 'ds': pd.to_datetime('2018-08-06'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'farmers_day', 'ds': pd.to_datetime('2019-08-05'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'farmers_day', 'ds': pd.to_datetime('2020-08-03'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'farmers_day', 'ds': pd.to_datetime('2021-08-02'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'farmers_day', 'ds': pd.to_datetime('2022-08-01'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'farmers_day', 'ds': pd.to_datetime('2023-08-07'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'farmers_day', 'ds': pd.to_datetime('2024-08-05'), 'lower_window': 0, 'upper_window': 1},

        # 12. National Prayer Day (18 October)
        {'holiday': 'national_prayer_day', 'ds': pd.to_datetime('2015-10-18'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'national_prayer_day', 'ds': pd.to_datetime('2016-10-18'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'national_prayer_day', 'ds': pd.to_datetime('2017-10-18'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'national_prayer_day', 'ds': pd.to_datetime('2018-10-18'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'national_prayer_day', 'ds': pd.to_datetime('2019-10-18'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'national_prayer_day', 'ds': pd.to_datetime('2020-10-18'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'national_prayer_day', 'ds': pd.to_datetime('2021-10-18'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'national_prayer_day', 'ds': pd.to_datetime('2022-10-18'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'national_prayer_day', 'ds': pd.to_datetime('2023-10-18'), 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'national_prayer_day', 'ds': pd.to_datetime('2024-10-18'), 'lower_window': 0, 'upper_window': 0},

        # 13. Independence Day (24 October)
        {'holiday': 'independence_day', 'ds': pd.to_datetime('2015-10-24'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'independence_day', 'ds': pd.to_datetime('2016-10-24'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'independence_day', 'ds': pd.to_datetime('2017-10-24'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'independence_day', 'ds': pd.to_datetime('2018-10-24'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'independence_day', 'ds': pd.to_datetime('2019-10-24'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'independence_day', 'ds': pd.to_datetime('2020-10-24'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'independence_day', 'ds': pd.to_datetime('2021-10-24'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'independence_day', 'ds': pd.to_datetime('2022-10-24'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'independence_day', 'ds': pd.to_datetime('2023-10-24'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'independence_day', 'ds': pd.to_datetime('2024-10-24'), 'lower_window': 0, 'upper_window': 1},

        # 14. Christmas Day (25 December)
        {'holiday': 'christmas_day', 'ds': pd.to_datetime('2015-12-25'), 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'christmas_day', 'ds': pd.to_datetime('2016-12-25'), 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'christmas_day', 'ds': pd.to_datetime('2017-12-25'), 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'christmas_day', 'ds': pd.to_datetime('2018-12-25'), 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'christmas_day', 'ds': pd.to_datetime('2019-12-25'), 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'christmas_day', 'ds': pd.to_datetime('2020-12-25'), 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'christmas_day', 'ds': pd.to_datetime('2021-12-25'), 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'christmas_day', 'ds': pd.to_datetime('2022-12-25'), 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'christmas_day', 'ds': pd.to_datetime('2023-12-25'), 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'christmas_day', 'ds': pd.to_datetime('2024-12-25'), 'lower_window': -1, 'upper_window': 1},

        # 15. Boxing Day (26 December)
        {'holiday': 'boxing_day', 'ds': pd.to_datetime('2015-12-26'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'boxing_day', 'ds': pd.to_datetime('2016-12-26'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'boxing_day', 'ds': pd.to_datetime('2017-12-26'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'boxing_day', 'ds': pd.to_datetime('2018-12-26'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'boxing_day', 'ds': pd.to_datetime('2019-12-26'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'boxing_day', 'ds': pd.to_datetime('2020-12-26'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'boxing_day', 'ds': pd.to_datetime('2021-12-26'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'boxing_day', 'ds': pd.to_datetime('2022-12-26'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'boxing_day', 'ds': pd.to_datetime('2023-12-26'), 'lower_window': 0, 'upper_window': 1},
        {'holiday': 'boxing_day', 'ds': pd.to_datetime('2024-12-26'), 'lower_window': 0, 'upper_window': 1},
    ]

    return pd.DataFrame(holidays_data)


def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    prophet_df = df[['date', 'pm25_ug_m3']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    return prophet_df


def train_prophet_model(train_df: pd.DataFrame, holidays: pd.DataFrame = None) -> Prophet:
    if not PROPHET_AVAILABLE:
        return None

    print("🔮 Training Prophet model...")

    prophet_train = prepare_prophet_data(train_df)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        mcmc_samples=0,
        interval_width=0.8,
        uncertainty_samples=1000
    )

    model.add_seasonality(
        name='annual_southern',
        period=365.25,
        fourier_order=10
    )

    model.fit(prophet_train)
    print("   ✅ Prophet model trained successfully")
    return model


def evaluate_prophet_model(model: Prophet, test_df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    if not PROPHET_AVAILABLE or model is None:
        return {}, pd.DataFrame()

    print("📈 Evaluating Prophet model...")

    prophet_test = prepare_prophet_data(test_df)
    forecast = model.predict(prophet_test)

    actual = prophet_test['y'].values
    predicted = forecast['yhat'].values

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    metrics = {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape}

    print(f"   📊 RMSE: {rmse:.4f}")
    print(f"   📊 MAE:  {mae:.4f}")
    print(f"   📊 MAPE: {mape:.2f}%")

    return metrics, forecast


def generate_prophet_forecasts(model: Prophet, periods: int = 30) -> pd.DataFrame:
    if not PROPHET_AVAILABLE or model is None:
        return pd.DataFrame()

    print(f"🔮 Generating {periods}-day Prophet forecast...")
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    forecast_only = forecast.tail(periods)
    print(f"   ✅ Generated {len(forecast_only)} forecasts")
    return forecast_only


def prophet_pipeline(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    print("🚀 Starting Prophet modeling pipeline...")

    holidays = get_zambian_holidays()
    print(f"   📅 Added {len(holidays)} Zambian holiday events")  # Should print 150

    full_train = pd.concat([train_df, val_df])
    model = train_prophet_model(full_train, holidays)
    metrics, predictions = evaluate_prophet_model(model, test_df)
    forecast_7d = generate_prophet_forecasts(model, periods=7)
    forecast_30d = generate_prophet_forecasts(model, periods=30)

    results = {
        'model': model,
        'metrics': metrics,
        'predictions': predictions,
        'forecast_7d': forecast_7d,
        'forecast_30d': forecast_30d,
        'holidays': holidays
    }

    return results


if __name__ == "__main__":
    from pathlib import Path
    import pickle

    train_path = Path("data/processed/train_2015_2024.csv")
    val_path = Path("data/processed/validation_2015_2024.csv")
    test_path = Path("data/processed/test_2015_2024.csv")

    if all(p.exists() for p in [train_path, val_path, test_path]):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        results = prophet_pipeline(train_df, val_df, test_df)

        print("\n🎉 Prophet modeling complete!")
        print(f"📊 Test RMSE: {results['metrics']['rmse']:.4f}")
        print(f"📊 Test MAE:  {results['metrics']['mae']:.4f}")
        print(f"📊 Test MAPE: {results['metrics']['mape']:.2f}%")

        model_path = Path("models/prophet_model.pkl")
        model_path.parent.mkdir(exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"💾 Model saved to: {model_path}")
    else:
        print("❌ Data files not found")
