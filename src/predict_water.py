import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from joblib import load
from pathlib import Path

# Base paths
base_dir = Path(__file__).resolve().parent
project_root = base_dir.parent
output_path = project_root / 'output'
models_path = project_root / 'models'
weather_path = project_root / 'data' / 'weather'

def predict_water_stress(bbox, weather_forecast, current_stress=None, days_ahead=7):
    model_filename = f'water_stress_prediction_{days_ahead}days.joblib'
    model_path = models_path / model_filename
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None

    model = load(model_path)

    lon_min, lat_min, lon_max, lat_max = bbox
    center_lon = (lon_min + lon_max) / 2
    center_lat = (lat_min + lat_max) / 2

    daily_results = {}
    # Tahmin tarihlerini weather_forecast['time']'dan al
    time_list = weather_forecast.get('time', None)
    for day in range(1, days_ahead + 1):
        if time_list and len(time_list) >= day:
            current_date = time_list[day-1]
            predict_date = datetime.strptime(current_date, '%Y-%m-%d')
        else:
            # fallback: bugünden itibaren
            predict_date = datetime.now() + timedelta(days=day)
            current_date = predict_date.strftime('%Y-%m-%d')
        day_of_year = predict_date.timetuple().tm_yday
        day_of_week = predict_date.weekday()
        month = predict_date.month
        quarter = (month - 1) // 3 + 1
        features = {
            'day_of_year': day_of_year,
            'day_of_week': day_of_week,
            'month': month,
            'quarter': quarter,
            'lat': center_lat,
            'lon': center_lon,
        }
        if current_stress is not None:
            features['water_stress_mean'] = current_stress
            features['water_stress_min'] = current_stress
            features['water_stress_max'] = current_stress
            features['coverage'] = 1.0
        else:
            try:
                dataset_path = output_path / 'water_stress_dataset.csv'
                if os.path.exists(dataset_path):
                    df = pd.read_csv(dataset_path)
                    nearby_data = df[
                        (df['month'] == month) &
                        (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
                        (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
                    ]
                    if len(nearby_data) > 0:
                        features['water_stress_mean'] = nearby_data['water_stress_mean'].mean()
                        features['water_stress_min'] = nearby_data['water_stress_min'].mean()
                        features['water_stress_max'] = nearby_data['water_stress_max'].mean()
                        features['coverage'] = nearby_data['coverage'].mean()
                    else:
                        features['water_stress_mean'] = df['water_stress_mean'].mean()
                        features['water_stress_min'] = df['water_stress_min'].mean()
                        features['water_stress_max'] = df['water_stress_max'].mean()
                        features['coverage'] = df['coverage'].mean()
                else:
                    features.update(dummy_stress())
            except Exception:
                features.update(dummy_stress())
        for key, value in weather_forecast.items():
            if key == 'time':
                continue
            if isinstance(value, (list, np.ndarray)) and len(value) >= day:
                features[key] = value[day - 1]
            else:
                features[key] = value
        input_df = pd.DataFrame([features])
        expected_features = (
            model.feature_names_in_
            if hasattr(model, 'feature_names_in_')
            else input_df.columns
        )
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[list(expected_features)]
        predicted_stress = float(model.predict(input_df)[0])
        predicted_stress = max(0.0, min(1.0, predicted_stress))
        daily_results[f"day_{day}"] = {
            'date': current_date,
            'predicted_water_stress': round(predicted_stress, 3),
            'stress_category': get_stress_category(predicted_stress)
        }
    return {
        'bbox': bbox,
        'center_coordinates': [center_lat, center_lon],
        'prediction_days': days_ahead,
        'daily_forecast': daily_results
    }

def dummy_stress():
    return {
        'water_stress_mean': 0.5,
        'water_stress_min': 0.2,
        'water_stress_max': 0.8,
        'coverage': 0.9
    }

def get_stress_category(stress_value):
    if stress_value <= 0.1:
        return "No Stress"
    elif stress_value <= 0.3:
        return "Low Stress"
    elif stress_value <= 0.7:
        return "Moderate Stress"
    else:
        return "High Stress"
