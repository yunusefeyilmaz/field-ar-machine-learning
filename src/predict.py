import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Base paths
base_dir = Path(__file__).resolve().parent.parent
output_path = base_dir / 'output'
models_path = base_dir / 'models'
weather_path = base_dir / 'data' / 'weather'

def predict_water_stress(bbox, weather_forecast, current_stress=None, days_ahead=7):
    """
    Predict water stress for a specific area for multiple days ahead.
    
    Args:
        bbox: Bounding box coordinates [lon_min, lat_min, lon_max, lat_max]
        weather_forecast: Weather forecast for the prediction period (can be daily arrays or means)
        current_stress: Current water stress value (optional)
        days_ahead: Number of days ahead to predict (default 7)
    
    Returns:
        Dictionary with predicted water stress values for each day
    """
    # Load model
    model_filename = f'water_stress_prediction_{days_ahead}days.joblib'
    model_path = models_path / model_filename
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train_model.py first to create the model.")
        return None
    model = load(model_path)
    print(f"Loaded model for {days_ahead}-day prediction.")

    lon_min, lat_min, lon_max, lat_max = bbox
    center_lon = (lon_min + lon_max) / 2
    center_lat = (lat_min + lat_max) / 2

    # Günlük tahminleri burada topla
    daily_results = {}
    today = datetime.now()
    for day in range(1, days_ahead + 1):
        predict_date = today + timedelta(days=day)
        current_date = predict_date.strftime('%Y-%m-%d')
        # Tarihsel özellikler
        day_of_year = predict_date.timetuple().tm_yday
        day_of_week = predict_date.weekday()
        month = predict_date.month
        quarter = (month - 1) // 3 + 1
        # Özellikler
        features = {
            'day_of_year': day_of_year,
            'day_of_week': day_of_week,
            'month': month,
            'quarter': quarter,
            'lat': center_lat,
            'lon': center_lon,
        }
        # Su stresi
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
                        (df['lat'] >= lat_min) &
                        (df['lat'] <= lat_max) &
                        (df['lon'] >= lon_min) &
                        (df['lon'] <= lon_max)
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
                    features['water_stress_mean'] = 0.5
                    features['water_stress_min'] = 0.2
                    features['water_stress_max'] = 0.8
                    features['coverage'] = 0.9
            except Exception as e:
                print(f"Error estimating historical water stress: {e}")
                features['water_stress_mean'] = 0.5
                features['water_stress_min'] = 0.2
                features['water_stress_max'] = 0.8
                features['coverage'] = 0.9
        # Hava durumu: Eğer weather_forecast'ta günlük array varsa, ilgili günü al
        for key, value in weather_forecast.items():
            if key == 'time':
                continue
            if isinstance(value, (list, np.ndarray)) and len(value) >= day:
                features[key] = value[day-1]
            else:
                features[key] = value
        # DataFrame oluştur
        input_df = pd.DataFrame([features])
        # Modelin beklediği sırada feature'lar
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        elif hasattr(model, 'named_steps') and 'scaler' in model.named_steps and hasattr(model.named_steps['scaler'], 'feature_names_in_'):
            expected_features = model.named_steps['scaler'].feature_names_in_
        else:
            expected_features = input_df.columns
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[list(expected_features)]
        # Tahmin
        predicted_stress = float(model.predict(input_df)[0])
        predicted_stress = max(0.0, min(1.0, predicted_stress))
        daily_results[f"day_{day}"] = {
            'date': current_date,
            'predicted_water_stress': round(predicted_stress, 3),
            'stress_category': get_stress_category(predicted_stress)
        }
    # Sonuç
    result = {
        'bbox': bbox,
        'center_coordinates': [center_lat, center_lon],
        'prediction_days': days_ahead,
        'daily_forecast': daily_results
    }
    return result

def get_stress_category(stress_value):
    """Convert numerical water stress to category."""
    if stress_value <= 0.1:
        return "No Stress"
    elif stress_value <= 0.3:
        return "Low Stress"
    elif stress_value <= 0.7:
        return "Moderate Stress"
    else:
        return "High Stress"

def load_weather_forecast(forecast_file=None):
    """
    Load weather forecast data from file or API.
    
    In a real application, this would likely call an API to get the forecast.
    For this example, we'll use sample data or a provided file.
    """
    if forecast_file and os.path.exists(forecast_file):
        with open(forecast_file, 'r') as f:
            forecast_data = json.load(f)
        return forecast_data
    
    # For demo purposes, use historical data from the weather file
    """ weather_file = weather_path / 'data.json'
    if os.path.exists(weather_file):
        with open(weather_file, 'r') as f:
            weather_data = json.load(f)
            
        # Get current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Find index of current date in weather data
        if current_date in weather_data['daily']['time']:
            idx = weather_data['daily']['time'].index(current_date)
            
            # Extract forecast for the next 7 days
            forecast = {}
            for param in weather_data['daily'].keys():
                if param == 'time':
                    forecast[param] = weather_data['daily'][param][idx:idx+7]
                elif idx < len(weather_data['daily'][param]):
                    # Use average of available values
                    forecast[param] = sum(weather_data['daily'][param][idx:idx+7]) / len(weather_data['daily'][param][idx:idx+7])
            
            return forecast """
    
    # If no weather data available, create dummy forecast
    print("Warning: Using dummy weather forecast data.")
    # Return dummy forecast as daily arrays, normalized to 0-1 range where appropriate
    days = 7
    return {
        'temperature_2m_max': [1] * days,  # Example: normalized between min/max expected (e.g., 0-50°C)
        'temperature_2m_mean': [0.6] * days,
        'precipitation_hours': [0.2] * days,  # Example: normalized (e.g., 0-10 hours)
        'wind_speed_10m_max': [0.3] * days,   # Example: normalized (e.g., 0-50 km/h)
        'relative_humidity_2m_max': [0.8] * days,  # Already 0-1
        'relative_humidity_2m_min': [0.4] * days,
        'relative_humidity_2m_mean': [0.6] * days,
        'soil_moisture_0_to_10cm_mean': [0.3] * days,  # Already 0-1
        'sunshine_duration': [0.33] * days,  # Example: 8h/24h
        'temperature_2m_min': [0.3] * days,
        'precipitation_sum': [0.25] * days,  # Example: normalized (e.g., 0-20mm)
        'shortwave_radiation_sum': [0.36] * days,  # Example: normalized (e.g., 0-50 MJ/m2)
        'et0_fao_evapotranspiration_sum': [0.2] * days,  # Example: normalized (e.g., 0-20mm)
        'time': [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days)]
    }

def main():
    """Main function to demonstrate water stress prediction."""
    bbox = [28.9123123, 40.1512313, 29.112312, 40.1123123]  # Example coordinates
    weather_forecast = load_weather_forecast()
    current_stress = 0.1  # Example: 0.3 if known
    days_ahead = 7
    result = predict_water_stress(bbox, weather_forecast, current_stress, days_ahead=days_ahead)
    if result:
        print("\n7 Günlük Su Stresi Tahmini:")
        for day, info in result['daily_forecast'].items():
            print(f"{day}: {info['date']} - Tahmin: {info['predicted_water_stress']:.3f} ({info['stress_category']})")
        # Save result to file
        output_file = output_path / 'prediction_result.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nPrediction result saved to {output_file}")

if __name__ == "__main__":
    main()
