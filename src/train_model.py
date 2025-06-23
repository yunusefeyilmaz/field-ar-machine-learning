import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Base paths
base_dir = Path(__file__).resolve().parent.parent
output_path = base_dir / 'output'
models_path = base_dir / 'models'

# Ensure models directory exists
os.makedirs(models_path, exist_ok=True)

def train_water_stress_model(prediction_days=7):
    print(f"Training model to predict water stress {prediction_days} days in the future...")
    
    # Load dataset
    dataset_path = output_path / 'water_stress_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run build_dataset.py first to generate the dataset.")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} rows and {df.columns.size} columns.")
    
    # Create target variable: water stress X days later for each grid cell
    print("Preparing features and target variables...")
    df = create_future_target(df, prediction_days)
    
    # Drop rows without target values
    df = df.dropna(subset=['future_water_stress'])
    print(f"Dataset size after creating future target: {len(df)} rows")
    
    if len(df) == 0:
        print("Error: No valid samples after creating future target.")
        return
    
    # Define features and target
    feature_columns = [
        # Date features
        'day_of_year', 'day_of_week', 'month', 'quarter',
        
        # Location features
        'lat', 'lon',
        
        # Current water stress
        'water_stress_mean', 'water_stress_min', 'water_stress_max', 'coverage',
        
        # Weather features - exclude 'time'
        'temperature_2m_max', 'temperature_2m_mean', 'temperature_2m_min',
        'precipitation_hours', 'precipitation_sum',
        'wind_speed_10m_max',
        'relative_humidity_2m_max', 'relative_humidity_2m_min', 'relative_humidity_2m_mean',
        'soil_moisture_0_to_10cm_mean', 'sunshine_duration',
        'shortwave_radiation_sum', 'et0_fao_evapotranspiration_sum'
    ]
    
    # Make sure all feature columns exist in the dataset
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    # Define target variable
    target_column = 'future_water_stress'
    
    # Split into features and target
    X = df[feature_columns]
    y = df[target_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42))
    ])

    # Define model parameters for grid search (RandomForest + GradientBoosting)
    param_grid = [
        {
            'model': [RandomForestRegressor(random_state=42)],
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5, 10]
        },
        {
            'model': [GradientBoostingRegressor(random_state=42)],
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 10],
            'model__learning_rate': [0.01, 0.1, 0.2]
        }
    ]

    # Use TimeSeriesSplit for time-dependent data
    tscv = TimeSeriesSplit(n_splits=5)

    # Train model with grid search
    print("Training model with grid search cross-validation (RandomForest & GradientBoosting)...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=tscv,
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best model parameters: {grid_search.best_params_}")
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Overfitting check
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    print("\nModel Performance:")
    print(f"Train RMSE: {train_rmse:.4f} | Test RMSE: {rmse:.4f}")
    print(f"Train MAE: {train_mae:.4f} | Test MAE: {mae:.4f}")
    print(f"Train R²: {train_r2:.4f} | Test R²: {r2:.4f}")
    # Plot actual vs predicted for train and test
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_train_pred, alpha=0.2, label='Train')
    plt.scatter(y_test, y_pred, alpha=0.3, label='Test')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual water stress')
    plt.ylabel('Predicted water stress')
    plt.title(f'Actual vs Predicted water stress ({prediction_days}-day forecast)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f'actual_vs_predicted_train_test_{prediction_days}days.png')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.named_steps['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Save model
    model_filename = f'water_stress_prediction_{prediction_days}days.joblib'
    model_path = models_path / model_filename
    dump(best_model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title(f'Feature importance for {prediction_days}-day water stress prediction')
    plt.tight_layout()
    plt.savefig(output_path / f'feature_importance_{prediction_days}days.png')
    
    # Save actual vs predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual water stress')
    plt.ylabel('Predicted water stress')
    plt.title(f'Actual vs Predicted water stress ({prediction_days}-day forecast)')
    plt.tight_layout()
    plt.savefig(output_path / f'actual_vs_predicted_{prediction_days}days.png')
    
    # Save model metadata and performance metrics
    model_info = {
        'prediction_days': prediction_days,
        'features': feature_columns,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'best_params': grid_search.best_params_,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    model_info_df = pd.DataFrame([model_info])
    model_info_df.to_json(output_path / f'model_info_{prediction_days}days.json', orient='records')
    
    return best_model

def create_future_target(df, days_ahead=7):
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Initialize target column
    df_copy['future_water_stress'] = np.nan
    
    # Group by grid cell coordinates
    for (lat, lon), group in df_copy.groupby(['lat', 'lon']):
        # Sort by date
        group = group.sort_values('date')
        dates = group['date'].tolist()
        
        # For each date, find the date X days ahead
        for i, date in enumerate(dates):
            if i + days_ahead < len(dates):
                future_date = dates[i + days_ahead]
                future_stress = group[group['date'] == future_date]['water_stress_mean'].values[0]
                df_copy.loc[(df_copy['date'] == date) & 
                           (df_copy['lat'] == lat) & 
                           (df_copy['lon'] == lon), 'future_water_stress'] = future_stress
    
    return df_copy

def predict_future_stress(lat, lon, current_date, weather_forecast, current_stress=None, days_ahead=7):
    # Load the model
    model_filename = f'water_stress_prediction_{days_ahead}days.joblib'
    model_path = models_path / model_filename
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    model = load(model_path)
    
    # Prepare input features
    current_date_obj = datetime.strptime(current_date, '%Y-%m-%d')
    day_of_year = current_date_obj.timetuple().tm_yday
    day_of_week = current_date_obj.weekday()
    month = current_date_obj.month
    quarter = (month - 1) // 3 + 1
    
    features = {
        'day_of_year': day_of_year,
        'day_of_week': day_of_week,
        'month': month,
        'quarter': quarter,
        'lat': lat,
        'lon': lon
    }
    
    # Add current water stress if available
    if current_stress is not None:
        features['water_stress_mean'] = current_stress
        features['water_stress_min'] = current_stress
        features['water_stress_max'] = current_stress
        features['coverage'] = 1.0
    else:
        # Get average stress from historical data for this location and time of year
        # (This would require additional implementation)
        pass
    
    # Add weather forecast
    for key, value in weather_forecast.items():
        if key != 'time':
            features[key] = value
    
    # Create DataFrame with one row
    input_df = pd.DataFrame([features])
    
    # Make prediction
    predicted_stress = model.predict(input_df)[0]
    
    return predicted_stress

if __name__ == "__main__":
    # Train model to predict water stress 7 days in the future
    train_water_stress_model(prediction_days=7)
