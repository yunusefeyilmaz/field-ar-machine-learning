import os
import sys
from pathlib import Path
import argparse

# Ensure the src directory is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    """Run the complete water stress prediction pipeline."""
    parser = argparse.ArgumentParser(description='Water Stress Prediction Pipeline')
    parser.add_argument('--build-dataset', action='store_true', help='Build dataset from satellite and weather data')
    parser.add_argument('--train-model', action='store_true', help='Train prediction model')
    parser.add_argument('--predict', action='store_true', help='Make predictions using trained model')
    parser.add_argument('--days', type=int, default=7, help='Number of days ahead to predict')
    parser.add_argument('--bbox', type=str, help='Bounding box coordinates (lon_min,lat_min,lon_max,lat_max)')
    parser.add_argument('--current-stress', type=float, help='Current water stress value if known')
    
    args = parser.parse_args()
      # If no arguments provided, run all steps
    if not (args.build_dataset or args.train_model or args.predict):
        args.build_dataset = True
        args.train_model = True
        args.predict = True
    
    # Build dataset
    if args.build_dataset:
        print("\n===== Building Dataset =====")
        print(f"Error with robust dataset builder: {e}")
        print("Falling back to original dataset builder...")
        from build_dataset import build_dataset

        build_dataset()
    
    # Train model
    if args.train_model:
        print("\n===== Training Model =====")
        from train_model import train_water_stress_model
        train_water_stress_model(prediction_days=args.days)
    
    # Make predictions
    if args.predict:
        print("\n===== Making Predictions =====")
        from predict import predict_water_stress, load_weather_forecast
        
        # Parse bbox if provided
        bbox = None
        if args.bbox:
            try:
                bbox = [float(x) for x in args.bbox.split(',')]
                if len(bbox) != 4:
                    raise ValueError("Bounding box must have 4 values: lon_min,lat_min,lon_max,lat_max")
            except Exception as e:
                print(f"Error parsing bbox: {e}")
                print("Using default bbox from config.py instead.")
        
        # If bbox not provided, use default from config
        if not bbox:
            from config import BBOX_FULL
            bbox = BBOX_FULL
            
        # Load weather forecast
        weather_forecast = load_weather_forecast()
        
        # Make prediction
        result = predict_water_stress(
            bbox=bbox, 
            weather_forecast=weather_forecast,
            current_stress=args.current_stress,
            days_ahead=args.days
        )
        
        if result:
            print("\nWater Stress Prediction Results:")
            print(f"Current date: {result['current_date']}")
            print(f"Prediction for: {result['prediction_date']} ({args.days} days ahead)")
            print(f"Area coordinates (bbox): {result['bbox']}")
            print(f"Predicted water stress: {result['predicted_water_stress']:.3f}")
            print(f"Stress category: {result['stress_category']}")

if __name__ == "__main__":
    main()
