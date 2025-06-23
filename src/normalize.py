""" 
date,day_of_year,day_of_week,month,quarter,water_stress_mean,water_stress_min,water_stress_max,coverage,lat,lon,temperature_2m_max,temperature_2m_mean,precipitation_hours,wind_speed_10m_max,relative_humidity_2m_max,relative_humidity_2m_min,relative_humidity_2m_mean,soil_moisture_0_to_10cm_mean,sunshine_duration,temperature_2m_min,precipitation_sum,shortwave_radiation_sum,et0_fao_evapotranspiration_sum

burdaki degerleri 
date,day_of_year,day_of_week,month,quarter,water_stress_mean,water_stress_min,water_stress_max,coverage,lat,lon
bunlar haric normalize et yeni dataset olustur

 """ 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

# Base paths
base_dir = script_dir.parent
output_path = base_dir / 'output'

def normalize_dataset(input_file, output_file):
    # Load dataset
    df = pd.read_csv(input_file)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    
    # Check if the dataset is empty
    if df.empty:
        print("Warning: The dataset is empty. No normalization will be performed.")
        return
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Normalize specified columns
    columns_to_normalize = ['temperature_2m_max', 'temperature_2m_mean',
        'precipitation_hours', 'wind_speed_10m_max', 'relative_humidity_2m_max',
        'relative_humidity_2m_min', 'relative_humidity_2m_mean',
        'soil_moisture_0_to_10cm_mean', 'sunshine_duration', 
        'temperature_2m_min', 'precipitation_sum', 
        'shortwave_radiation_sum', 'et0_fao_evapotranspiration_sum'
    ]
    # Check if columns to normalize exist in the DataFrame
    missing_columns = [col for col in columns_to_normalize if col not in df.columns]
    if missing_columns:
        print(f"Warning: The following columns are missing and will not be normalized: {missing_columns}")
        columns_to_normalize = [col for col in columns_to_normalize if col in df.columns]
    if not columns_to_normalize:
        print("No columns to normalize. Exiting.")
        return
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    
    # Save the normalized dataset
    df.to_csv(output_file, index=False)
    
    print(f"Normalized dataset saved to {output_file}")
if __name__ == "__main__":
    input_file = output_path / 'water_stress_dataset.csv'  # Adjust this path as needed
    output_file = output_path / 'normalized_water_stress_dataset.csv'  # Adjust this path as needed
    
    normalize_dataset(input_file, output_file)
