""" 
visulation.py 
Advanced visualization of water stress data and its relationship with environmental factors
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
from config import BBOX_FULL, COLOR_TO_SCORE, CLOUD_RGB

# Base paths
base_dir = script_dir.parent
output_path = base_dir / 'output'


def plot_water_stress(data: pd.DataFrame):
    """
    Plots the water stress data from the provided DataFrame.
    
    Parameters:
    - data: pd.DataFrame containing the water stress data.
    
    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['water_stress_mean'], label='Mean Water Stress', color='blue')
    plt.fill_between(data['date'], data['water_stress_min'], data['water_stress_max'], color='lightblue', alpha=0.5, label='Water Stress Range')
    
    plt.title('Water Stress Over Time')
    plt.xlabel('Date')
    plt.ylabel('Water Stress')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_temperature(data: pd.DataFrame):
    """
    Plots the temperature data from the provided DataFrame.
    Parameters:
    - data: pd.DataFrame containing the temperature data.
    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['temperature_2m_max'], label='Max Temperature', color='red')
    plt.plot(data['date'], data['temperature_2m_mean'], label='Mean Temperature', color='orange')
    plt.plot(data['date'], data['temperature_2m_min'], label='Min Temperature', color='yellow')
    
    plt.title('Temperature Over Time')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_precipitation(data: pd.DataFrame):
    """
    Plots the precipitation data from the provided DataFrame.
    Parameters:
    - data: pd.DataFrame containing the precipitation data.
    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.bar(data['date'], data['precipitation_sum'], label='Precipitation', color='blue', alpha=0.6)
    
    plt.title('Precipitation Over Time')
    plt.xlabel('Date')
    plt.ylabel('Precipitation (mm)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_wind_speed(data: pd.DataFrame):
    """
    Plots the wind speed data from the provided DataFrame.
    Parameters:
    - data: pd.DataFrame containing the wind speed data.
    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['wind_speed_10m_max'], label='Max Wind Speed', color='green')
    
    plt.title('Wind Speed Over Time')
    plt.xlabel('Date')
    plt.ylabel('Wind Speed (m/s)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_relative_humidity(data: pd.DataFrame):
    """
    Plots the relative humidity data from the provided DataFrame.
    Parameters:
    - data: pd.DataFrame containing the relative humidity data.
    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['relative_humidity_2m_max'], label='Max Relative Humidity', color='purple')
    plt.plot(data['date'], data['relative_humidity_2m_min'], label='Min Relative Humidity', color='pink')
    plt.plot(data['date'], data['relative_humidity_2m_mean'], label='Mean Relative Humidity', color='violet')
    
    plt.title('Relative Humidity Over Time')
    plt.xlabel('Date')
    plt.ylabel('Relative Humidity (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_soil_moisture(data: pd.DataFrame):
    """
    Plots the soil moisture data from the provided DataFrame.
    Parameters:
    - data: pd.DataFrame containing the soil moisture data.
    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['soil_moisture_0_to_10cm_mean'], label='Mean Soil Moisture (0-10 cm)', color='brown')
    
    plt.title('Soil Moisture Over Time')
    plt.xlabel('Date')
    plt.ylabel('Soil Moisture (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_sunshine_duration(data: pd.DataFrame):
    """
    Plots the sunshine duration data from the provided DataFrame.
    Parameters:
    - data: pd.DataFrame containing the sunshine duration data.
    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['sunshine_duration'], label='Sunshine Duration', color='gold')
    
    plt.title('Sunshine Duration Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sunshine Duration (hours)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_et0_fao_evapotranspiration(data: pd.DataFrame):
    """
    Plots the ET0 FAO evapotranspiration data from the provided DataFrame.
    Parameters:
    - data: pd.DataFrame containing the ET0 FAO evapotranspiration data.
    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['et0_fao_evapotranspiration_sum'], label='ET0 FAO Evapotranspiration', color='cyan')
    
    plt.title('ET0 FAO Evapotranspiration Over Time')
    plt.xlabel('Date')
    plt.ylabel('ET0 FAO Evapotranspiration (mm)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_all_time_series_grid(data: pd.DataFrame):
    """
    Plots all main variables as time series in a grid for easy comparison.
    """
    variables = [
        ('water_stress_mean', 'Water Stress Mean'),
        ('temperature_2m_max', 'Temp Max (°C)'),
        ('temperature_2m_mean', 'Temp Mean (°C)'),
        ('temperature_2m_min', 'Temp Min (°C)'),
        ('precipitation_sum', 'Precipitation (mm)'),
        ('wind_speed_10m_max', 'Wind Speed Max (m/s)'),
        ('relative_humidity_2m_mean', 'Rel. Humidity Mean (%)'),
        ('soil_moisture_0_to_10cm_mean', 'Soil Moisture (%)'),
        ('sunshine_duration', 'Sunshine Duration (h)'),
        ('et0_fao_evapotranspiration_sum', 'Evapotranspiration (mm)'),
    ]
    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3*n), sharex=True)
    for i, (col, label) in enumerate(variables):
        axes[i].plot(data['date'], data[col], label=label)
        axes[i].set_ylabel(label)
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel('Date')
    plt.suptitle('All Variables Over Time')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
def plot_all_time_series_single(data: pd.DataFrame):
    """
    Plots water stress as a line, all other variables as dots on a single plot for easy comparison.
    """
    variables = [
        ('water_stress_mean', 'Water Stress Mean', 'blue'),
       # ('temperature_2m_max', 'Temp Max (°C)', 'red'),
       # ('temperature_2m_mean', 'Temp Mean (°C)', 'orange'),
       # ('temperature_2m_min', 'Temp Min (°C)', 'yellow'),
       # ('precipitation_sum', 'Precipitation (mm)', 'purple'),
       # ('wind_speed_10m_max', 'Wind Speed Max (m/s)', 'green'),
       # ('relative_humidity_2m_mean', 'Rel. Humidity Mean (%)', 'violet'),
       ('soil_moisture_0_to_10cm_mean', 'Soil Moisture (%)', 'brown'),
       # ('sunshine_duration', 'Sunshine Duration (h)', 'gold'),
       # ('et0_fao_evapotranspiration_sum', 'Evapotranspiration (mm)', 'cyan'),
    ]
    plt.figure(figsize=(16, 8))
    # Water stress as line
    plt.plot(data['date'], data['water_stress_mean'], label='Water Stress Mean', color='blue', linewidth=2)
    # Others as dots
    for col, label, color in variables:
        if col == 'water_stress_mean':
            continue
        plt.scatter(data['date'], data[col], label=label, color=color, s=40, alpha=0.7)
    plt.title('Water Stress (Line) and Other Variables (Dots) Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    """
    Plots all main variables normalized (0-1) on a single plot for easy comparison.
    """
    variables = [
        ('water_stress_mean', 'Water Stress Mean', 'blue'),
        ('temperature_2m_max', 'Temp Max (°C)', 'red'),
        ('temperature_2m_mean', 'Temp Mean (°C)', 'orange'),
        ('temperature_2m_min', 'Temp Min (°C)', 'yellow'),
        ('precipitation_sum', 'Precipitation (mm)', 'purple'),
        ('wind_speed_10m_max', 'Wind Speed Max (m/s)', 'green'),
        ('relative_humidity_2m_mean', 'Rel. Humidity Mean (%)', 'violet'),
        ('soil_moisture_0_to_10cm_mean', 'Soil Moisture (%)', 'brown'),
        ('sunshine_duration', 'Sunshine Duration (h)', 'gold'),
        ('et0_fao_evapotranspiration_sum', 'Evapotranspiration (mm)', 'cyan'),
    ]
    plt.figure(figsize=(16, 8))
    for col, label, color in variables:
        vals = data[col].values.astype(float)
        if np.nanmax(vals) != np.nanmin(vals):
            norm_vals = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
            plt.plot(data['date'], norm_vals, label=label, color=color)
    plt.title('All Variables Over Time (Normalized 0-1)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_correlation_heatmap(data: pd.DataFrame):
    """
    Plots a correlation heatmap for all numeric variables.
    """
    plt.figure(figsize=(12, 8))
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
def plot_water_stress_vs_factors(data: pd.DataFrame):
    """
    Plots scatter plots of water stress mean vs. each main factor.
    """
    factors = [
        'temperature_2m_max', 'temperature_2m_mean', 'temperature_2m_min',
        'precipitation_sum', 'wind_speed_10m_max',
        'relative_humidity_2m_mean', 'soil_moisture_0_to_10cm_mean',
        'sunshine_duration', 'et0_fao_evapotranspiration_sum'
    ]
    n = len(factors)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    for i, factor in enumerate(factors):
        ax = axes[i//ncols, i%ncols]
        sns.scatterplot(x=data[factor], y=data['water_stress_mean'], ax=ax, alpha=0.5)
        ax.set_xlabel(factor)
        ax.set_ylabel('Water Stress Mean')
        ax.set_title(f'Water Stress vs. {factor}')
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes[j//ncols, j%ncols])
    plt.tight_layout()
    plt.show()
def plot_paired_time_series(data: pd.DataFrame, y1: str, y2: str, y1_label: str, y2_label: str, color1: str = 'blue', color2: str = 'green'):
    """
    Plots two time series as points (no lines), each value as a dot.
    y1: main variable (solid dots), y2: secondary variable (dotted dots)
    Each call opens a new window.
    """
    plt.figure(figsize=(14, 6))
    plt.scatter(data['date'], data[y1], label=y1_label, color=color1, marker='o', s=40, alpha=0.8)
    plt.scatter(data['date'], data[y2], label=y2_label, color=color2, marker='o', s=40, alpha=0.8)
    plt.title(f'{y1_label} and {y2_label} Over Time (Dots Only)')
    plt.xlabel('Date')
    plt.ylabel(f'{y1_label} / {y2_label}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_all(data: pd.DataFrame):
    """
    Plots all relevant data from the provided DataFrame.
    Parameters:
    - data: pd.DataFrame containing the relevant data.
    Returns:
    - None
    """
    plot_all_time_series_grid(data)
    plot_correlation_heatmap(data)
    plot_water_stress_vs_factors(data)
    # Paired plots for direct comparison (each opens in a new window)
    plot_paired_time_series(data, 'water_stress_mean', 'wind_speed_10m_max', 'Water Stress Mean', 'Wind Speed Max (m/s)', 'blue', 'green')
    plot_paired_time_series(data, 'water_stress_mean', 'temperature_2m_mean', 'Water Stress Mean', 'Temperature Mean (°C)', 'blue', 'orange')
    plot_paired_time_series(data, 'water_stress_mean', 'precipitation_sum', 'Water Stress Mean', 'Precipitation (mm)', 'blue', 'purple')
    # ...add more pairs as needed
    # Optionally, keep the old plots:
    # plot_water_stress(data)
    # plot_temperature(data)
    # plot_precipitation(data)
    # plot_wind_speed(data)
    # plot_relative_humidity(data)
    # plot_soil_moisture(data)
    # plot_sunshine_duration(data)
    # plot_et0_fao_evapotranspiration(data)
def plot_cell_time_series(data: pd.DataFrame, lat: float, lon: float):
    """
    Plots all main variables over time for a specific grid cell (lat, lon).
    """
    cell_data = data[(data['lat'] == lat) & (data['lon'] == lon)].sort_values('date')
    if cell_data.empty:
        print(f"No data found for lat={lat}, lon={lon}")
        return
    plot_all_time_series_single(cell_data)
    """  
    plot_correlation_heatmap(cell_data)
    plot_water_stress_vs_factors(cell_data)
    plot_paired_time_series(cell_data, 'water_stress_mean', 'wind_speed_10m_max', 'Water Stress Mean', 'Wind Speed Max (m/s)', 'blue', 'green')
    plot_paired_time_series(cell_data, 'water_stress_mean', 'temperature_2m_mean', 'Water Stress Mean', 'Temperature Mean (°C)', 'blue', 'orange')
    plot_paired_time_series(cell_data, 'water_stress_mean', 'precipitation_sum', 'Water Stress Mean', 'Precipitation (mm)', 'blue', 'purple')
    """
if __name__ == "__main__":
    # Example usage
    # Load your data into a DataFrame
    filePath = output_path / 'normalized_water_stress_dataset.csv'
    data = pd.read_csv(filePath, parse_dates=['date'])
    
    # Call the plot_all function to visualize all data
    #plot_all(data)
    
    # Example: visualize for a specific grid cell (change lat/lon as needed)
    # Pick the first available (lat, lon) pair from the data
    unique_cells = data[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
    if not unique_cells.empty:
        example_lat = unique_cells.loc[0, 'lat']
        example_lon = unique_cells.loc[0, 'lon']
        print(f"Plotting for lat={example_lat}, lon={example_lon}")
        plot_cell_time_series(data, example_lat, example_lon)