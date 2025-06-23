import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
from pathlib import Path

# Add the project directory to path so we can import the config
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
from config import BBOX_FULL, COLOR_TO_SCORE, CLOUD_RGB

# Base paths
base_dir = script_dir.parent
satellite_path = base_dir / 'data' / 'satellite'
weather_path = base_dir / 'data' / 'weather'
output_path = base_dir / 'output'

# Print debug info
print(f"Base directory: {base_dir}")
print(f"Satellite path: {satellite_path}")
print(f"Weather path: {weather_path}")
print(f"Output path: {output_path}")

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

def load_weather_data(weather_file='data.json'):
    """Load weather data from JSON file."""
    with open(weather_path / weather_file, 'r') as f:
        weather_data = json.load(f)
    return weather_data

def load_satellite_dates():
    """Get list of dates with available satellite images."""
    print(f"Looking for satellite images in: {satellite_path}")
    
    try:
        # Check if directory exists
        if not os.path.exists(satellite_path):
            print(f"ERROR: Satellite directory not found: {satellite_path}")
            # Try to create sample images for testing
            create_sample_images()
            if not os.path.exists(satellite_path):
                return {}
        
        # List files in directory
        satellite_files = os.listdir(satellite_path)
        print(f"Found {len(satellite_files)} files in satellite directory")
        
        # Check for any PNG files
        png_files = [f for f in satellite_files if f.lower().endswith('.png')]
        print(f"Found {len(png_files)} PNG files")
        
        # Generate dictionary of dates -> filenames
        satellite_dates = {}
        for file in satellite_files:
            if file.endswith('_WaterStress.png'):
                try:
                    date_str = file.split('_')[0]  # Extract date part from filename
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    satellite_dates[date_obj.strftime('%Y-%m-%d')] = file
                except Exception as e:
                    print(f"Error processing filename {file}: {e}")
        
        print(f"Found {len(satellite_dates)} valid satellite images")
        return satellite_dates
    except Exception as e:
        print(f"Error loading satellite dates: {e}")
        return {}

def create_sample_images():
    """Create sample test images if satellite directory doesn't exist."""
    try:
        # Create directories if they don't exist
        os.makedirs(satellite_path, exist_ok=True)
        
        # Create a simple test image
        # Create a simple test image
        from PIL import Image
        
        print("Creating sample test images...")
        # Create images with different water stress levels
        # Use the exact keys and values from COLOR_TO_SCORE for sample images
        from config import COLOR_TO_SCORE
        colors = list(COLOR_TO_SCORE.keys())

        # Create sample images for today and previous days
        for i, color in enumerate(colors):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            img_path = os.path.join(satellite_path, f"{date}_WaterStress.png")
            img = Image.new('RGB', (100, 100), color)
            img.save(img_path)
            print(f"Created sample image: {img_path}")
        print("Sample images created successfully")
    except Exception as e:
        print(f"Error creating sample images: {e}")

def closest_color(pixel_color):
    """Find the closest color in COLOR_TO_SCORE dictionary."""
    try:
        # Make sure pixel_color has enough elements to access
        if len(pixel_color) < 3:
            return None
            
        # Check for transparency if alpha channel exists
        if len(pixel_color) > 3 and pixel_color[3] == 0:
            return None
            
        # Check if it's a cloud pixel
        if tuple(pixel_color[:3]) == CLOUD_RGB:
            return None
        
        min_distance = float('inf')
        closest = None
        
        for color in COLOR_TO_SCORE.keys():
            # Ensure we only compare the first 3 elements (RGB)
            pixel_rgb = pixel_color[:3]
            distance = np.sqrt(sum((np.array(color) - np.array(pixel_rgb))**2))
            if distance < min_distance:
                min_distance = distance
                closest = color
        
        # Only return a match if it's reasonably close (threshold of 30)
        return closest if min_distance < 30 else None
    except Exception as e:
        print(f"Error in closest_color: {e}, pixel_color={pixel_color}")
        return None

def process_satellite_image(image_file, grid_size=100):
    """
    Process a satellite image and extract water stress scores for each grid cell.
    
    Args:
        image_file: Path to the satellite image file
        grid_size: Size of each grid cell in pixels
    
    Returns:
        Dictionary with grid cell coordinates as keys and water stress scores as values
    """
    # Always use PIL to read the file to handle special characters in paths
    try:
        from PIL import Image
        pil_img = Image.open(image_file).convert("RGBA")
        img = np.array(pil_img)

        # If image has 3 channels, convert to 4 channels (add alpha)
        if len(img.shape) == 3 and img.shape[2] == 3:
            alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
            img = np.concatenate((img, alpha), axis=2)
    except Exception as e:
        print(f"Error: Unable to read image file {image_file}: {e}")
        return {}
    
    # Calculate grid dimensions
    num_rows = int(np.ceil(img.shape[0] / grid_size))
    num_cols = int(np.ceil(img.shape[1] / grid_size))
    
    # Calculate geo coordinates step sizes
    lon_min, lat_min, lon_max, lat_max = BBOX_FULL
    lon_step = (lon_max - lon_min) / num_cols
    lat_step = (lat_max - lat_min) / num_rows
    
    grid_scores = {}
    
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate grid cell boundaries
            y_start = row * grid_size
            y_end = min((row + 1) * grid_size, img.shape[0])
            x_start = col * grid_size
            x_end = min((col + 1) * grid_size, img.shape[1])
            
            # Calculate geo coordinates for this grid cell
            cell_lon_min = lon_min + col * lon_step
            cell_lon_max = lon_min + (col + 1) * lon_step
            cell_lat_max = lat_max - row * lat_step  # Inverted because rows go from top to bottom
            cell_lat_min = lat_max - (row + 1) * lat_step
            
            # Cell centroid coordinates
            cell_lon = (cell_lon_min + cell_lon_max) / 2
            cell_lat = (cell_lat_min + cell_lat_max) / 2
            
            # Extract grid cell
            cell = img[y_start:y_end, x_start:x_end]
            
            # Process colors in the grid
            valid_scores = []
            total_pixels = (y_end - y_start) * (x_end - x_start)
            non_transparent_pixels = 0
            for y in range(y_end - y_start):
                for x in range(x_end - x_start):
                    try:
                        # Safely get pixel color as tuple
                        pixel = cell[y, x]

                        # Ensure pixel_color is always a tuple of ints
                        if isinstance(pixel, np.ndarray):
                            pixel_color = tuple(int(v) for v in pixel.tolist())
                        elif isinstance(pixel, (list, tuple)):
                            pixel_color = tuple(int(v) for v in pixel)
                        else:
                            # Single channel image (grayscale), skip
                            continue

                        # Skip transparent pixels if we have an alpha channel
                        if len(pixel_color) > 3 and pixel_color[3] == 0:
                            continue

                        non_transparent_pixels += 1
                        closest = closest_color(pixel_color)

                        if closest and COLOR_TO_SCORE[closest] is not None:
                            valid_scores.append(COLOR_TO_SCORE[closest])
                    except Exception as e:
                        # Skip problematic pixels
                        print(f"Error processing pixel at ({x},{y}): {e}")
                        continue
            
            # Calculate coverage percentage (how much of the cell is not transparent)
            coverage = non_transparent_pixels / total_pixels if total_pixels > 0 else 0
            
            # Only consider cells with significant coverage and valid scores
            if coverage > 0.25 and valid_scores:
                grid_scores[(cell_lat, cell_lon)] = {
                    'water_stress_mean': round(np.mean(valid_scores), 3),
                    'water_stress_min': round(min(valid_scores), 3),
                    'water_stress_max': round(max(valid_scores), 3),
                    'coverage': round(coverage, 3),
                    'lat': round(cell_lat, 6),
                    'lon': round(cell_lon, 6)
                }
    
    return grid_scores

def build_dataset():
    """Build the complete dataset combining satellite and weather data."""
    # Load weather data
    print("Loading weather data...")
    try:
        weather_data = load_weather_data()
        weather_daily = weather_data['daily']
        weather_times = weather_daily['time']
    except Exception as e:
        print(f"Error loading weather data: {e}")
        return pd.DataFrame()
    
    # Get dates with satellite images
    print("Finding available satellite images...")
    try:
        satellite_dates = load_satellite_dates()
        if not satellite_dates:
            print("No satellite images found. Check satellite path.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error finding satellite images: {e}")
        return pd.DataFrame()
    
    # Prepare dataset
    dataset_rows = []
    
    # Track count of processed images for progress reporting
    total_images = len(satellite_dates)
    processed_images = 0
    
    print(f"Processing {total_images} satellite images...")
    
    # Process each date with a satellite image
    for date_str, image_file in tqdm(satellite_dates.items()):
        try:
            processed_images += 1
            print(f"Processing image {processed_images}/{total_images}: {image_file}")
            
            # Find corresponding weather data index
            if date_str not in weather_times:
                print(f"Warning: No weather data available for {date_str}")
                continue
            
            weather_idx = weather_times.index(date_str)
            
            # Process satellite image to get water stress scores by grid cell
            image_path = satellite_path / image_file
            
            # Debug information
            print(f"Reading image file: {image_path}")
            print(f"File exists: {os.path.exists(image_path)}")
            
            grid_scores = process_satellite_image(image_path)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue
        
        if not grid_scores:
            print(f"Warning: No valid data extracted from {image_file}")
            continue
        
        # Get date components for features
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        day_of_year = date_obj.timetuple().tm_yday
        day_of_week = date_obj.weekday()
        month = date_obj.month
        quarter = (month - 1) // 3 + 1
        
        # Add each grid cell as a separate row
        for _, score_data in grid_scores.items():
            row_data = {
                'date': date_str,
                'day_of_year': day_of_year,
                'day_of_week': day_of_week,
                'month': month,
                'quarter': quarter,
            }
            
            # Add water stress data
            row_data.update(score_data)
            
            # Add weather data for this date
            for weather_param in weather_daily.keys():
                if weather_param != 'time' and weather_idx < len(weather_daily[weather_param]):
                    param_value = weather_daily[weather_param][weather_idx]
                    row_data[weather_param] = param_value
            
            dataset_rows.append(row_data)
    df = pd.DataFrame(dataset_rows)
    
    # Save to CSV
    output_file = output_path / 'water_stress_dataset.csv'
    print(f"Saving dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"Dataset creation complete. Total rows: {len(df)}")
    print(f"Dataset saved to: {output_file}")
    
    # Display some statistics
    if len(df) > 0:
        print("\nDataset Statistics:")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Number of unique grid cells: {df[['lat', 'lon']].drop_duplicates().shape[0]}")
        print(f"Average water stress: {df['water_stress_mean'].mean():.3f}")
        print("\nColumn names in the dataset:")
        for col in df.columns:
            print(f"- {col}")
    
    return df

if __name__ == "__main__":
    build_dataset()
