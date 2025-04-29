# Script to parse over csv file to parse and format output data(labels to be predicted)

# Import
import pandas as pd
import numpy as np



def extract_weather_data(csv_path, output_path='weather_data.npy'):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Select relevant columns (handle missing values if necessary)
    selected = df[['DATE', 'TAVG', 'PRCP', 'SNOW']].dropna()

    # Optionally sort by date
    selected = selected.sort_values('DATE')

    # Extract features (TAVG, PRCP, SNOW) as float32 NumPy array
    features = selected[['TAVG', 'PRCP', 'SNOW']].to_numpy(dtype=np.float32)

    # Save as .npy file
    np.save(output_path, features)

    print(f"Saved weather data to {output_path} with shape {features.shape}")
    return features


# CALL NEW FUNCTION
extract_weather_data('3997445.csv')

