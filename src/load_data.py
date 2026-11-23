import pandas as pd
import glob
import os

# Dataset folder path
data_path = "data/raw/PRSA_Data_20130301-20170228"

# Get all CSV files
csv_files = glob.glob(os.path.join(data_path, "*.csv"))

# Load and combine all CSV files
dataframes = []

for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

# Display dataset information
print("Dataset Loaded Successfully!")
print("Shape:", data.shape)
print("\nFirst 5 Rows:")
print(data.head())

print("\nColumns:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())
