
import pandas as pd
import glob
import os

# Load all CSV files
data_path = "data/raw/PRSA_Data_20130301-20170228"

csv_files = glob.glob(os.path.join(data_path, "*.csv"))

dataframes = []

for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

# Create datetime column
data["datetime"] = pd.to_datetime(
    data[["year", "month", "day", "hour"]]
)

# Sort data chronologically within each station
data = data.sort_values(["station", "datetime"])

# Remove unnecessary column
data = data.drop(columns=["No"])

# Remove rows where target PM2.5 is missing
data = data.dropna(subset=["PM2.5"])

# Handle missing categorical values
data["wd"] = data["wd"].fillna("Unknown")

# Keep missing numerical features for later imputation
# Imputation will be learned from training data only

# Save cleaned dataset
os.makedirs("data/processed", exist_ok=True)

data.to_csv(
    "data/processed/cleaned_data.csv",
    index=False
)

print("Data Cleaning Completed!")
print("Final Shape:", data.shape)

print("\nMissing Values:")
print(data.isnull().sum())
