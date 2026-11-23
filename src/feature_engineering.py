import pandas as pd
import os

# Load cleaned dataset
data = pd.read_csv("data/processed/cleaned_data.csv")

# Convert datetime
data["datetime"] = pd.to_datetime(data["datetime"])

# Extract date features
data["year"] = data["datetime"].dt.year
data["month"] = data["datetime"].dt.month
data["day"] = data["datetime"].dt.day
data["hour"] = data["datetime"].dt.hour
data["day_of_week"] = data["datetime"].dt.day_of_week
# Create season feature
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

data["season"] = data["month"].apply(get_season)

# Encode categorical columns
data = pd.get_dummies(
    data,
    columns=["station", "wd", "season"],
    dtype=int
)

# Save engineered dataset
os.makedirs("data/processed", exist_ok=True)

data.to_csv(
    "data/processed/engineered_data.csv",
    index=False
)

print("Feature Engineering Completed!")
print("Dataset Shape:", data.shape)
print("Engineered dataset saved successfully!")