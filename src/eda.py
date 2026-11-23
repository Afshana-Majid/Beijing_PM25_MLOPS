
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create reports folder if not available
os.makedirs("reports", exist_ok=True)

# Load cleaned dataset
data = pd.read_csv("data/processed/cleaned_data.csv")

# Dataset information
print("Dataset Shape:", data.shape)

print("\nStatistical Summary:")
print(data.describe())

# PM2.5 Distribution
plt.figure(figsize=(8, 5))
plt.hist(data["PM2.5"], bins=50)
plt.title("PM2.5 Distribution")
plt.xlabel("PM2.5 Concentration")
plt.ylabel("Frequency")
plt.savefig("reports/pm25_distribution.png")
plt.close()

# Average PM2.5 by Station
station_pm25 = data.groupby("station")["PM2.5"].mean()

plt.figure(figsize=(10, 5))
station_pm25.plot(kind="bar")
plt.title("Average PM2.5 by Station")
plt.xlabel("Station")
plt.ylabel("Average PM2.5")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/pm25_by_station.png")
plt.close()

print("\nEDA Completed!")
print("Both graphs saved successfully!")
