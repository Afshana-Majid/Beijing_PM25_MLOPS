from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RAW = (
    ROOT
    / "data"
    / "raw"
    / "PRSA_Data_20130301-20170228"
    / "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
)

OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

if not RAW.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW}")

df = pd.read_csv(RAW)

print("DATASET LOADED SUCCESSFULLY")
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
# TIMESTAMP ASSESSMENT

df["timestamp"] = pd.to_datetime(
    df[["year", "month", "day", "hour"]],
    errors="coerce"
)

invalid_timestamps = df["timestamp"].isna().sum()
duplicate_timestamps = df["timestamp"].duplicated().sum()
is_sorted = df["timestamp"].is_monotonic_increasing

print("\nTIMESTAMP ASSESSMENT")
print("Invalid Timestamps:", invalid_timestamps)
print("Duplicate Timestamps:", duplicate_timestamps)
print("Chronologically Sorted:", is_sorted)
# MISSING VALUES ASSESSMENT

missing_counts = df.isnull().sum()
missing_percentages = (missing_counts / len(df)) * 100

print("\nMISSING VALUES PER COLUMN")
print(missing_counts)

print("\nMISSING VALUES PERCENTAGE")
print(missing_percentages.round(2))

# EXACT DUPLICATE ROWS

duplicate_rows = df.duplicated().sum()

print("\nEXACT DUPLICATE ROWS:", duplicate_rows)
# NEGATIVE POLLUTANT VALUES ASSESSMENT

pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

print("\nNEGATIVE POLLUTANT VALUES")

for column in pollutants:
    negative_count = (df[column] < 0).sum()
    print(f"{column}: {negative_count}")
    # SAFE DATA CLEANING

clean = df.copy()

# Remove exact duplicate rows
clean = clean.drop_duplicates()

# Remove invalid timestamps
clean = clean.dropna(subset=["timestamp"])

# Sort data chronologically
clean = clean.sort_values("timestamp").reset_index(drop=True)

# Convert negative pollutant values to NaN
for column in pollutants:
    clean.loc[clean[column] < 0, column] = pd.NA

print("\nSAFE CLEANING COMPLETED")
print("Original Rows:", len(df))
print("Cleaned Rows:", len(clean))
print("Missing PM2.5 After Cleaning:", clean["PM2.5"].isna().sum())
# MISSING HOURLY TIMESTAMPS

expected_hours = pd.date_range(
    start=clean["timestamp"].min(),
    end=clean["timestamp"].max(),
    freq="h"
)

actual_hours = pd.DatetimeIndex(clean["timestamp"])

missing_hours = expected_hours.difference(actual_hours)

print("\nMISSING HOURS ASSESSMENT")
print("Expected Hours:", len(expected_hours))
print("Actual Hours:", len(actual_hours))
print("Missing Hours:", len(missing_hours))
# SAVE CLEANED DATASET

cleaned_file = OUT / "aotizhongxin_quality_checked.csv"

clean.to_csv(cleaned_file, index=False)

print("\nCLEANED DATASET SAVED")
print("File:", cleaned_file)
print("Final Shape:", clean.shape)
# SAVE QUALITY REPORT

report_file = REPORTS / "practical03_quality_report.txt"

report = f"""
PRACTICAL 03 - DATA QUALITY ASSESSMENT REPORT

DATASET INFORMATION
Original Rows: {len(df)}
Original Columns: {len(df.columns) - 1}

TIMESTAMP ASSESSMENT
Invalid Timestamps: {invalid_timestamps}
Duplicate Timestamps: {duplicate_timestamps}
Chronologically Sorted: {is_sorted}

MISSING VALUES PER COLUMN
{missing_counts.to_string()}

MISSING VALUES PERCENTAGE
{missing_percentages.round(2).to_string()}

EXACT DUPLICATE ROWS
{duplicate_rows}

NEGATIVE POLLUTANT VALUES
{(df[pollutants] < 0).sum().to_string()}

SAFE CLEANING
Original Rows: {len(df)}
Cleaned Rows: {len(clean)}
Missing PM2.5 After Cleaning: {clean['PM2.5'].isna().sum()}

MISSING HOURS
Expected Hours: {len(expected_hours)}
Actual Hours: {len(actual_hours)}
Missing Hours: {len(missing_hours)}

FINAL CLEANED DATASET
Shape: {clean.shape}
"""

report_file.write_text(report, encoding="utf-8")

print("\nQUALITY REPORT SAVED:", report_file)