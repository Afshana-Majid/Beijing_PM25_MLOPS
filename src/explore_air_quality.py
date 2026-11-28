from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228" / (
    "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
)

if not CSV_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print("DATASET LOADED")
print("Rows and columns:", df.shape)
print("Column names:", df.columns.tolist())
print("First five rows:")
print(df.head(10))
print(df.tail())
print("\nDATA TYPES")
print(df.dtypes)
print("\nMISSING VALUES PER COLUMN")

missing_counts = df.isna().sum()

print(missing_counts)
print("\nMISSING PM2.5 VALUES")
print(df["PM2.5"].isna().sum())
print("\nTEMPERATURE SUMMARY")
print(df["TEMP"].describe())

print("\nTOTAL MISSING CELLS")
print(missing_counts.sum())
print("\nPM2.5 SUMMARY")
print(df["PM2.5"].describe())
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())
print("\nCREATING TIMESTAMP")

df["timestamp"] = pd.to_datetime(
    df[["year", "month", "day", "hour"]]
)

print("Earliest Timestamp:", df["timestamp"].min())
print("Latest Timestamp:", df["timestamp"].max())
print("Is Timestamp Sorted:", df["timestamp"].is_monotonic_increasing)
report_path = PROJECT_ROOT / "reports" / "inspection_report.txt"

with report_path.open("w", encoding="utf-8") as report:
    report.write("AIR QUALITY DATA INSPECTION REPORT\n")
    report.write("=" * 40 + "\n\n")

    report.write(f"Dataset Shape:\n{df.shape}\n\n")

    report.write("Column Names:\n")
    report.write(str(df.columns.tolist()) + "\n\n")

    report.write("Data Types:\n")
    report.write(str(df.dtypes) + "\n\n")

    report.write("Missing Values:\n")
    report.write(str(missing_counts) + "\n\n")

    report.write(f"Total Missing Cells: {missing_counts.sum()}\n\n")

    report.write("PM2.5 Summary:\n")
    report.write(str(df["PM2.5"].describe()) + "\n\n")

    report.write(f"Duplicate Rows: {df.duplicated().sum()}\n\n")

    report.write(f"Earliest Timestamp: {df['timestamp'].min()}\n")
    report.write(f"Latest Timestamp: {df['timestamp'].max()}\n")
    report.write(
        f"Timestamp Sorted: {df['timestamp'].is_monotonic_increasing}\n"
    )

print("\nInspection report saved successfully!")

