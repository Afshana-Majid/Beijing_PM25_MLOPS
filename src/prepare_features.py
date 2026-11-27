from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error


# ==========================================
# STAGE A: IMPORTS, PATHS AND LOADING
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "processed" / "aotizhongxin_quality_checked.csv"
OUTPUT = ROOT / "data" / "processed"

OUTPUT.mkdir(parents=True, exist_ok=True)

if not INPUT.is_file():
    raise FileNotFoundError(f"Cleaned input not found: {INPUT}")

df = pd.read_csv(INPUT)

print("Loaded rows:", len(df))
print("Columns:", df.columns.tolist())


# ==========================================
# STAGE B: CONSTRUCT AND VALIDATE TIMESTAMPS
# ==========================================

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )
else:
    required = ["year", "month", "day", "hour"]

    if not all(col in df.columns for col in required):
        raise ValueError(
            "Need timestamp or year/month/day/hour columns"
        )

    df["timestamp"] = pd.to_datetime(
        df[required], errors="coerce"
    )

if df["timestamp"].isna().any():
    raise ValueError("Some timestamps cannot be interpreted")

df = df.sort_values("timestamp").reset_index(drop=True)

if df["timestamp"].duplicated().any():
    raise ValueError(
        "Duplicate timestamps: review Practical 03 cleaning"
    )

intervals = df["timestamp"].diff().dropna()

if not intervals.eq(pd.Timedelta(hours=1)).all():
    raise ValueError(
        "Hourly gaps detected; repair or document before shifting"
    )

print("Hourly timestamp sequence verified")


# ==========================================
# STAGE C: DEFINE PREDICTION TARGET AND FEATURES
# ==========================================

df["PM2.5"] = pd.to_numeric(
    df["PM2.5"], errors="coerce"
)

# Create lag features
df["pm25_lag1"] = df["PM2.5"].shift(1)
df["pm25_lag2"] = df["PM2.5"].shift(2)

# Extract hour of day
df["hour_of_day"] = df["timestamp"].dt.hour

# Create one-hour-ahead target
df["target_pm25_next_hour"] = df["PM2.5"].shift(-1)

# Define features and target
feature_cols = [
    "PM2.5",
    "pm25_lag1",
    "pm25_lag2",
    "hour_of_day"
]

target_col = "target_pm25_next_hour"

# Create model-ready dataset
model_df = df[
    ["timestamp"] + feature_cols + [target_col]
].dropna().copy()

print("Usable supervised rows:", len(model_df))
print(model_df.head(3).to_string(index=False))


# ==========================================
# STAGE D: CHRONOLOGICAL DATA SPLITTING
# ==========================================

n = len(model_df)

train_end = int(n * 0.70)
valid_end = int(n * 0.85)

train = model_df.iloc[:train_end].copy()
valid = model_df.iloc[train_end:valid_end].copy()
test = model_df.iloc[valid_end:].copy()

# Verify splits
assert len(train) > 0
assert len(valid) > 0
assert len(test) > 0

assert train["timestamp"].max() < valid["timestamp"].min()
assert valid["timestamp"].max() < test["timestamp"].min()

print("Training rows:", len(train))
print("Validation rows:", len(valid))
print("Testing rows:", len(test))


# ==========================================
# STAGE E: PREVENT TARGET CROSSING
# ==========================================

train = train.iloc[:-1].copy()
valid = valid.iloc[:-1].copy()

for name, part in [
    ("train", train),
    ("validation", valid),
    ("test", test)
]:
    part.to_csv(
        OUTPUT / f"{name}_features.csv",
        index=False
    )

    print(name, "rows:", len(part))


# ==========================================
# STAGE F: PERSISTENCE BASELINE EVALUATION
# ==========================================

baseline_predictions = test["PM2.5"]
actual_values = test[target_col]

mae = mean_absolute_error(
    actual_values,
    baseline_predictions
)

print(f"Test persistence MAE: {mae:.2f}")


# ==========================================
# SAVE SPLIT SUMMARY
# ==========================================

with open(
    OUTPUT / "split_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(f"Training rows: {len(train)}\n")
    f.write(f"Validation rows: {len(valid)}\n")
    f.write(f"Test rows: {len(test)}\n")
    f.write(f"Persistence test MAE: {mae:.2f}\n")

print("Split summary saved successfully!")