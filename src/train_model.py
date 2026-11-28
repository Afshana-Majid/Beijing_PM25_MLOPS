from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


# STAGE A: PROJECT PATHS

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# STAGE B: LOAD DATA

def load_split(filename):
    path = DATA / filename

    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


train = load_split("train_features.csv")
valid = load_split("validation_features.csv")
test = load_split("test_features.csv")

print("Rows:", len(train), len(valid), len(test))


# STAGE C: FEATURES AND TARGET

FEATURES = [
    "PM2.5",
    "pm25_lag1",
    "pm25_lag2",
    "hour_of_day"
]

TARGET = "target_pm25_next_hour"

required = FEATURES + [TARGET]

for name, frame in [
    ("train", train),
    ("validation", valid),
    ("test", test)
]:

    missing = [c for c in required if c not in frame.columns]

    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    if frame[TARGET].isna().any():
        raise ValueError(f"{name} contains missing target values")


X_train, y_train = train[FEATURES], train[TARGET]
X_valid, y_valid = valid[FEATURES], valid[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]


# STAGE D: RIDGE REGRESSION PIPELINE

pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)

print("Model fitted on training rows only")


# STAGE E: VALIDATION

model_valid_pred = pipeline.predict(X_valid)

baseline_valid_pred = valid["PM2.5"].to_numpy()

model_mae = mean_absolute_error(y_valid, model_valid_pred)

baseline_mae = mean_absolute_error(
    y_valid, baseline_valid_pred
)

model_rmse = mean_squared_error(
    y_valid, model_valid_pred
) ** 0.5

baseline_rmse = mean_squared_error(
    y_valid, baseline_valid_pred
) ** 0.5


print("\nVALIDATION RESULTS")
print("---------------------------")

print(f"Ridge MAE: {model_mae:.2f}")
print(f"Persistence MAE: {baseline_mae:.2f}")

print(f"Ridge RMSE: {model_rmse:.2f}")
print(f"Persistence RMSE: {baseline_rmse:.2f}")


# STAGE F: MODEL SELECTION AND TEST

if model_mae < baseline_mae:

    selected = "ridge"
    test_pred = pipeline.predict(X_test)

else:

    selected = "persistence"
    test_pred = test["PM2.5"].to_numpy()


test_mae = mean_absolute_error(y_test, test_pred)

test_rmse = mean_squared_error(
    y_test, test_pred
) ** 0.5


print("\nTEST RESULTS")
print("---------------------------")

print("Selected model:", selected)
print(f"Test MAE: {test_mae:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")


# STAGE G: SAVE MODEL AND REPORT

model_path = MODEL_DIR / "ridge_pm25_pipeline.joblib"

joblib.dump(pipeline, model_path)


results = {
    "features": FEATURES,
    "target": TARGET,
    "selected_on_validation": selected,

    "validation_ridge_mae": float(model_mae),
    "validation_persistence_mae": float(baseline_mae),

    "validation_ridge_rmse": float(model_rmse),
    "validation_persistence_rmse": float(baseline_rmse),

    "held_out_test_mae": float(test_mae),
    "held_out_test_rmse": float(test_rmse),

    "train_rows": len(train),
    "validation_rows": len(valid),
    "test_rows": len(test)
}


report_path = REPORT_DIR / "practical05_metrics.json"

report_path.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)

print("\nSAVED FILES")
print("Model:", model_path)
print("Report:", report_path)


# STAGE H: RELOAD VERIFICATION

reloaded = joblib.load(model_path)

original = pipeline.predict(X_valid.iloc[:3])
restored = reloaded.predict(X_valid.iloc[:3])

assert all(
    abs(a - b) < 1e-9
    for a, b in zip(original, restored)
)

print("\nReload verification passed")