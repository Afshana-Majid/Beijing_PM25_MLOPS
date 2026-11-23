import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data/processed/engineered_data.csv")

# Sort data chronologically
data["datetime"] = pd.to_datetime(data["datetime"])
data = data.sort_values("datetime")

# Separate features and target
X = data.drop(columns=["PM2.5", "datetime"])
y = data["PM2.5"]

# Train-test split
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Create Random Forest pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("Random Forest Model Results")
print("--------------------------------")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
import joblib
import os
# Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest_model.pkl")

print("\nModel saved successfully!")
