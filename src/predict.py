import pandas as pd
import joblib

# Load saved model
model = joblib.load("models/random_forest_model.pkl")

# Load dataset
data = pd.read_csv("data/processed/engineered_data.csv")

# Prepare test data
X = data.drop(columns=["PM2.5", "datetime"])
y = data["PM2.5"]

# Select sample records
X_sample = X.iloc[:5]
y_actual = y.iloc[:5]

# Make predictions
predictions = model.predict(X_sample)

# Display results
print("PM2.5 Prediction Results")
print("------------------------")

for actual, predicted in zip(y_actual, predictions):
    print("Actual:", actual, "| Predicted:", round(predicted, 2))

print("\nModel Loaded Successfully!")