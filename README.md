# Beijing Multi-Site Air Quality Prediction (MLOps)

## 1. Project Overview

This project focuses on predicting PM2.5 air pollution levels using historical air quality data collected from 12 monitoring stations in Beijing, China.

Machine Learning techniques are used to analyze environmental factors and predict PM2.5 concentration.

## 2. Project Objectives

- Analyze Beijing air quality data.
- Clean and preprocess the dataset.
- Perform Exploratory Data Analysis (EDA).
- Apply Feature Engineering.
- Train Machine Learning models.
- Evaluate model performance.
- Generate PM2.5 predictions.
- Perform model testing.

## 3. Dataset Description

The dataset contains air quality measurements from 12 monitoring stations in Beijing.

It includes the following features:

- PM2.5
- PM10
- SO2
- NO2
- CO
- O3
- Temperature (TEMP)
- Pressure (PRES)
- Dew Point (DEWP)
- Rain
- Wind Speed (WSPM)
- Wind Direction (wd)

**Dataset Period:** March 2013 to February 2017.

**Total Original Records:** 420,768

**Records After Cleaning:** 412,029

## 4. Technologies Used

- Python 3.14
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Git

## 5. Project Structure

```text
Beijing_PM25_MLOPS/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── cleaned_data.csv
│       └── engineered_data.csv
│
├── models/
│   └── random_forest_model.pkl
│
├── reports/
│   ├── pm25_distribution.png
│   └── pm25_by_station.png
│
├── src/
│   ├── check_environment.py
│   ├── load_data.py
│   ├── clean_data.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
│
├── tests/
│   └── test_model.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Project Workflow

1. Environment Setup
2. Data Loading
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Engineering
6. Model Training
7. Model Evaluation
8. Model Prediction
9. Model Testing

## 7. Machine Learning Models

Two regression models were trained:

1. Linear Regression
2. Random Forest Regressor

## 8. Model Performance

| Evaluation Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 16.75 | 12.69 |
| RMSE | 24.25 | 21.44 |
| R2 Score | 0.915 | 0.934 |

**Best Performing Model:** Random Forest Regressor

Random Forest performed better because it achieved lower MAE and RMSE and a higher R2 score.

## 9. How to Run the Project

### Step 1: Create Virtual Environment

```bash
python -m venv .venv
```

### Step 2: Activate Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Load Dataset

```bash
python src/load_data.py
```

### Step 5: Clean Dataset