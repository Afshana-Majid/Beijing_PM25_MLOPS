# Beijing PM2.5 One-Hour-Ahead Forecasting

MSc AI/ML Semester III MLOps Laboratory Project.

## 1. Project Overview

This project focuses on predicting PM2.5 concentration one hour ahead using historical air-quality observations from Beijing, China.

Machine Learning techniques are used to analyze air-quality data and forecast future PM2.5 levels.

## 2. Project Objectives

- Inspect and understand the dataset.
- Clean and preprocess air-quality data.
- Perform Exploratory Data Analysis (EDA).
- Apply feature engineering.
- Prepare time-based training, validation, and test datasets.
- Train and evaluate Machine Learning models.
- Apply Git version control.

## 3. Dataset Description

The project uses the Beijing Multi-Site Air Quality dataset.

The forecasting work focuses on the Aotizhongxin monitoring station.

Important features include:

- PM2.5
- PM10
- SO2
- NO2
- CO
- O3
- TEMP
- PRES
- DEWP
- RAIN
- Wind Direction (wd)
- Wind Speed (WSPM)

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Git

## 5. Project Structure

```text
Beijing_PM25_MLOPS/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Project Workflow

1. Environment Setup
2. Data Inspection
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Engineering
6. Time-Based Data Splitting
7. Model Training and Evaluation
8. Git Version Control

## 7. Model Performance

The Ridge Regression model was selected during Practical 05.

Test Results:

- Test MAE: 10.69
- Test RMSE: 19.75

The trained model pipeline was saved as:

`models/ridge_pm25_pipeline.joblib`

The evaluation report was saved as:

`reports/practical05_metrics.json`

## 8. How to Run the Project

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 9. Practical Progress

- Practical 01: Project Setup
- Practical 02: Data Inspection and EDA
- Practical 03: Data Cleaning
- Practical 04: Feature Engineering and Data Splitting
- Practical 05: Model Training and Evaluation
- Practical 06: Git Version Control

## 10. Version Control

Git is used to track project changes and maintain version history.

Large datasets, trained models, virtual environments, and local configuration files are excluded through `.gitignore`.