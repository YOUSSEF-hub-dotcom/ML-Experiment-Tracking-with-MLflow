# ML Experiment Tracking with MLflow

## Description

This project demonstrates ML experiment tracking with **MLflow** on the
**California Housing** dataset. A single regression model
(`GradientBoostingRegressor`) is trained three times with different
hyperparameters, and MLflow records each run's parameters, validation metrics
(RMSE, MAE, R²) and the trained model artifact. The best model is selected by
the lowest validation **RMSE**.

Pipeline: **Train → Track → Compare → Select**.

## Project Structure

```
mlflow-task/
├── train.py          # loads data, trains 3 runs, logs to MLflow
├── requirements.txt  # pinned dependencies
└── README.md
```

## How to Run

From the project root (the folder containing `housing.csv`):

```powershell
python -m venv mlflow_env
.\mlflow_env\Scripts\Activate.ps1
pip install -r mlflow-task\requirements.txt
python mlflow-task\train.py
```

The script prints a comparison table and the best run, and writes all runs to
the local `mlruns/` tracking store.

To open the MLflow UI (MLflow 3.x requires opting in to the file store):

```powershell
$env:MLFLOW_ALLOW_FILE_STORE="true"
mlflow ui --backend-store-uri .\mlruns
```

Then open <http://127.0.0.1:5000> and select the `california-housing`
experiment.

## Experiment Results

![MLflow UI screenshot](screenshot.png)

> Add your MLflow UI screenshot as `mlflow-task/screenshot.png`, or replace
> the image link above with the path to your screenshot.

Comparison of the three runs (validation set, 20%). Lower RMSE is better.

| Run   | Max Depth | Learning Rate | RMSE     | MAE      | R²     |
|-------|-----------|---------------|----------|----------|--------|
| Run 1 | 3         | 0.1           | 55878.17 | 38249.31 | 0.7617 |
| Run 2 | 5         | 0.05          | 54088.28 | 36710.46 | 0.7767 |
| Run 3 | 7         | 0.01          | 69746.25 | 52707.83 | 0.6288 |

## Selected Best Model

**Run 2 — `max_depth=5`, `learning_rate=0.05`** (RMSE = 54088.28).

### Why

Run 2 has the **lowest RMSE** (the primary selection metric), and it also
achieves the best MAE and highest R². Run 3 uses a very small learning rate
(0.01) that, with the default 100 boosting estimators, is not enough to fit the
data well, so it underfits and performs worst. Run 1 is close but slightly
worse than Run 2. Run 2 offers the best bias/variance trade-off of the three
configurations.
