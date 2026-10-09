"""ML Experiment Tracking with MLflow.

Train a single scikit-learn model (GradientBoostingRegressor) three times with
different hyperparameters on the California Housing dataset, tracking params,
metrics (RMSE / MAE / R2) and the trained model as an MLflow artifact.
The best run is selected by lowest validation RMSE.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "median_house_value"
EXPERIMENT_NAME = "california-housing"

# The three hyperparameter configurations from the task specification.
CONFIGS = [
    {"max_depth": 3, "learning_rate": 0.1},
    {"max_depth": 5, "learning_rate": 0.05},
    {"max_depth": 7, "learning_rate": 0.01},
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "housing.csv"


def load_data(csv_path):
    """Load the dataset and return features (X) and target (y)."""
    df = pd.read_csv(csv_path)
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
    df = pd.get_dummies(df, columns=["ocean_proximity"])
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def evaluate(model, X, y):
    """Return validation RMSE, MAE and R2."""
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    parser = argparse.ArgumentParser(description="MLflow experiment tracking demo.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to housing.csv")
    args = parser.parse_args()

    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data(args.data)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []
    for idx, config in enumerate(CONFIGS, start=1):
        run_name = f"Run {idx}"
        with mlflow.start_run(run_name=run_name):
            model = GradientBoostingRegressor(
                max_depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_val, y_val)

            mlflow.log_param("max_depth", config["max_depth"])
            mlflow.log_param("learning_rate", config["learning_rate"])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

        results.append({"run": run_name, **config, **metrics})
        print(
            f"{run_name}: max_depth={config['max_depth']} "
            f"learning_rate={config['learning_rate']} "
            f"RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f} R2={metrics['r2']:.4f}"
        )

    best = min(results, key=lambda r: r["rmse"])
    print("\nComparison (sorted by RMSE, lower is better):")
    print(f"{'Run':<6}{'Max Depth':<11}{'LR':<8}{'RMSE':<14}{'MAE':<14}{'R2'}")
    for row in sorted(results, key=lambda r: r["rmse"]):
        print(
            f"{row['run']:<6}{row['max_depth']:<11}{row['learning_rate']:<8}"
            f"{row['rmse']:<14.2f}{row['mae']:<14.2f}{row['r2']:.4f}"
        )
    print(
        f"\nBest run: {best['run']} "
        f"(max_depth={best['max_depth']}, learning_rate={best['learning_rate']}, "
        f"RMSE={best['rmse']:.2f})"
    )


if __name__ == "__main__":
    main()
