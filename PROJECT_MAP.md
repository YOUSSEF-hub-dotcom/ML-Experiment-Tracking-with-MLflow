# PROJECT_MAP

## TECH_STACK
| Component        | Library / Tool              | Version |
|------------------|-----------------------------|---------|
| Language         | Python                      | 3.11.8  |
| Experiment track | mlflow                      | 3.17.0  |
| ML model         | scikit-learn (GradientBoostingRegressor) | 1.9.1 |
| Data handling    | pandas                      | 3.0.6   |
| Numerics         | numpy                       | 2.4.6   |
| Environment      | venv `mlflow_env`           | -       |
| Dataset          | California Housing (`housing.csv`, 20640 rows) | - |
| Tracking store   | local `mlruns/`             | -       |

## SYSTEM_FLOW
1. Read `housing.csv`.
2. Preprocess: median-impute `total_bedrooms`, one-hot encode `ocean_proximity`.
3. Split features/target, 80% train / 20% validation, `random_state=42`.
4. For each config `(max_depth, learning_rate)` in `[(3,0.1), (5,0.05), (7,0.01)]`:
   a. Train `GradientBoostingRegressor`.
   b. Evaluate on validation: RMSE, MAE, R2.
   c. Log params, metrics, and model artifact to MLflow.
5. Compare runs, select best by lowest RMSE.
6. Inspect results in the MLflow UI (`mlflow ui`).

Pipeline: Train -> Track -> Compare -> Select.
