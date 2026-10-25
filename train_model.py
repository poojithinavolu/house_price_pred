import os
# Prevent MLflow 3.x filestore warning on Windows/Docker
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import pickle
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. PREPARE THE DATASET
# ==========================================
data = {
    "area": [
        500, 600, 750, 800, 900,
        1000, 1100, 1200, 1300, 1400,
        1500, 1600, 1700, 1800, 2000,
        2200, 2400, 2600, 2800, 3000
    ],
    "bedrooms": [
        1, 1, 2, 2, 2,
        2, 3, 3, 3, 3,
        3, 3, 4, 4, 4,
        4, 4, 5, 5, 5
    ],
    "age": [
        15, 12, 10, 8, 10,
        7, 8, 6, 5, 7,
        5, 4, 6, 3, 4,
        3, 2, 3, 2, 1
    ],
    "price": [
        2000000, 2400000, 3000000, 3200000, 3500000,
        4000000, 4200000, 4700000, 5000000, 5200000,
        5700000, 6100000, 6500000, 7000000, 7600000,
        8200000, 9000000, 9600000, 10500000, 11500000
    ]
}

df = pd.DataFrame(data)
X = df[["area", "bedrooms", "age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 2. DEFINE MLFLOW EXPERIMENT
# ==========================================
# An Experiment is the folder/project that groups all related runs together
mlflow.set_experiment("House_Price_Prediction_Comparison")

# Define 3 candidate models to compare
experiments = [
    {
        "run_name": "Run_1_LinearRegression",
        "model_type": "LinearRegression",
        "model": LinearRegression(),
        "params": {"fit_intercept": True}
    },
    {
        "run_name": "Run_2_DecisionTree",
        "model_type": "DecisionTreeRegressor",
        "model": DecisionTreeRegressor(max_depth=4, random_state=42),
        "params": {"max_depth": 4, "criterion": "squared_error"}
    },
    {
        "run_name": "Run_3_RandomForest",
        "model_type": "RandomForestRegressor",
        "model": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
        "params": {"n_estimators": 50, "max_depth": 5}
    }
]

# ==========================================
# 3. TRAIN & LOG ALL 3 RUNS IN MLFLOW
# ==========================================
best_r2 = -float("inf")
best_model = None

for exp in experiments:
    # with mlflow.start_run starts tracking a fresh trial
    with mlflow.start_run(run_name=exp["run_name"]):
        print(f"\n🚀 Training: {exp['run_name']}...")
        
        # Train model
        model = exp["model"]
        model.fit(X_train, y_train)
        
        # Predict & Evaluate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # -----------------------------------------------
        # MLFLOW LOGGING: 3 Core Pillars of MLflow Tracking
        # -----------------------------------------------
        # 1. Log Parameters (Inputs / Hyperparameters)
        mlflow.log_param("model_type", exp["model_type"])
        for param_name, param_val in exp["params"].items():
            mlflow.log_param(param_name, param_val)
        
        # 2. Log Metrics (Outputs / Evaluation Scores)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # 3. Log Model Artifact (Save the model file inside MLflow)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"   Logged -> MAE: ₹{mae:,.2f} | R²: {r2:.4f}")

        # Keep track of the winning model for our Streamlit app
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

# ==========================================
# 4. EXPORT THE BEST MODEL FOR STREAMLIT APP
# ==========================================
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n🏆 Successfully logged 3 runs to MLflow!")
print(f"Saved winning model to house_price_model.pkl (Best R²: {best_r2:.4f})")