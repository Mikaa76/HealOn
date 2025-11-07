
# This is to train model for hearth and diabetes risk

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

#  Load 
df = pd.read_csv("./data/diabetes/health_risk.csv")

# Columns
categorical_cols = ["gender", "blood_group", "chest pain","family_history"]
numeric_cols = [col for col in df.columns if col not in categorical_cols + ["risk_score"]]

X = df[categorical_cols + numeric_cols]
y = df["risk_score"]

# Preprocessor 
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Candidate Models 
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
}

best_model = None
best_score = -np.inf

for name, regressor in models.items():
    print(f"\nTraining {name}...")

    # Build pipeline: preprocessing + model
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", regressor)
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} -> MSE={mse:.4f}, R²={r2:.4f}")

    # Save best
    if r2 > best_score:
        best_score = r2
        best_model = pipeline
        joblib.dump(best_model, "health_risk.pkl")
        print(f"✅ Saved {name} as best pipeline with R²={r2:.4f}")

print("\nBest model pipeline saved as health_risk.pkl")
