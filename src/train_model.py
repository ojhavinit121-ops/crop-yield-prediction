"""Train and evaluate a crop-yield regression model. Run: python src/train_model.py"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "yield_features.csv"
REPORTS_DIR = ROOT / "reports"
TARGET = "hg/ha_yield"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Add the modeling CSV at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.drop_duplicates().dropna(subset=[TARGET])
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocess),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42,
                                        n_jobs=-1, min_samples_leaf=2)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "model": "RandomForestRegressor",
        "target": TARGET,
        "test_rows": int(len(y_test)),
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, REPORTS_DIR / "crop_yield_random_forest.joblib")
    (REPORTS_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
