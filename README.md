# Crop Yield Prediction

A beginner-friendly machine learning project to estimate crop yield from historical agricultural and climate-related data.

## Project workflow
1. Load and inspect the dataset.
2. Check missing values and duplicate records.
3. Explore numerical and categorical features.
4. Split data into training and test sets.
5. Preprocess features and train a Random Forest regressor.
6. Evaluate using MAE, RMSE, and R².

## Dataset
Put the supplied CSV files in `data/`:
- `yield_features.csv` — modeling dataset; target column: `hg/ha_yield`.
- `yield_raw.csv` — additional historical dataset.

**Note:** The CSV files are not included in this initial GitHub upload. Download the prepared ZIP from this ChatGPT conversation and copy both CSVs into `data/`. Confirm the original data source and license before redistribution.

## Setup
Python 3.9 or later:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
From the repository root:

```bash
python src/train_model.py
```

The script prints MAE, RMSE, and R², then saves the trained pipeline and metrics in `reports/`.

## Model notes
A Random Forest baseline is used. `Area` and `Item` are one-hot encoded. Preprocessing is inside a scikit-learn pipeline to avoid leakage. A random split is a baseline; use a time-based split when testing future-year forecasting.

Run the model before reporting performance. No accuracy score is claimed in advance.

## Author
**Vinit Ojha** — Aspiring Data Scientist
