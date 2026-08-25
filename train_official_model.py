"""Train Mannina Maga's yield model from a prepared official/verified merged CSV.

Expected file: data/official_merged.csv
Required columns:
  crop, rainfall_mm, temp_c, humidity_pct, nitrogen, phosphorus,
  potassium, ph, yield_t_ha

Recommended provenance:
  - production/area/yield: Government of India OGD crop production statistics
  - historical rainfall: IMD
  - temperature/humidity/soil: measured or clearly labeled estimates
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import sklearn

BASE = Path(__file__).resolve().parent
CSV = BASE / "data" / "official_merged.csv"
MODEL = BASE / "models" / "yield_model.joblib"
META = BASE / "models" / "model_metadata.json"
REQUIRED = ["crop","rainfall_mm","temp_c","humidity_pct","nitrogen","phosphorus","potassium","ph","yield_t_ha"]

if not CSV.exists():
    raise SystemExit("Place your cleaned official merged dataset at data/official_merged.csv")

df = pd.read_csv(CSV)
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {', '.join(missing)}")

df = df[REQUIRED].dropna().copy()
if len(df) < 200:
    raise SystemExit("Use at least 200 clean records for a meaningful model.")

X = df.drop(columns=["yield_t_ha"])
y = df["yield_t_ha"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ("crop", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["crop"]),
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", pre),
    ("regressor", GradientBoostingRegressor(random_state=42, n_estimators=220, learning_rate=0.04, max_depth=3, loss="huber")),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
MODEL.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL)
META.write_text(json.dumps({
    "model": "GradientBoostingRegressor",
    "training_data": "Official/verified merged dataset",
    "records": int(len(df)),
    "r2": round(float(r2_score(y_test, pred)), 4),
    "mae": round(float(mean_absolute_error(y_test, pred)), 4),
    "sklearn_version": sklearn.__version__,
}, indent=2), encoding="utf-8")
print("Model trained and saved to", MODEL)
print(META.read_text(encoding="utf-8"))
