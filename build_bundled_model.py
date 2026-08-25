from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "yield_model.joblib"
META_PATH = BASE / "models" / "model_metadata.json"

CROPS = {
    "Rice":        {"temp": (22,34),"rain":(1000,2500),"humidity":(60,90),"ph":(5.0,7.0),"n":80,"p":40,"k":40,"base":4.2},
    "Wheat":       {"temp": (10,25),"rain":(300,900),"humidity":(40,70),"ph":(6.0,7.5),"n":70,"p":35,"k":35,"base":3.4},
    "Maize":       {"temp": (18,30),"rain":(500,1200),"humidity":(40,75),"ph":(5.5,7.5),"n":65,"p":35,"k":40,"base":4.5},
    "Cotton":      {"temp": (21,35),"rain":(500,1200),"humidity":(40,70),"ph":(5.8,8.0),"n":60,"p":30,"k":45,"base":2.2},
    "Soybean":     {"temp": (20,30),"rain":(500,1000),"humidity":(45,75),"ph":(6.0,7.5),"n":40,"p":40,"k":40,"base":2.8},
    "Groundnut":   {"temp": (20,32),"rain":(400,1000),"humidity":(40,70),"ph":(5.5,7.0),"n":35,"p":35,"k":45,"base":2.3},
    "Sugarcane":   {"temp": (20,35),"rain":(750,1500),"humidity":(50,80),"ph":(6.0,8.0),"n":90,"p":50,"k":60,"base":70.0},
    "Jowar":       {"temp": (25,32),"rain":(400,800),"humidity":(35,65),"ph":(5.5,8.0),"n":45,"p":25,"k":30,"base":2.5},
    "Bajra":       {"temp": (25,35),"rain":(250,650),"humidity":(30,60),"ph":(5.5,7.5),"n":40,"p":22,"k":28,"base":2.1},
    "Ragi":        {"temp": (20,30),"rain":(500,1000),"humidity":(45,75),"ph":(5.0,7.5),"n":45,"p":25,"k":30,"base":2.4},
    "Pigeon Pea":  {"temp": (20,35),"rain":(600,1000),"humidity":(40,70),"ph":(5.0,7.5),"n":30,"p":35,"k":30,"base":1.7},
    "Chickpea":    {"temp": (15,30),"rain":(300,650),"humidity":(30,60),"ph":(6.0,8.0),"n":25,"p":35,"k":30,"base":1.8},
    "Sunflower":   {"temp": (20,30),"rain":(400,750),"humidity":(35,65),"ph":(6.0,7.5),"n":45,"p":30,"k":40,"base":1.9},
    "Onion":       {"temp": (13,30),"rain":(350,700),"humidity":(45,75),"ph":(6.0,7.5),"n":70,"p":40,"k":60,"base":18.0},
    "Tomato":      {"temp": (18,30),"rain":(400,800),"humidity":(45,75),"ph":(5.5,7.5),"n":75,"p":45,"k":65,"base":28.0},
}

rng = np.random.default_rng(42)
rows = []

def range_score(v, lo, hi):
    if lo <= v <= hi:
        return 1.0
    d = min(abs(v-lo), abs(v-hi))
    return max(0.0, 1.0 - d/max(hi-lo, 1))

for crop, p in CROPS.items():
    for _ in range(550):
        temp = rng.uniform(p["temp"][0]-5, p["temp"][1]+5)
        rain = rng.uniform(max(50,p["rain"][0]-250), p["rain"][1]+300)
        hum = rng.uniform(max(15,p["humidity"][0]-15), min(95,p["humidity"][1]+15))
        ph = rng.uniform(max(4,p["ph"][0]-0.8), min(9,p["ph"][1]+0.8))
        n = max(5, rng.normal(p["n"], max(8,p["n"]*0.22)))
        phos = max(5, rng.normal(p["p"], max(6,p["p"]*0.22)))
        k = max(5, rng.normal(p["k"], max(7,p["k"]*0.22)))

        env = (
            range_score(temp,*p["temp"]) +
            range_score(rain,*p["rain"]) +
            range_score(hum,*p["humidity"]) +
            range_score(ph,*p["ph"]) +
            max(0,1-abs(n-p["n"])/max(p["n"],1)) +
            max(0,1-abs(phos-p["p"])/max(p["p"],1)) +
            max(0,1-abs(k-p["k"])/max(p["k"],1))
        ) / 7

        y = p["base"] * (0.48 + 0.62*env) * rng.normal(1.0, 0.035)
        rows.append({
            "crop":crop, "rainfall_mm":rain, "temp_c":temp, "humidity_pct":hum,
            "nitrogen":n, "phosphorus":phos, "potassium":k, "ph":ph,
            "yield_t_ha":max(0.05,y)
        })

df = pd.DataFrame(rows)
X = df.drop(columns=["yield_t_ha"])
y = df["yield_t_ha"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

pre = ColumnTransformer([
    ("crop", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["crop"]),
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", pre),
    ("regressor", GradientBoostingRegressor(random_state=42,n_estimators=240,learning_rate=0.04,max_depth=3,loss="huber")),
])
model.fit(X_train,y_train)
pred = model.predict(X_test)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
META_PATH.write_text(json.dumps({
    "r2": round(float(r2_score(y_test,pred)),4),
    "mae": round(float(mean_absolute_error(y_test,pred)),4),
    "records": int(len(df)),
    "crops": list(CROPS.keys()),
    "crop_count": len(CROPS),
    "model": "GradientBoostingRegressor",
    "sklearn_version": sklearn.__version__,
    "training_data": "Bundled calibration sample for 15 supported crops",
    "official_retraining_supported": True
}, indent=2), encoding="utf-8")

print(f"Saved model for {len(CROPS)} crops to {MODEL_PATH}")
