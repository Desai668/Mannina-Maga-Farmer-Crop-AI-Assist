# Mannina Maga Data Sources

Mannina Maga keeps data provenance explicit.

## Selected official sources for production use / retraining

1. Government of India - Open Government Data Platform
   - Dataset: District-wise, season-wise crop production statistics from 1997
   - Publisher: Ministry of Agriculture & Farmers Welfare / Directorate of Economics and Statistics
   - Page: https://www.data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997

2. India Meteorological Department (IMD)
   - Dataset: 0.25 x 0.25 degree daily gridded rainfall over India
   - Page: https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html

3. Current weather
   - Open-Meteo Forecast API and Geocoding API
   - https://open-meteo.com/en/docs
   - https://open-meteo.com/en/docs/geocoding-api

## Bundled model note

The repository ships with a scikit-learn model so the app works immediately after cloning.
That starter model is trained on `ml_training_sample.csv`, a deterministic calibration dataset generated from crop requirement ranges. It is NOT claimed to be Government of India or IMD measured data.

For a research/final deployment, replace the starter model with a model trained on a cleaned, merged official dataset. Use `train_official_model.py` after preparing `data/official_merged.csv`.

Recommended merged columns:

- crop
- rainfall_mm
- temp_c
- humidity_pct
- nitrogen
- phosphorus
- potassium
- ph
- yield_t_ha

Production and area/yield should come from Government of India records. Historical rainfall should come from IMD. Temperature, humidity and soil features must be marked as measured, farmer-entered or estimated rather than presented as official if they are not directly sourced.
