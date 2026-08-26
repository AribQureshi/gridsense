"""
GridSense — Step 3: Feature Engineering
==========================================
Goal:
    Merge demand + weather data and build the feature set our regression
    models will actually use. Every feature here is deliberate — each one
    exists because Step 1's EDA gave us a specific reason to add it.

Where each feature comes from (traceability matters for your README):
    - temp_mean_c, temp_max_c, temp_min_c, humidity, precipitation
        -> straight from Step 2's weather pull
    - temp_mean_c_sq (temperature squared)
        -> Step 1 showed demand is U-shaped in temperature (high in both
           cold Jan/Dec AND hot May/Jun) — plain linear regression can't
           capture a U-shape, but adding temp^2 as a feature lets a LINEAR
           model fit a curve. This is the textbook trick for polynomial
           regression without leaving the linear-model family.
    - HDD / CDD (heating/cooling degree days)
        -> the actual industry-standard way utilities featurize temperature.
           HDD = how far below a comfort baseline (18C) the day was -> heating load
           CDD = how far above that baseline -> cooling/AC load
           This is more interpretable than raw temperature for a reader.
    - day_of_year_sin / cos (cyclic encoding)
        -> Step 1 showed strong seasonality. Encoding day-of-year as a raw
           integer (1-365) is wrong: Dec 31 (day 365) and Jan 1 (day 1) are
           adjacent in reality but far apart numerically. Sin/cos encoding
           fixes that by mapping the day onto a circle.
    - demand_lag_1, demand_lag_7
        -> yesterday's demand and same-day-last-week are strong predictors
           of today's demand in any load forecasting setting. This also
           gives us a natural "naive baseline" to compare our model against.
    - is_lockdown
        -> Step 1 found a real demand shock during the COVID lockdown
           (2020-03-25 to 2020-05-23). Without flagging this, the model
           will try to "explain" the shock using temperature/season and
           get it wrong. This flag lets the model isolate it instead.
    - is_weekend
        -> Step 1's day-of-week plot showed almost NO weekend effect for
           this state-level data, but we keep the feature anyway and let
           the model/diagnostics CONFIRM it's not useful (near-zero
           coefficient) rather than assuming this from the plot alone.
           That's a more rigorous way to make the same point in the report.

Run this with:  python3 src/build_features.py
"""

import pandas as pd
import numpy as np
import os

DEMAND_PATH = "data/processed/delhi_demand_daily.csv"
WEATHER_PATH = "data/raw/delhi_weather.csv"
OUT_PATH = "data/processed/gridsense_features.csv"

COMFORT_BASELINE_C = 18.0  # standard HDD/CDD reference temperature
LOCKDOWN_START = "2020-03-25"
LOCKDOWN_END = "2020-05-23"

# ---------------------------------------------------------------------
# 1. Load and merge
# ---------------------------------------------------------------------
demand = pd.read_csv(DEMAND_PATH, parse_dates=["date"])
weather = pd.read_csv(WEATHER_PATH, parse_dates=["date"])

print("Demand rows:", len(demand), "| Weather rows:", len(weather))

df = pd.merge(demand, weather, on="date", how="inner")
print("Merged rows (inner join, only dates present in both):", len(df))

df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------------
# 2. Temperature-derived features
# ---------------------------------------------------------------------
df["temp_mean_c_sq"] = df["temp_mean_c"] ** 2

df["hdd"] = (COMFORT_BASELINE_C - df["temp_mean_c"]).clip(lower=0)  # heating degree days
df["cdd"] = (df["temp_mean_c"] - COMFORT_BASELINE_C).clip(lower=0)  # cooling degree days

# ---------------------------------------------------------------------
# 3. Cyclic time encoding
# ---------------------------------------------------------------------
df["day_of_year"] = df["date"].dt.dayofyear
df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

# ---------------------------------------------------------------------
# 4. Lag features (need sorted, gap-aware data — we built a continuous
#    daily index in Step 1, so simple .shift() is safe here)
# ---------------------------------------------------------------------
df["demand_lag_1"] = df["demand_mu"].shift(1)
df["demand_lag_7"] = df["demand_mu"].shift(7)

# ---------------------------------------------------------------------
# 5. Calendar / event flags
# ---------------------------------------------------------------------
df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
df["is_lockdown"] = df["date"].between(LOCKDOWN_START, LOCKDOWN_END).astype(int)

# ---------------------------------------------------------------------
# 6. Drop rows with NaNs introduced by lag features (first 7 rows)
# ---------------------------------------------------------------------
n_before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"Dropped {n_before - len(df)} rows with missing lag values (start of series)")

# ---------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print(f"\nFinal feature set shape: {df.shape}")
print("Columns:", list(df.columns))
print(f"\nSaved to {OUT_PATH}")

print("\nFeature summary stats:")
print(df[["demand_mu", "temp_mean_c", "hdd", "cdd", "demand_lag_1", "demand_lag_7"]].describe().round(2))

print("\nQuick correlation check with target (demand_mu):")
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("demand_mu")
corrs = df[numeric_cols].corrwith(df["demand_mu"]).sort_values(key=abs, ascending=False)
print(corrs.round(3))
