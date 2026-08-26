

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
