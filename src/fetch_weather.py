

import requests
import pandas as pd
import os

# ---------------------------------------------------------------------
# 0. Setup — Delhi's coordinates (same ones present in the POSOCO dataset)
# ---------------------------------------------------------------------
LATITUDE = 28.6699929
LONGITUDE = 77.23000403

# Match the exact window we decided on in Step 1 (EDA)
START_DATE = "2019-01-02"
END_DATE = "2020-06-30"

OUT_PATH = "data/raw/delhi_weather.csv"

# ---------------------------------------------------------------------
# 1. Build the request
# ---------------------------------------------------------------------
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily": ",".join([
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
    ]),
    "timezone": "Asia/Kolkata",
}

print("Requesting weather data from Open-Meteo...")
response = requests.get(url, params=params, timeout=30)
response.raise_for_status()  # crash loudly if the request failed, don't fail silently
data = response.json()

# ---------------------------------------------------------------------
# 2. Parse the JSON response into a DataFrame
# ---------------------------------------------------------------------
daily = data["daily"]
weather_df = pd.DataFrame({
    "date": pd.to_datetime(daily["time"]),
    "temp_max_c": daily["temperature_2m_max"],
    "temp_min_c": daily["temperature_2m_min"],
    "temp_mean_c": daily["temperature_2m_mean"],
    "humidity_mean_pct": daily["relative_humidity_2m_mean"],
    "precipitation_mm": daily["precipitation_sum"],
})

print(f"Retrieved {len(weather_df)} days of weather data")
print(weather_df.head())
print("\nAny missing values?")
print(weather_df.isna().sum())

# ---------------------------------------------------------------------
# 3. Save
# ---------------------------------------------------------------------
os.makedirs("data/raw", exist_ok=True)
weather_df.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")

print("\nQuick sanity check — summer vs winter temps (should differ a lot for Delhi):")
weather_df["month"] = weather_df["date"].dt.month
print(weather_df.groupby("month")["temp_mean_c"].mean().round(1))
