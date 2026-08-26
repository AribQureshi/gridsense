"""
GridSense — Diagnostic 2: Visualize the temperature-demand relationship directly
====================================================================================
The lockdown-confound theory was just ruled out (correlations barely moved
when lockdown days were excluded). That leaves the other explanation from
Step 1: temperature vs demand is U-shaped, and Pearson correlation is
mathematically blind to U-shapes (positive and negative deviations cancel).

This script tests that directly by binning temperature into deciles and
plotting average demand per bin. A U-shape here will *prove* the theory
instead of just asserting it.

Run this with:  python3 src/check_temperature_shape.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/processed/gridsense_features.csv", parse_dates=["date"])

# Bin temperature into 10 equal-sized groups (deciles)
df["temp_decile"] = pd.qcut(df["temp_mean_c"], q=10)

bin_means = df.groupby("temp_decile", observed=True)["demand_mu"].mean()
print("Average demand by temperature decile:")
print(bin_means.round(1))

plt.figure(figsize=(11, 5))
bin_means.plot(kind="bar", color="#2563eb")
plt.title("Average Demand by Temperature Decile — Is it U-shaped?")
plt.xlabel("Temperature range (°C)")
plt.ylabel("Average Demand (MU)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("outputs/plots/04_temperature_demand_shape.png", dpi=120)
plt.close()

# Quantify it properly: fit demand ~ temp (linear only) vs demand ~ temp + temp^2
# and compare R^2 -- if the quadratic term meaningfully improves fit, that's
# hard evidence of the U-shape, not just a visual impression.
import numpy as np
from numpy.polynomial import polynomial as P

x = df["temp_mean_c"].values
y = df["demand_mu"].values

# Linear fit
lin_coef = np.polyfit(x, y, 1)
lin_pred = np.polyval(lin_coef, x)
lin_r2 = 1 - np.sum((y - lin_pred)**2) / np.sum((y - y.mean())**2)

# Quadratic fit
quad_coef = np.polyfit(x, y, 2)
quad_pred = np.polyval(quad_coef, x)
quad_r2 = 1 - np.sum((y - quad_pred)**2) / np.sum((y - y.mean())**2)

print(f"\nR² using temperature only (linear):    {lin_r2:.4f}")
print(f"R² using temperature + temperature² :  {quad_r2:.4f}")
print(f"Improvement from adding the squared term: {quad_r2 - lin_r2:.4f}")
print(f"\nQuadratic coefficient sign: {'positive (U-shape confirmed)' if quad_coef[0] > 0 else 'negative (inverted-U)'}")

print("\nSaved plot to outputs/plots/04_temperature_demand_shape.png")
