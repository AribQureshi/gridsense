

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------
sns.set_theme(style="whitegrid")
RAW_PATH = "data/raw/long_data_.csv"
PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

STATE = "Delhi"  # <-- change this to any state in the dataset if you want

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
print("Raw shape:", df.shape)
print(df.head())

# Clean column names, parse dates
df.columns = [c.strip() for c in df.columns]
df["Dates"] = pd.to_datetime(df["Dates"], format="%d/%m/%Y %H:%M:%S")

# ---------------------------------------------------------------------
# 2. Filter down to ONE state — this becomes our whole project's dataset
# ---------------------------------------------------------------------
state_df = df[df["States"] == STATE].copy()
state_df = state_df[["Dates", "Usage"]].rename(columns={"Dates": "date", "Usage": "demand_mu"})

# Real scraped data has a few duplicate dates (double-reported rows) —
# average them instead of dropping, so we don't lose information.
n_before = len(state_df)
state_df = state_df.groupby("date", as_index=False)["demand_mu"].mean()
state_df = state_df.sort_values("date").reset_index(drop=True)
print(f"Collapsed {n_before - len(state_df)} duplicate-date rows via mean")

print(f"\n{STATE} data shape:", state_df.shape)
print("Full date range in file:", state_df["date"].min(), "to", state_df["date"].max())

# ---------------------------------------------------------------------
# 3. Data quality check — IMPORTANT FINDING
# ---------------------------------------------------------------------
# Inspecting the gaps between consecutive observed dates shows the data is
# dense and near-daily from Jan 2019 through June 2020, but from July 2020
# onward it drops to roughly ONE point per month (a 26-27 day gap every
# time). That's not "missing daily data" — it's a different, much coarser
# sampling rate, and filling it in with interpolation would fabricate a
# smooth trend that never happened.
#
# Decision: scope the project to the reliable, near-daily window
# (Jan 2019 -> Jun 2020) and drop the sparse monthly-snapshot tail.
# This kind of scoping decision -- and writing down WHY -- is exactly the
# judgment call a real data scientist has to make and document.
CUTOFF = "2020-06-30"
dropped = (state_df["date"] > CUTOFF).sum()
state_df = state_df[state_df["date"] <= CUTOFF].reset_index(drop=True)
print(f"Dropped {dropped} sparse monthly-snapshot rows after {CUTOFF}")

full_range = pd.date_range(state_df["date"].min(), state_df["date"].max(), freq="D")
missing_dates = full_range.difference(state_df["date"])
print(f"Missing calendar days in kept window: {len(missing_dates)} out of {len(full_range)} "
      f"({len(missing_dates)/len(full_range):.1%})")

# Now interpolation is safe: remaining gaps are short (max ~9 days),
# so a linear fill is a reasonable approximation, not a fabrication.
state_df = state_df.set_index("date").reindex(full_range)
state_df.index.name = "date"
state_df["was_interpolated"] = state_df["demand_mu"].isna()
state_df["demand_mu"] = state_df["demand_mu"].interpolate(method="linear")
state_df = state_df.reset_index()

# ---------------------------------------------------------------------
# 4. PLOT 1 — Full time series (look for trend, seasonality, the COVID dip)
# ---------------------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(state_df["date"], state_df["demand_mu"], color="#2563eb", linewidth=1, zorder=2)
interp_pts = state_df[state_df["was_interpolated"]]
plt.scatter(interp_pts["date"], interp_pts["demand_mu"], color="orange", s=15,
            zorder=3, label="Interpolated (short gap ≤9 days)")
plt.axvspan(pd.Timestamp("2020-03-25"), pd.Timestamp("2020-05-23"),
            color="red", alpha=0.1, label="COVID lockdown period")
plt.title(f"{STATE} Daily Electricity Demand (Jan 2019 \u2013 Jun 2020)")
plt.xlabel("Date")
plt.ylabel("Demand (Million Units)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_full_timeseries.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 5. PLOT 2 — Monthly seasonality (boxplot by month)
# ---------------------------------------------------------------------
state_df["month"] = state_df["date"].dt.month_name().str[:3]
state_df["month_num"] = state_df["date"].dt.month
month_order = state_df.sort_values("month_num")["month"].unique()

plt.figure(figsize=(12, 5))
sns.boxplot(data=state_df, x="month", y="demand_mu", order=month_order, color="#93c5fd")
plt.title(f"{STATE} Demand Distribution by Month (seasonality check)")
plt.xlabel("Month")
plt.ylabel("Demand (MU)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_monthly_seasonality.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 6. PLOT 3 — Day-of-week pattern (weekday vs weekend)
# ---------------------------------------------------------------------
state_df["day_of_week"] = state_df["date"].dt.day_name()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

plt.figure(figsize=(10, 5))
sns.boxplot(data=state_df, x="day_of_week", y="demand_mu", order=dow_order, color="#86efac")
plt.title(f"{STATE} Demand by Day of Week")
plt.xlabel("")
plt.ylabel("Demand (MU)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_day_of_week.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 7. Save the cleaned, single-state dataset for the next step (feature engineering)
# ---------------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
state_df[["date", "demand_mu", "was_interpolated"]].to_csv(
    "data/processed/delhi_demand_daily.csv", index=False
)

print("\nSaved 3 plots to outputs/plots/")
print("Saved cleaned dataset to data/processed/delhi_demand_daily.csv")
print("\nSummary stats:")
print(state_df["demand_mu"].describe())
