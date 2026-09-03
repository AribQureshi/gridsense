"""
GridSense — Step 4: Modeling, Regularization, and Diagnostics
==================================================================
This is the technical core of the project. It answers three questions,
in order:

  Q1. Does weather add real predictive power on top of lag features alone?
      (We found weather's raw correlation with demand is weak -- but lag
      features might already explain away some of what weather explains.
      We test this properly with a nested model comparison.)

  Q2. Among Ridge / Lasso / ElasticNet, which regularization approach fits
      this data best, and why does that make sense given our features?
      (We have correlated weather features -- temp_mean_c, hdd, cdd, and
      temp_mean_c_sq are all functions of the same underlying temperature,
      so we EXPECT multicollinearity. That's exactly Ridge's use case.)

  Q3. Is the final model statistically sound?
      (VIF for multicollinearity, Breusch-Pagan for heteroscedasticity,
      Q-Q plot for residual normality, Cook's distance for influential
      outliers.)

IMPORTANT: This is time series data, so we do NOT use a random train/test
split (that would let the model "see the future" via nearby days leaking
between train and test). We use a chronological split and TimeSeriesSplit
for cross-validation.

Run this with:  python3 src/train_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid")

DATA_PATH = "data/processed/gridsense_features.csv"
PLOTS_DIR = "outputs/plots"
MODELS_DIR = "outputs/models"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load data and define feature sets
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

TARGET = "demand_mu"

BASELINE_FEATURES = ["demand_lag_1", "demand_lag_7", "is_weekend", "is_lockdown",
                      "day_of_year_sin", "day_of_year_cos"]

WEATHER_FEATURES = ["temp_mean_c", "temp_mean_c_sq", "hdd", "cdd",
                     "humidity_mean_pct", "precipitation_mm"]

FULL_FEATURES = BASELINE_FEATURES + WEATHER_FEATURES

# ---------------------------------------------------------------------
# 2. Chronological train/test split (time series -> no shuffling!)
#
# IMPORTANT: a plain 80/20 split on this data puts the ENTIRE COVID
# lockdown period inside the test set (since it happened near the end of
# our date range). That would mean `is_lockdown` is a constant 0 column
# during training -- the model could never learn to use it, and several
# diagnostics (VIF, coefficient estimates) become degenerate or blow up.
#
# Fix: split on a fixed date chosen so BOTH train and test contain some
# lockdown days. This is still a legitimate chronological split (no
# future leakage) -- we're just choosing WHERE to cut, not shuffling.
# ---------------------------------------------------------------------
TRAIN_END_DATE = "2020-04-20"  # falls in the middle of the lockdown window
train_df = df[df["date"] <= TRAIN_END_DATE]
test_df = df[df["date"] > TRAIN_END_DATE]

# Safety check: make sure no feature is constant within the training set.
# A constant column has zero variance, which breaks VIF (division by zero)
# and gives regularized models nothing to learn from. Better to catch this
# loudly now than debug a cryptic NaN three steps later.
constant_cols = [c for c in BASELINE_FEATURES + WEATHER_FEATURES
                  if train_df[c].nunique() <= 1]
if constant_cols:
    raise ValueError(
        f"These features are constant in the training set and must be "
        f"handled before modeling: {constant_cols}. Consider adjusting "
        f"TRAIN_END_DATE so both splits contain variation in these columns."
    )
print(f"Train: {len(train_df)} days ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
print(f"Test:  {len(test_df)} days ({test_df['date'].min().date()} to {test_df['date'].max().date()})")


def evaluate(y_true, y_pred):
    """Compute RMSE, MAE, and R^2 on held-out data.

    NOTE: We deliberately do NOT report Adjusted R^2 for held-out test
    evaluation. Adjusted R^2 is designed to penalize extra parameters
    relative to the SAMPLE SIZE USED TO FIT the model -- it's an in-sample
    overfitting diagnostic, not an out-of-sample comparison metric. Our
    test set is only 46 days; applying the adjusted-R^2 formula there
    produces a wildly oversized, misleading penalty for models with more
    features (as we saw: RMSE/MAE both improved with weather features
    added, while a naively-computed "test Adjusted R^2" went down). RMSE
    and MAE are the correct, standard metrics for comparing models on
    held-out data. Adjusted R^2 is reported separately, correctly, from
    the statsmodels OLS summary on the TRAINING data below.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ---------------------------------------------------------------------
# 3. Q1 — Does weather help? Nested model comparison (plain OLS, no
#    regularization yet -- keep this comparison simple and interpretable)
# ---------------------------------------------------------------------
results = {}

for name, feats in [("Baseline (lag + calendar only)", BASELINE_FEATURES),
                     ("Baseline + Weather", FULL_FEATURES)]:
    X_train, y_train = train_df[feats], train_df[TARGET]
    X_test, y_test = test_df[feats], test_df[TARGET]

    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = evaluate(y_test, preds)

print("\n=== Q1: Does weather add predictive power? (OLS, held-out test set) ===")
print(pd.DataFrame(results).T.round(4))

# ---------------------------------------------------------------------
# 4. Full statsmodels OLS on the full feature set — for p-values,
#    confidence intervals, and coefficient interpretation
# ---------------------------------------------------------------------
X_train_sm = sm.add_constant(train_df[FULL_FEATURES])
ols_model = sm.OLS(train_df[TARGET], X_train_sm).fit()
print("\n=== OLS Summary (full feature set, train data) ===")
print(ols_model.summary())

with open("outputs/ols_summary.txt", "w") as f:
    f.write(ols_model.summary().as_text())

# ---------------------------------------------------------------------
# 5. VIF — check multicollinearity among weather features
#    (we EXPECT high VIF here: temp_mean_c, temp_mean_c_sq, hdd, cdd are
#     all derived from the same underlying temperature)
# ---------------------------------------------------------------------
X_vif = train_df[FULL_FEATURES].copy()
X_vif = sm.add_constant(X_vif)
vif_data = pd.DataFrame({
    "feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print("\n=== Variance Inflation Factors ===")
print(vif_data.round(2))
vif_data.to_csv("outputs/vif_scores.csv", index=False)

# ---------------------------------------------------------------------
# 6. Q2 — Ridge / Lasso / ElasticNet with proper time-series CV
#    Features are standardized first -- required for regularized models,
#    since penalty terms are scale-sensitive.
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[FULL_FEATURES])
X_test_scaled = scaler.transform(test_df[FULL_FEATURES])
y_train = train_df[TARGET].values
y_test = test_df[TARGET].values

tscv = TimeSeriesSplit(n_splits=5)
alphas = np.logspace(-3, 3, 50)

models = {
    "Ridge": RidgeCV(alphas=alphas, cv=tscv),
    "Lasso": LassoCV(alphas=alphas, cv=tscv, max_iter=10000),
    "ElasticNet": ElasticNetCV(alphas=alphas, l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1],
                                cv=tscv, max_iter=10000),
}

reg_results = {}
fitted_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    reg_results[name] = evaluate(y_test, preds)
    fitted_models[name] = model
    alpha_attr = getattr(model, "alpha_", None)
    print(f"{name}: best alpha = {alpha_attr:.4f}" if alpha_attr else name)

print("\n=== Q2: Regularized model comparison (held-out test set) ===")
print(pd.DataFrame(reg_results).T.round(4))

# Which features did Lasso zero out? (feature selection behavior)
lasso_coefs = pd.Series(fitted_models["Lasso"].coef_, index=FULL_FEATURES)
print("\nLasso coefficients (0 = feature eliminated):")
print(lasso_coefs.round(4).sort_values(key=abs, ascending=False))

# ---------------------------------------------------------------------
# 7. Pick best model by test RMSE, save it + the scaler for the API later
# ---------------------------------------------------------------------
all_results = {**results, **reg_results}
best_name = min(all_results, key=lambda k: all_results[k]["RMSE"])
print(f"\nBest model by test RMSE: {best_name}")

joblib.dump(fitted_models.get(best_name, ols_model), f"{MODELS_DIR}/best_model.pkl")
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
joblib.dump(FULL_FEATURES, f"{MODELS_DIR}/feature_list.pkl")

pd.DataFrame(all_results).T.round(4).to_csv("outputs/model_comparison.csv")

# ---------------------------------------------------------------------
# 8. Q3 — Residual diagnostics on the full OLS model
# ---------------------------------------------------------------------
fitted_vals = ols_model.fittedvalues
residuals = ols_model.resid

# 8a. Residuals vs Fitted (look for patterns = sign of misspecification)
plt.figure(figsize=(8, 5))
plt.scatter(fitted_vals, residuals, alpha=0.5, color="#2563eb")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_residuals_vs_fitted.png", dpi=120)
plt.close()

# 8b. Q-Q plot (check normality assumption)
fig = sm.qqplot(residuals, line="45", fit=True)
plt.title("Q-Q Plot of Residuals")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_qq_plot.png", dpi=120)
plt.close()

# 8c. Breusch-Pagan test for heteroscedasticity
bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_train_sm)
print(f"\n=== Breusch-Pagan Test ===")
print(f"Statistic: {bp_stat:.4f}, p-value: {bp_pvalue:.4f}")
print("Interpretation:", "Heteroscedasticity detected (p < 0.05) -- residual variance is NOT constant"
      if bp_pvalue < 0.05 else "No strong evidence of heteroscedasticity (p >= 0.05)")

# 8d. Cook's distance (influential points / outliers)
influence = ols_model.get_influence()
cooks_d = influence.cooks_distance[0]
threshold = 4 / len(train_df)
n_influential = (cooks_d > threshold).sum()

plt.figure(figsize=(11, 4))
plt.stem(range(len(cooks_d)), cooks_d, markerfmt=",")
plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold (4/n = {threshold:.4f})")
plt.xlabel("Observation index")
plt.ylabel("Cook's Distance")
plt.title(f"Cook's Distance -- {n_influential} influential points flagged")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_cooks_distance.png", dpi=120)
plt.close()

print(f"\nInfluential points flagged by Cook's distance (> 4/n): {n_influential} out of {len(train_df)}")

print("\n=== DONE ===")
print("Saved: model comparison CSV, VIF scores CSV, OLS summary txt,")
print("       best model + scaler (.pkl), and 3 diagnostic plots.")
