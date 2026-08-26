"""
GridSense — Quick Diagnostic: Is the lockdown confounding our temperature signal?
====================================================================================
We saw near-zero (and wrong-signed) correlation between temperature/CDD and
demand in the full dataset. Hypothesis: the COVID lockdown (which suppressed
demand) overlaps with peak summer heat, masking the true weather relationship.

Test: recompute the same correlations with lockdown days excluded, and compare.

Run this with:  python3 src/check_lockdown_confound.py
"""

import pandas as pd

df = pd.read_csv("data/processed/gridsense_features.csv", parse_dates=["date"])

print("=== FULL DATA (includes lockdown) ===")
full_corr = df[["temp_mean_c", "cdd", "hdd", "temp_mean_c_sq"]].corrwith(df["demand_mu"])
print(full_corr.round(3))

non_lockdown = df[df["is_lockdown"] == 0]
print(f"\n=== EXCLUDING LOCKDOWN DAYS ({len(df) - len(non_lockdown)} removed) ===")
clean_corr = non_lockdown[["temp_mean_c", "cdd", "hdd", "temp_mean_c_sq"]].corrwith(non_lockdown["demand_mu"])
print(clean_corr.round(3))

print("\n=== Comparison ===")
comparison = pd.DataFrame({"full_data": full_corr, "excl_lockdown": clean_corr})
comparison["flipped_sign"] = (comparison["full_data"] * comparison["excl_lockdown"]) < 0
print(comparison.round(3))

print("\nIf cdd and temp_mean_c_sq show noticeably stronger (and correctly-signed,")
print("positive) correlation once lockdown days are excluded, that confirms the")
print("lockdown was masking the real temperature-demand relationship -- and it")
print("means our is_lockdown flag is doing real, necessary work in the model.")
