import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
import warnings
import sys

# Add the TCDF folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "TCDF"))
import runTCDF

warnings.filterwarnings("ignore")

# === CONFIG ===
data_file = "data/dec_4/9610PBUE.csv"  # update to your path
target = "p1_four_humidite_produit"
maxlag = 3

# === LOAD & PREPROCESS ===
data = pd.read_csv(data_file)

# Drop timestamp or other non-numeric columns automatically
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
if 't_stamp' in data.columns:
    non_numeric_cols.append('t_stamp')
data = data.drop(columns=non_numeric_cols)

# Encode categorical columns (few unique values)
for col in data.columns:
    if data[col].dtype == object or data[col].nunique() < 10:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Keep only numeric columns
data = data.select_dtypes(include=[np.number])

if target not in data.columns:
    raise ValueError(f"Target variable '{target}' not found in numeric columns after cleaning: {list(data.columns)}")

var_names = list(data.columns)

# Scale
data_scaled = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)

# === 1. PCMCI (ParCorr) ===
print("▶ Running PCMCI (ParCorr)...")
df_tigra = pp.DataFrame(data_scaled.values, var_names=var_names)
indep_test = ParCorr(significance='analytic')
pcmci = PCMCI(dataframe=df_tigra, cond_ind_test=indep_test)
results_pcmci = pcmci.run_pcmci(tau_min=0, tau_max=maxlag, pc_alpha=0.05)

target_idx = var_names.index(target)
pcmci_scores = {}
for src in range(len(var_names)):
    if src == target_idx:
        continue
    vals = [abs(results_pcmci['val_matrix'][src, target_idx, lag]) for lag in range(maxlag)]
    pcmci_scores[var_names[src]] = np.sum(vals)

# === 2. Granger causality ===
print("▶ Running Granger causality...")
granger_scores = {}
for var in var_names:
    if var == target:
        continue
    try:
        res = grangercausalitytests(data_scaled[[target, var]], maxlag=maxlag, verbose=False)
        pvals = [res[i+1][0]['ssr_ftest'][1] for i in range(maxlag)]
        granger_scores[var] = -np.log(min(pvals) + 1e-6)
    except Exception:
        granger_scores[var] = 0

# === 3. TCDF (via runTCDF) ===
tcdf_scores = {}
try:
    print("▶ Running TCDF via runTCDF...")

    # TCDF hyperparameters
    cuda = False
    nrepochs = 500
    kernel_size = 3
    levels = 2
    loginterval = 50
    learningrate = 0.01
    optimizername = "Adam"
    dilation_c = 2
    seed = 1111
    significance = 0.8
    targets = ['p1_four_humidite_produit']
    # Save numeric-only data to a temp CSV for TCDF
    temp_file = "temp_numeric_data.csv"
    data.to_csv(temp_file, index=False)

    # Run TCDF properly with hyperparameters
    allcauses, alldelays, allreallosses, allscores, columns = runTCDF.runTCDF(
        temp_file,
        cuda=cuda,
        nrepochs=nrepochs,
        kernel_size=kernel_size,
        levels=levels,
        loginterval=loginterval,
        learningrate=learningrate,
        optimizername=optimizername,
        dilation_c=dilation_c,
        seed=seed,
        significance=significance,
        targets = targets
    )

    target_idx = columns.index(target)

    # Map attention scores for all variables toward the target
    for i, var in enumerate(columns):
        if var == target:
            continue
        # if attention exists, use it, else 0
        if target_idx in allscores and i < len(allscores[target_idx]):
            tcdf_scores[var] = allscores[target_idx][i]
        else:
            tcdf_scores[var] = 0

    os.remove(temp_file)

except Exception as e:
    tcdf_scores = {}
    print("⚠️ TCDF not run:", e)


# === COMBINE & RANK ===
methods = ["PCMCI", "Granger"] + (["TCDF"] if tcdf_scores else [])
all_vars = [v for v in var_names if v != target]

comparison = pd.DataFrame({
    "Variable": all_vars,
    "PCMCI": [pcmci_scores.get(v, 0) for v in all_vars],
    "Granger": [granger_scores.get(v, 0) for v in all_vars],
    **({"TCDF": [tcdf_scores.get(v, 0) for v in all_vars]} if tcdf_scores else {})
})

# Convert to ranks (1 = strongest causal)
for m in methods:
    comparison[m + "_Rank"] = comparison[m].rank(ascending=False, method="min")

# Compute mean rank
comparison["MeanRank"] = comparison[[m+"_Rank" for m in methods]].mean(axis=1)

# === SORT BY MEAN RANK (most causal first) ===
comparison = comparison.sort_values("MeanRank").reset_index(drop=True)

# Display and save CSV
print("\n=== Consensus causal ranking for", target, "===")
print(comparison[["Variable"] + [m+"_Rank" for m in methods] + ["MeanRank"]])

out_csv = f"causal_comparison_{target}.csv"
comparison.to_csv(out_csv, index=False)
print(f"\n✅ Results saved to {out_csv}")

# === Add TCDF raw attention scores column ===
if tcdf_scores:
    comparison["TCDF_Score"] = [tcdf_scores.get(v, 0) for v in comparison["Variable"]]

# === VISUALIZATION ===
plt.figure(figsize=(14, 6))
bar_width = 0.25
x = np.arange(len(comparison))

for i, m in enumerate(methods):
    # Use ranks for PCMCI and Granger
    plt.bar(x + i * bar_width, comparison[m + "_Rank"], width=bar_width, label=m + " (rank)")

# Center x-ticks under bars
plt.xticks(x + bar_width*(len(methods)-1)/2, comparison["Variable"], rotation=45, ha="right")
plt.ylabel("Rank")
plt.title(f"Causal ranking comparison for {target}")
plt.legend()
plt.tight_layout()

barplot_file = f"causal_comparison_{target}_barplot.png"
plt.savefig(barplot_file, dpi=300)
plt.show()
print(f"📊 Bar plot saved to {barplot_file}")

