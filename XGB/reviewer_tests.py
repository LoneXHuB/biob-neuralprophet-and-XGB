"""
Reviewer validation tests — standalone script.

Runs the three tests added in response to reviewer comments:

  1. Stationarity tests (ADF + KPSS) focused on humidity and its causal drivers
  2. Chronological train / test split report
  3. Forecast accuracy metrics (MAE, RMSE in % moisture) + Diebold-Mariano test

Significance level: 10%  (alpha=0.10), appropriate for industrial process data
where perfect stationarity is rarely achieved but series are mean-reverting
around the process operating point.

Run from the XGB/ folder:
    python reviewer_tests.py [--file PATH]

Outputs (saved next to this script):
    stationarity_tests.csv
    dm_test_forecast_comparison.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller, kpss
from neuralprophet import NeuralProphet

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_FILE = "../data/dec_4/9610PBUE.csv"
TARGET       = "p1_four_humidite_produit"
TEST_SECS    = 3600        # last 1 hour held out as test set
ALPHA        = 0.10        # significance level (10% — practical industrial threshold)

# Causal regressors selected by PCMCI (CF_ON model)
CAUSAL_REGRESSORS = [
    "p1_mouleuse_pct_sortie",
    "p1_four_tempz1",
    "p1_four_tempz2",
    "p1_four_tempz3",
    "p1_four_tempz4",
    "p1_melangeur_qtetremie2",
    "p1_melangeur_humidite_ambiante",
    "p1_melangeur_temperature_ambiante",
]

# All regressors used by the control model (CF_OFF)
ALL_REGRESSORS = [
    "p1_four_temperature_interne",
    "p1_mouleuse_pct_sortie",
    "p1_four_tempz3",
    "p1_four_tempz4",
    "p1_four_tempz2",
    "p1_melangeur_qtetremie2",
    "p1_mouleuse_amperage",
    "p1_prodmarche",
    "p1_four_tempz1",
    "p1_melangeur_humidite_ambiante",
    "p1_lot",
    "p1_melangeur_qteeau",
    "p1_mouleuse_vitesse",
    "p1_four_tempz1_sp",
    "p1_four_tempz2_sp",
    "p1_four_tempz4_sp",
    "p1_melangeur_temperature_ambiante",
    "p1_four_tempz3_sp",
]


# =============================================================================
# Helpers
# =============================================================================

def read_csv_auto(file_path: str) -> pd.DataFrame:
    """Read CSV, auto-detecting separator (comma or semicolon)."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    return pd.read_csv(file_path, sep=sep)


def available_regressors(df_cols: list, wanted: list) -> list:
    """Return only the regressors from *wanted* that exist in *df_cols*."""
    present = set(df_cols)
    return [r for r in wanted if r in present]


def load_and_resample(file_path: str, regressors: list):
    """
    Load CSV, auto-detect separator, keep only regressors present in the file,
    rename columns, 1-second resample.

    Also drops regressors that are constant in the training data: NeuralProphet
    silently removes them during fit() but they remain in the dataframe, causing
    an 'Unexpected column' ValueError in predict().

    Returns (df, df_train, df_test, used_regressors).
    """
    raw = read_csv_auto(file_path)
    raw["t_stamp"] = pd.to_datetime(raw["t_stamp"])
    raw = raw.sort_values("t_stamp").reset_index(drop=True)
    raw.rename(columns={"t_stamp": "ds", TARGET: "y"}, inplace=True)

    # Keep only regressors that exist in this file
    used = available_regressors(raw.columns.tolist(), regressors)
    dropped = [r for r in regressors if r not in used]
    if dropped:
        print(f"  [INFO] Regressors not found in file, skipped: {dropped}")

    cols = ["ds", "y"] + used
    df = raw[cols].dropna().reset_index(drop=True)
    df.set_index("ds", inplace=True)
    df = df.resample("1S").interpolate(method="linear").reset_index()

    df_train = df.iloc[:-TEST_SECS]
    df_test  = df.iloc[-TEST_SECS:]

    # Drop regressors that are constant in the training split.
    # NeuralProphet silently removes constant columns during fit(), but if they
    # remain in the dataframe passed to predict() it raises 'Unexpected column'.
    varied   = [r for r in used if df_train[r].nunique() >= 2]
    constant = [r for r in used if r not in varied]
    if constant:
        print(f"  [INFO] Constant regressors in training data, excluded: {constant}")

    # Rebuild df with only the columns that will actually be used
    cols = ["ds", "y"] + varied
    df       = df[cols]
    df_train = df.iloc[:-TEST_SECS]
    df_test  = df.iloc[-TEST_SECS:]

    return df, df_train, df_test, varied


def train_neuralprophet(df_train: pd.DataFrame, regressors: list) -> NeuralProphet:
    """Build and fit a NeuralProphet model on df_train."""
    model = NeuralProphet(
        n_lags=20,
        n_forecasts=1,
        ar_layers=[20, 20],
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    for reg in regressors:
        model = model.add_future_regressor(reg)
    model.fit(df_train, freq="S", progress="print")
    return model


def make_forecast(model: NeuralProphet, df: pd.DataFrame,
                  regressors: list) -> pd.DataFrame:
    """
    Produce 1-step-ahead predictions at every timestamp in df.

    The model was trained on df_train (all but the last TEST_SECS rows).
    Here we pass the full df (train + test) with periods=0 so NeuralProphet
    treats every row — including the test period — as a historic prediction
    target. No future regressor values are needed beyond the dataset.
    Evaluation is then done on the last TEST_SECS rows of the result.
    """
    df_reg = df[["ds", "y"] + regressors].copy()
    future = model.make_future_dataframe(
        df_reg,
        periods=0,
        n_historic_predictions=True,
    )
    forecast = model.predict(future)
    return pd.merge(df[["ds", "y"]], forecast[["ds", "yhat1"]], on="ds", how="left")


def diebold_mariano(y, f1, f2, h=1):
    """
    Two-sided Diebold-Mariano test (Harvey, Leybourne & Newbold 1997).
    Loss: squared error.  Positive DM → f1 is worse than f2.
    H0: equal predictive accuracy.
    """
    e1 = (y - f1) ** 2
    e2 = (y - f2) ** 2
    d  = e1 - e2
    T  = len(d)
    d_bar  = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    lrv    = gamma0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
        lrv    += 2 * (1 - k / h) * gamma_k
    dm_stat = d_bar / np.sqrt(lrv / T)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def stationarity_verdict(adf_p: float, kpss_p: float, alpha: float = ALPHA) -> str:
    """
    Practical stationarity verdict for industrial process data.
    Both tests must agree; borderline cases are labelled explicitly.
    """
    adf_ok  = adf_p  < alpha          # ADF rejects unit root → stationary signal
    kpss_ok = kpss_p > alpha          # KPSS fails to reject stationarity
    if adf_ok and kpss_ok:
        return "Stationary"
    if not adf_ok and not kpss_ok:
        return "Non-stationary"
    # Tests disagree → borderline (common in slowly-drifting process data)
    return "Borderline"


# =============================================================================
# Test 1 — Stationarity focused on humidity and its drivers
# =============================================================================

def test_stationarity(file_path: str, label: str = ""):
    print("\n" + "=" * 70)
    print(f"TEST 1 — Stationarity of humidity and causal drivers  [{label}]")
    print(f"         Significance level: {int(ALPHA*100)}%  (industrial process threshold)")
    print("=" * 70)

    raw = read_csv_auto(file_path)
    raw["t_stamp"] = pd.to_datetime(raw["t_stamp"])
    raw = raw.sort_values("t_stamp").reset_index(drop=True)

    vars_to_test = [TARGET] + available_regressors(raw.columns.tolist(), CAUSAL_REGRESSORS)

    rows = []
    for col in vars_to_test:
        series = raw[col].dropna()
        _, adf_p, *_ = adfuller(series, autolag="AIC")
        try:
            _, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
        except Exception:
            kpss_p = float("nan")

        verdict = stationarity_verdict(adf_p, kpss_p)
        is_target = "← TARGET" if col == TARGET else ""

        rows.append({
            "file":          label,
            "variable":      col,
            "ADF p":         round(adf_p,    4),
            "KPSS p":        round(kpss_p,   4),
            "verdict":       verdict,
            "note":          is_target,
        })
        print(f"  {col:<40} ADF p={adf_p:.3f}  KPSS p={kpss_p:.3f}  → {verdict}  {is_target}")

    df_out = pd.DataFrame(rows)
    csv_name = f"stationarity_{label}.csv"
    df_out.to_csv(csv_name, index=False)

    n_stat   = sum(1 for r in rows if r["verdict"] == "Stationary")
    n_border = sum(1 for r in rows if r["verdict"] == "Borderline")
    n_non    = sum(1 for r in rows if r["verdict"] == "Non-stationary")
    target_verdict = next(r["verdict"] for r in rows if r["variable"] == TARGET)

    print(f"\n  Summary: {n_stat} stationary, {n_border} borderline, {n_non} non-stationary")
    print(f"  Humidity target ({TARGET}): {target_verdict}")
    if target_verdict != "Stationary":
        print("  → Series shows slow drift around the process setpoint. This is")
        print("    typical for industrial continuous baking — the oven reaches a")
        print("    quasi-steady state within each production run. PCMCI results")
        print("    should be interpreted as within-run causal structure.")
    print(f"Saved → {csv_name}")
    return df_out


# =============================================================================
# Test 2 — Train / test split report
# =============================================================================

def report_split(df, df_train, df_test, file_path):
    print("\n" + "=" * 70)
    print("TEST 2 — Chronological Train / Test Split")
    print("=" * 70)
    print(f"  File            : {file_path}")
    print(f"  Total samples   : {len(df):,}  (after 1-s interpolation)")
    print(f"  Training set    : {len(df_train):,} samples")
    print(f"    from  {df_train['ds'].min()}")
    print(f"    to    {df_train['ds'].max()}")
    print(f"  Test set        : {len(df_test):,} samples  ({TEST_SECS} s = 1 hour)")
    print(f"    from  {df_test['ds'].min()}")
    print(f"    to    {df_test['ds'].max()}")
    print(f"  Test fraction   : {100 * len(df_test) / len(df):.1f}%")
    print(f"  NeuralProphet   : n_lags=20 s,  n_forecasts=1 s  (1-step-ahead)")
    print(f"  Lag leakage     : None — test window is strictly after all training data")


# =============================================================================
# Test 3 — Forecast metrics + Diebold-Mariano
# =============================================================================

def test_forecast_accuracy(df_forecast_on, df_forecast_off, label: str = ""):
    print("\n" + "=" * 70)
    print(f"TEST 3 — Forecast Accuracy & Diebold-Mariano Test  [{label}]")
    print(f"         Significance level: {int(ALPHA*100)}%")
    print("=" * 70)

    on_eval  = df_forecast_on.iloc[-TEST_SECS:].dropna(subset=["y", "yhat1"]).reset_index(drop=True)
    off_eval = df_forecast_off.iloc[-TEST_SECS:].dropna(subset=["y", "yhat1"]).reset_index(drop=True)

    merged = on_eval[["ds", "y", "yhat1"]].merge(
        off_eval[["ds", "yhat1"]].rename(columns={"yhat1": "yhat1_off"}),
        on="ds", how="inner"
    )

    y_true   = merged["y"].values
    yhat_on  = merged["yhat1"].values
    yhat_off = merged["yhat1_off"].values

    mae_on   = mean_absolute_error(y_true, yhat_on)
    rmse_on  = np.sqrt(mean_squared_error(y_true, yhat_on))
    mae_off  = mean_absolute_error(y_true, yhat_off)
    rmse_off = np.sqrt(mean_squared_error(y_true, yhat_off))

    print("\nUnits: percentage points of moisture content (% w.b.)")
    print(f"{'Model':<24} {'MAE (%)':>10} {'RMSE (%)':>10}")
    print("-" * 46)
    print(f"{'CF_ON  (causal)':<24} {mae_on:>10.4f} {rmse_on:>10.4f}")
    print(f"{'CF_OFF (all vars)':<24} {mae_off:>10.4f} {rmse_off:>10.4f}")
    print("-" * 46)
    mae_ratio  = mae_off  / mae_on  if mae_on  > 0 else float("nan")
    rmse_ratio = rmse_off / rmse_on if rmse_on > 0 else float("nan")
    print(f"  MAE improvement  (CF_OFF/CF_ON) : {mae_ratio:.2f}x")
    print(f"  RMSE improvement (CF_OFF/CF_ON) : {rmse_ratio:.2f}x")

    dm_stat, dm_p = diebold_mariano(y_true, yhat_off, yhat_on, h=1)
    reject = dm_p < ALPHA

    print(f"\n  Diebold-Mariano test  (H0: CF_ON == CF_OFF accuracy)")
    print(f"  DM statistic   : {dm_stat:+.4f}")
    print(f"  p-value        : {dm_p:.4f}")
    if reject:
        winner = "CF_ON" if dm_stat > 0 else "CF_OFF"
        print(f"  Result: Reject H0 at {int(ALPHA*100)}% — {winner} is significantly more accurate.")
    else:
        print(f"  Result: Fail to reject H0 at {int(ALPHA*100)}% — difference not significant.")

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    ax = axes[0]
    ax.plot(merged["ds"], y_true,   label="Observed humidity (% w.b.)",
            color="black", linewidth=1.5)
    ax.plot(merged["ds"], yhat_on,  linestyle="--", color="royalblue",
            label=f"CF_ON   MAE={mae_on:.4f}%  RMSE={rmse_on:.4f}%")
    ax.plot(merged["ds"], yhat_off, linestyle="-.", color="darkorange",
            label=f"CF_OFF  MAE={mae_off:.4f}%  RMSE={rmse_off:.4f}%")
    ax.set_ylabel("Moisture content (% w.b.)")
    ax.set_title(f"[{label}] Held-out test set (1 hour) — CF_ON vs CF_OFF")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax2 = axes[1]
    ax2.plot(merged["ds"], np.abs(y_true - yhat_on),
             label="CF_ON  |error|",  color="royalblue",  alpha=0.8)
    ax2.plot(merged["ds"], np.abs(y_true - yhat_off),
             label="CF_OFF |error|", color="darkorange", alpha=0.8)
    ax2.set_ylabel("|Error| (% w.b.)")
    ax2.set_xlabel("Timestamp")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)
    fig_name = f"dm_test_{label}.png"
    fig.tight_layout()
    fig.savefig(fig_name, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {fig_name}")

    return {
        "file":        label,
        "n_test":      len(merged),
        "mae_on":      round(mae_on,   4),
        "rmse_on":     round(rmse_on,  4),
        "mae_off":     round(mae_off,  4),
        "rmse_off":    round(rmse_off, 4),
        "mae_ratio":   round(mae_ratio,  2),
        "rmse_ratio":  round(rmse_ratio, 2),
        "DM_stat":     round(dm_stat, 4),
        "DM_p":        round(dm_p,    4),
        "CF_ON_wins":  "Yes" if (reject and dm_stat > 0) else "No",
    }


# =============================================================================
# Main
# =============================================================================

ALL_FILES = [
    "../data/dec_4/9610UE.csv",
    "../data/dec_4/9610PBUE.csv",
    "../data/dec_4/9640GRUE.csv",
    "../data/dec_4/9705UE.csv",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None,
                        help="Single CSV file to evaluate (default: run all four)")
    args = parser.parse_args()
    files = [args.file] if args.file else ALL_FILES

    summary_rows = []

    for file_path in files:
        label = file_path.split("/")[-1].replace(".csv", "")
        print(f"\n{'#' * 70}")
        print(f"  FILE: {label}")
        print(f"{'#' * 70}")

        # Test 1 — stationarity
        test_stationarity(file_path, label=label)

        # Load data + train both models
        print(f"\nTraining CF_ON model ({label})...")
        df_on, df_train_on, df_test_on, used_causal = load_and_resample(file_path, CAUSAL_REGRESSORS)
        model_on = train_neuralprophet(df_train_on, used_causal)
        df_forecast_on = make_forecast(model_on, df_on, used_causal)

        print(f"\nTraining CF_OFF model ({label})...")
        df_off, df_train_off, df_test_off, used_all = load_and_resample(file_path, ALL_REGRESSORS)
        model_off = train_neuralprophet(df_train_off, used_all)
        df_forecast_off = make_forecast(model_off, df_off, used_all)

        # Test 2 — split report (now after both models trained, before accuracy)
        report_split(df_on, df_train_on, df_test_on, file_path)

        # Test 3 — metrics + Diebold-Mariano
        row = test_forecast_accuracy(df_forecast_on, df_forecast_off, label=label)
        summary_rows.append(row)

    # ── Cross-file summary table ──────────────────────────────────────────────
    if len(summary_rows) > 1:
        print(f"\n{'=' * 80}")
        print("SUMMARY ACROSS ALL FILES")
        print(f"{'=' * 80}")
        hdr = f"{'File':<14} {'MAE ON':>8} {'MAE OFF':>9} {'MAE ratio':>10} {'RMSE ON':>8} {'RMSE OFF':>9} {'DM p':>7} {'CF_ON wins':>11}"
        print(hdr)
        print("-" * 80)
        for r in summary_rows:
            print(f"{r['file']:<14} {r['mae_on']:>8.4f} {r['mae_off']:>9.4f} "
                  f"{r['mae_ratio']:>9.2f}x {r['rmse_on']:>8.4f} {r['rmse_off']:>9.4f} "
                  f"{r['DM_p']:>7.4f} {r['CF_ON_wins']:>11}")
        print("-" * 80)
        wins = sum(1 for r in summary_rows if r["CF_ON_wins"] == "Yes")
        avg_mae_ratio  = np.mean([r["mae_ratio"]  for r in summary_rows])
        avg_rmse_ratio = np.mean([r["rmse_ratio"] for r in summary_rows])
        print(f"  CF_ON wins: {wins}/{len(summary_rows)} files")
        print(f"  Mean MAE improvement  : {avg_mae_ratio:.2f}x")
        print(f"  Mean RMSE improvement : {avg_rmse_ratio:.2f}x")

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv("summary_all_files.csv", index=False)
        print("Saved → summary_all_files.csv")
