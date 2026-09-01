# -*- coding: utf-8 -*-
"""
Measure wall-clock runtime of the causal-discovery steps, to answer the
reviewer's question on computational cost. PCMCI (tigramite) + Granger
(statsmodels) only; TCDF needs torch which is not installed here.

Run from XGB/:  python time_causal_pipeline.py
"""
import time

import pandas as pd

# Inlined from reviewer_tests.py (importing it pulls in neuralprophet, not needed here)
TARGET = "p1_four_humidite_produit"
ALL_REGRESSORS = [
    "p1_four_temperature_interne", "p1_mouleuse_pct_sortie", "p1_four_tempz3",
    "p1_four_tempz4", "p1_four_tempz2", "p1_melangeur_qtetremie2",
    "p1_mouleuse_amperage", "p1_prodmarche", "p1_four_tempz1",
    "p1_melangeur_humidite_ambiante", "p1_lot", "p1_melangeur_qteeau",
    "p1_mouleuse_vitesse", "p1_four_tempz1_sp", "p1_four_tempz2_sp",
    "p1_four_tempz4_sp", "p1_melangeur_temperature_ambiante", "p1_four_tempz3_sp",
]

FILE = "../data/dec_4/9610UE.csv"
TAU_MAX = 20          # matches the n_lags = 20 s used by the forecasting model
PC_ALPHA = 0.05


def load_matrix():
    with open(FILE, "r", encoding="utf-8", errors="replace") as fh:
        sep = ";" if fh.readline().count(";") > 1 else ","
    raw = pd.read_csv(FILE, sep=sep)
    cols = [TARGET] + [r for r in ALL_REGRESSORS if r in raw.columns]
    df = raw[["t_stamp"] + cols].copy()
    df["t_stamp"] = pd.to_datetime(df["t_stamp"], errors="coerce")
    df = df.dropna(subset=["t_stamp"]).set_index("t_stamp").sort_index()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.resample("1S").interpolate(method="linear").dropna()
    # PCMCI needs non-constant columns
    cols = [c for c in cols if df[c].std() > 0]
    return df, cols


if __name__ == "__main__":
    df, cols = load_matrix()
    N, T = len(cols), len(df)
    print(f"file      : {FILE}")
    print(f"variables : N = {N}")
    print(f"samples   : T = {T} (1 Hz)")
    print(f"tau_max   : {TAU_MAX}")

    # ── PCMCI ────────────────────────────────────────────────────────────────
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    dataframe = pp.DataFrame(df[cols].to_numpy(), var_names=cols)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)

    t0 = time.perf_counter()
    pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=PC_ALPHA)
    pcmci_s = time.perf_counter() - t0
    print(f"\nPCMCI     : {pcmci_s:8.1f} s  ({pcmci_s/60:.1f} min)")

    # ── Granger (pairwise vs target) ─────────────────────────────────────────
    from statsmodels.tsa.stattools import grangercausalitytests

    t0 = time.perf_counter()
    for c in cols[1:]:
        grangercausalitytests(df[[TARGET, c]].to_numpy(), maxlag=TAU_MAX, verbose=False)
    granger_s = time.perf_counter() - t0
    print(f"Granger   : {granger_s:8.1f} s  ({granger_s/60:.1f} min)  for {N-1} pairs")

    print(f"\nTOTAL     : {pcmci_s + granger_s:8.1f} s")
    assert pcmci_s > 0 and granger_s > 0
