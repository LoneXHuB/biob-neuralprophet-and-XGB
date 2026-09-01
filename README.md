# Causal Forecasting for Industrial Process Data

Find out **which sensors actually drive** a variable you care about, then use
only those sensors to **forecast** it.

The example throughout is an industrial baking line: the target is product
moisture out of the oven (`p1_four_humidite_produit`), the candidate causes are
oven zone temperatures, mixer settings, moulder speed, and so on. Nothing here
is specific to baking — any set of time series works.

The core claim the code tests: a forecast built on the **causal** subset of
sensors beats one built on **all** sensors.

> **No data is included in this repo.** The plant data is confidential. Bring
> your own CSVs — the format is described below, and
> [example_data_template.csv](example_data_template.csv) shows the exact shape.

---

## 1. Install

```bash
conda env create -f environment.yml
conda activate causal-t3
```

Two optional extras, only if you need them:

```bash
# F-PCMCI (feature selection + PCMCI), used by F-PCMCI.py
pip install fpcmci

# TCDF (deep-learning causal discovery), used by algo-comparison.py
git clone https://github.com/M-Nauta/TCDF.git
```

Other upstream tools this project compares against — clone only if you want them:

```bash
git clone https://github.com/jakobrunge/tigramite.git   # PCMCI reference implementation
git clone https://github.com/pwollstadt/IDTxl.git       # information-theoretic measures
```

---

## 2. What your data has to look like

**One CSV per production run** (or per machine, or per day). One row per
timestamp, one column per sensor.

| Requirement | Detail |
|---|---|
| Timestamp column | Must be named **`t_stamp`** and be readable by `pandas.to_datetime`, e.g. `2025-01-01 08:00:00` |
| Sorting | Chronological order, oldest row first |
| Sampling rate | Roughly **one sample per second**. The scripts resample to exactly 1 s and fill small gaps by linear interpolation |
| Other columns | Plain numbers. Text columns are dropped automatically (or label-encoded in `algo-comparison.py`) |
| Separator | Comma or semicolon — the `XGB/` scripts detect it, the root scripts assume comma |
| Length | At least a few thousand rows. `XGB/reviewer_tests.py` holds out the **last 3600 rows (1 hour)** as the test set, so you need meaningfully more than that |
| Constant columns | A sensor that never changes carries no information and is dropped automatically |
| Missing values | A few gaps are fine (interpolated). Long dead stretches should be cut out first |

Minimal example — this is the whole format:

```csv
t_stamp,p1_four_humidite_produit,p1_four_tempz1,p1_mouleuse_pct_sortie
2025-01-01 08:00:00,20.10,420.0,5
2025-01-01 08:00:01,20.12,420.3,5
2025-01-01 08:00:02,20.09,420.1,6
```

Column **names are yours to choose** — they only have to match the names you
put in the config at the top of each script (section 4).

Put your files anywhere; `data/` is the convention here and is git-ignored, so
nothing confidential can be committed by accident.

---

## 3. The three things you can run

### a) Which sensors cause my target? — `pcmci.py`

PCMCI is a statistical test that asks, for every pair of sensors and every time
lag: *does A at time t−k tell me anything about B at time t that no other
sensor already told me?* That last part is what separates it from plain
correlation — it removes indirect links.

```bash
python pcmci.py
```

Outputs into `data/results/`:

- `*_influence_matrix_latest.csv` — strength of each cause → effect link
- `*_p_value_matrix_latest.csv` — p-value per link (below 0.05 = trustworthy)
- `*_causal_matrix_latest.png` — the same thing as a heatmap

Read the heatmap column by column: the column is the **cause**, the row is the
**effect**. A strong colour with a small p-value on it is a real driver.

Variants of the same idea:

- [F-PCMCI.py](F-PCMCI.py) — pre-filters the sensors with transfer entropy
  before running PCMCI, and also draws the causal graph (DAG). Faster on wide
  datasets.
- [algo-comparison.py](algo-comparison.py) — runs **PCMCI, Granger causality
  and TCDF** on the same data and ranks the sensors by average rank across the
  three. Use it when you want a consensus rather than one method's opinion.

### b) Does the causal subset forecast better? — `XGB/reviewer_tests.py`

This is the main experiment, and it is where **NeuralProphet** comes in.

NeuralProphet is a forecasting model: you give it a time series plus other
sensors ("regressors") and it learns to predict the next value. It is a neural
network underneath, but you drive it like a regression — no deep learning
knowledge needed.

The script trains **two** NeuralProphet models on the same data:

| Model | Regressors it gets |
|---|---|
| **CF_ON** | only the sensors PCMCI flagged as causes (`CAUSAL_REGRESSORS`) |
| **CF_OFF** | every available sensor (`ALL_REGRESSORS`) |

Both predict one second ahead from the previous 20 seconds (`n_lags=20`,
`n_forecasts=1`). Training uses everything except the last hour; that last hour
is held out and never seen during training, so there is no leakage.

```bash
cd XGB
python reviewer_tests.py                              # all files listed in ALL_FILES
python reviewer_tests.py --file ../data/my_run.csv    # or just one
```

It prints and saves three things:

1. **Stationarity tests** (ADF + KPSS) → `stationarity_<file>.csv`. A sanity
   check that your series is not simply drifting; drifting series make causal
   results specific to that one run.
2. **Train/test split report** — exact row counts and timestamps, so the split
   is auditable.
3. **Accuracy + Diebold-Mariano test** → `dm_test_<file>.png` and
   `summary_all_files.csv`. MAE and RMSE for both models, plus the DM test,
   which answers "is CF_ON's advantage real or just luck?" A p-value under 0.10
   means the difference is statistically significant.

### c) Just look at the data — `vizualize_data.py`

An interactive browser dashboard: pick any sensors from a dropdown, see them
plotted together.

```bash
python vizualize_data.py     # then open http://127.0.0.1:8050
```

Always the first thing to run on a new file. Dead sensors, stuck values and
unit problems are obvious on a plot and invisible in a p-value.

---

## 4. Running it on your own data

Every script keeps its configuration in plain variables at the top (or bottom).
Edit those, nothing else.

| File | What to change |
|---|---|
| [vizualize_data.py](vizualize_data.py) | `csv_file` — path to your CSV |
| [pcmci.py](pcmci.py) | `data_file` at the bottom; `tau_max` (how many seconds back to look) and `pc_alpha` (significance, 0.05) inside |
| [F-PCMCI.py](F-PCMCI.py) | `data_file` at the bottom; `min_lag` / `max_lag` / `alpha` inside |
| [algo-comparison.py](algo-comparison.py) | `data_file`, `target` (the column you want to explain), `maxlag` |
| [XGB/reviewer_tests.py](XGB/reviewer_tests.py) | `TARGET`, `CAUSAL_REGRESSORS`, `ALL_REGRESSORS`, `ALL_FILES`, `TEST_SECS` |
| [XGB/time_causal_pipeline.py](XGB/time_causal_pipeline.py) | `FILE`, `TARGET`, `ALL_REGRESSORS`, `TAU_MAX` — measures how long causal discovery takes |

Use **forward slashes** in paths (`data/my_run.csv`). They work on Windows too
and avoid backslash-escape surprises.

The normal order of work:

1. Plot the file (`vizualize_data.py`) and throw out obviously broken sensors.
2. Run `pcmci.py` (or `algo-comparison.py` for a consensus) to get the list of
   real drivers of your target.
3. Paste that list into `CAUSAL_REGRESSORS` in `XGB/reviewer_tests.py`, and put
   everything else into `ALL_REGRESSORS`.
4. Run `reviewer_tests.py` and read the DM p-value.

---

## 5. Handling confidential files

[encrypt-data.py](encrypt-data.py) encrypts and decrypts CSVs with a Fernet key,
so raw plant data can be stored or shared safely:

```python
generate_key()                                  # run ONCE, creates key.key
encrypt_csv('data/run.csv', 'data/run.bin')     # share the .bin
decrypt_to_csv('data/run.bin', 'data/run.csv')  # on the other side
```

**The `.key` file is the password. Never commit it, and never send it alongside
the `.bin`.** `.gitignore` already blocks `*.key`, `*.bin`, `*.csv` and the
whole `data/` folder.

---

## 6. Deliberately not in this repo

`.gitignore` keeps these on your disk but out of the repo:

- **`data/`, `*.csv`, `*.bin`, `*.key`** — confidential plant data and the
  encryption keys.
- **`*.ipynb`** — the notebooks store their outputs inside the file, including
  plots and printed rows of real data. Strip them first if you want to add one:
  `jupyter nbconvert --clear-output --inplace notebook.ipynb`
- **`apply_revisions.py`, `build_response_letter*.py`, `fix_content*.py`,
  `fix_revisions.py`, `make_tracked_v3.py`** — tooling for the journal
  submission. They embed the manuscript ID and verbatim reviewer comments for a
  paper still under double-blind review. Un-ignore them once it is published.
- **`cmiknn.py`** — a verbatim copy of tigramite's `CMIknn` (GPL-3.0, Jakob
  Runge). The `tigramite` package already provides it, and shipping the copy
  would make this whole repo GPL.
- **`IDTxl/`, `TCDF/`, `tigramite/`, `CausalT/`** — upstream projects. Clone
  them yourself, see section 1.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ValueError: Unexpected column` from NeuralProphet | A regressor was constant in the training split. `load_and_resample` already drops those — go through it rather than passing a raw dataframe |
| `Target variable '...' not found` | Your `TARGET` name does not match a column, or the column was dropped as non-numeric |
| `FutureWarning` about `'1S'` | Newer pandas wants lowercase `'1s'`. Harmless today; change the `resample("1S")` calls if it ever becomes an error |
| PCMCI takes forever | Cost grows fast with `tau_max` and the number of columns. Start with `tau_max=5` and few sensors |
| TCDF step skipped in `algo-comparison.py` | `TCDF/` is not cloned, or PyTorch is not installed. The other two methods still run |
