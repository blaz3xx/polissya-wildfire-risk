# Results

Complete numbers from the reference run, so anything quoted in the README can be checked against its source.

## Run environment

| | |
|---|---|
| Platform | Google Colab, CPU runtime (no GPU allocated) |
| TensorFlow | 2.20.0 |
| NumPy / pandas | 2.0.2 / 2.2.2 |
| Random state | 42, fixed for `random`, `numpy`, `PYTHONHASHSEED`, TensorFlow |
| Live inference date | 18 May 2026 |

## Dataset

| Quantity | Value |
|---|---|
| Period | 2016-01-01 → 2025-12-31 |
| Daily records | 3,653 |
| Columns after merge | 50 |
| Forest patches (≥ 2 km²) | 4,145 |
| Forest mask area | 58,276 km² |
| Weather grid nodes | 19 |
| API requests (10 batches × 10 years) | 100 |

### FIRMS filtering cascade

| Stage | Records | Share of previous |
|---|---|---|
| Raw detections in bounding box | 140,073 | — |
| Confidence filter | 119,132 | 85.0% |
| Daytime only | 78,734 | 66.1% |
| Brightness ≥ 300 K | 78,734 | 100.0% |
| Clipped to forest mask | 17,669 | 22.4% |

| Label statistics | Value |
|---|---|
| Days with any forest fire detection | 1,165 (31.89%) |
| Days with significant fire activity (target) | 468 (12.81%) |
| Maximum daily FRP | 2,174.3 MW |
| Detection date range | 2016-02-07 → 2025-12-29 |

## Feature engineering

| Quantity | Value |
|---|---|
| Predictors after engineering | 170 (excluding `date`) |
| Newly created features | 121 |
| Rolling-window features | 108 |
| Calendar features | 5 |
| Predictors selected for modelling | 161 |
| VPD mean / max | 0.381 / 1.80 kPa |
| FWI mean / max | 2.55 / 30.3 |
| Longest dry spell | 21 days |

### Stationarity (Augmented Dickey-Fuller)

| Series | ADF statistic | p-value |
|---|---|---|
| Maximum temperature | −3.713 | 3.93 × 10⁻³ |
| Minimum relative humidity | −5.963 | 2.02 × 10⁻⁷ |

Both reject the unit root at the 1% level.

## Splits and tensors

| Split | Days | Range | Tensor shape | Positive rate |
|---|---|---|---|---|
| Train | 2,557 | 2016-01-01 → 2022-12-31 | (2543, 14, 161) | 11.64% |
| Validation | 548 | 2023-01-01 → 2024-07-01 | (534, 14, 161) | 12.92% |
| Test | 548 | 2024-07-02 → 2025-12-31 | (534, 14, 161) | 19.29% |

Class weights: `{0: 1.0, 1: 7.59}`. Flattened representation for classical models: 2,254 features (14 × 161).

## Model comparison (test set)

| Model | ROC-AUC | Avg. Precision | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.748 | 0.415 | 0.407 | 0.534 | 0.462 |
| Random Forest | 0.872 | 0.529 | 0.524 | 0.107 | 0.177 |
| CatBoost | 0.834 | 0.607 | 0.396 | 0.650 | 0.493 |
| **BiLSTM** | **0.877** | 0.577 | 0.401 | **0.922** | **0.559** |

Baseline (random classifier) precision equals the test positive rate, 0.193.

### Reading the trade-offs

**Random Forest is the instructive failure.** ROC-AUC 0.872 sits second overall, yet recall is 0.107. The model ranks days well but its probability distribution is compressed near the low end, so at the operating threshold it fires almost no alerts. If AUC were the only reported metric, this would look like a near-tie with the winning model while being operationally useless.

**CatBoost holds the best average precision** (0.607 vs. 0.577). Under a precision-first objective, where false alarms are the binding constraint, CatBoost is the better pick. The BiLSTM's advantage is specifically that it misses far fewer real events.

**Precision is ~0.40 across the board.** Around three in five alerts are not followed by a detected fire day. Some of those are genuine near-misses where fire weather was severe and ignition simply did not occur, but from a duty officer's chair they are false alarms. This is the main obstacle to operational use and the first thing worth improving.

**BiLSTM parameter count**: 50,977 (50,849 trainable). Small enough to train on CPU in minutes, which matters for a system meant to be retrained regularly.

## BiLSTM permutation importance

20 shuffles per feature, reported as mean ROC-AUC drop ± standard deviation. Baseline test AUC: 0.8773.

| # | Feature | ΔAUC |
|---|---|---|
| 1 | `precipitation_sum_reg_min_std14` | 0.0144 ± 0.0021 |
| 2 | `relative_humidity_2m_max` | 0.0083 ± 0.0030 |
| 3 | `relative_humidity_2m_min_reg_min` | 0.0082 ± 0.0024 |
| 4 | `sunshine_duration` | 0.0079 ± 0.0018 |
| 5 | `temperature_2m_mean_reg_max_std14` | 0.0079 ± 0.0020 |
| 6 | `relative_humidity_2m_mean` | 0.0073 ± 0.0016 |
| 7 | `is_fire_season` | 0.0072 ± 0.0019 |
| 8 | `temperature_2m_mean_std14` | 0.0066 ± 0.0017 |
| 9 | `vapour_pressure_deficit_max_ma3` | 0.0065 ± 0.0023 |
| 10 | `temperature_2m_max_reg_max_std7` | 0.0062 ± 0.0019 |
| 11 | `surface_pressure_mean` | 0.0058 ± 0.0017 |
| 12 | `shortwave_radiation_sum_reg_max` | 0.0055 ± 0.0023 |
| 13 | `temperature_2m_max_std14` | 0.0053 ± 0.0015 |
| 14 | `DC` (Drought Code) | 0.0053 ± 0.0026 |
| 15 | `FWI` | 0.0053 ± 0.0021 |
| 16 | `days_no_rain` | 0.0052 ± 0.0016 |
| 17 | `apparent_temperature_max_reg_max` | 0.0052 ± 0.0033 |
| 18 | `precipitation_sum_reg_min_std3` | 0.0048 ± 0.0012 |
| 19 | `vapour_pressure_deficit_mean_reg_max` | 0.0047 ± 0.0018 |
| 20 | `relative_humidity_2m_mean_reg_min` | 0.0046 ± 0.0014 |

Three things stand out.

**Variability beats level.** The top feature is the 14-day standard deviation of minimum regional precipitation, not its total. How unevenly rain arrives matters more than how much arrives, which fits the physics: steady drizzle keeps fine fuel damp, while the same total delivered as one storm followed by twelve dry days leaves the litter fully cured.

**Humidity outranks temperature.** Four of the top six are humidity variables. Temperature enters mostly through 14-day standard deviations rather than levels, again pointing at spell structure over daily conditions.

**The FWI system earns its place.** DC and FWI both contribute measurably, which is a useful independent check: a fire-danger index calibrated on Canadian forests in 1985 still carries signal in Ukrainian peat forest four decades later.

## Alert tier distribution (test period)

| Tier | Threshold | Days | Share |
|---|---|---|---|
| 🟢 Green | p < 0.30 | 268 | 50.2% |
| 🟠 Orange | 0.30 ≤ p ≤ 0.70 | 194 | 36.3% |
| 🔴 Red | p > 0.70 | 72 | 13.5% |

The threshold pair was set on the validation split, not the test split.

## Live inference (18 May 2026)

| | |
|---|---|
| Live weather window | 2026-03-19 → 2026-05-18 (61 days) |
| Warm-up history joined | 461 days total after concatenation |
| Predicted probability | 51.58% |
| Alert tier | 🟠 Orange |

Distribution over the 61-day live window: 5 days green (8.2%), 23 orange (37.7%), 33 red (54.1%). Spring 2026 in the region ran dry.

## Reproducing these numbers

1. Open the notebook in Colab, add `FIRMS_MAP_KEY` to Colab Secrets, run all cells.
2. First run downloads and caches data. Later runs read the cache and are much faster.
3. Seeds are fixed, but exact metrics drift by roughly ±0.01 AUC between runs because cuDNN kernels are not deterministic on GPU. The numbers above come from a CPU run. Treat sub-0.01 differences as noise.
4. `RANDOM_STATE = 42` is set in `src/config.py` and at the top of the notebook.

## Artefacts produced by a full run

| File | Contents |
|---|---|
| `polissya_dataset_raw.csv` | Merged daily weather and fire dataset |
| `polissya_features.csv` | Dataset after feature engineering |
| `polissya_tensors.npz` | Train/val/test tensors and scaler state |
| `bilstm_fire_prediction_final.keras` | Trained network |
| `dynamic_forecast.png` | Live forecast chart |
| `risk_animation.html` | Animated geospatial risk map |
| `permutation_importance_lstm.png` | Feature importance chart |
