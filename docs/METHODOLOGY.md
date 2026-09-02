# Methodology

An English walkthrough of the decisions behind the pipeline. The full formal report, with equations and figure-by-figure discussion, is in [`report/coursework-report-uk.docx`](report/coursework-report-uk.docx) (Ukrainian, 47 pages). The executable version of everything below is [`../notebooks/polissya_wildfire_risk.ipynb`](../notebooks/polissya_wildfire_risk.ipynb).

## Problem statement

Predict, from weather alone, whether the Polissya forest zone will experience a **significant fire day** tomorrow. Binary classification on a daily time series, with a positive class that occupies 12.8% of the record.

The target has to be defined carefully, because "fire" in satellite data means "a thermal anomaly a satellite happened to see". Section 3 below covers how that is turned into a usable label.

## Phase 1 — Data collection

### 1.1 Defining the region

The study area is an intersection, not a rectangle:

1. **Bounding box**: 50.5–52.5°N, 23.5–34.0°E, covering the Ukrainian Polissya belt.
2. **OSM forest polygons**: `landuse=forest` and `natural=wood`, downloaded through OSMnx.
3. **Minimum patch size**: 2 km². This is the filter that separates forest from urban greenery. City parks and roadside shelterbelts pass a naive tag query and would pollute the mask.
4. **Ukrainian administrative border**, to clip cross-border geometry.

Result: 4,145 patches, 58,276 km².

Two practical notes. Geometries are simplified to ~100 m before `unary_union`; without it the union exhausts Colab memory, and at the scale of the whole belt the loss of detail is invisible. The resulting mask is cached to GeoPackage, so re-runs read from disk in seconds instead of re-querying Overpass.

### 1.2 Weather

Open-Meteo's historical endpoint serves ERA5 reanalysis: consistent, gap-free, ~25 km resolution, free, no key.

The API takes a coordinate, not a polygon, so "the weather of the zone" is approximated by a **regular grid of points inside the forest mask**. That yields 19 nodes, each roughly one ERA5 cell.

Downloads are batched by node pair and by year (100 requests total) and cached per batch-year. Every node series is then aggregated to one row per day, keeping three statistics per variable:

- **mean** across nodes, the regional average condition,
- **regional max** (`_reg_max`), and
- **regional min** (`_reg_min`).

The extremes matter more than they look. A fire starts at the driest, hottest point in the zone, not the average one, so a day with a mild mean but a severe regional maximum is a dangerous day. Permutation importance later confirmed this: several `_reg_min` and `_reg_max` features rank above their mean counterparts.

### 1.3 Fire detections

NASA FIRMS, three sensors: MODIS (Aqua/Terra), VIIRS S-NPP, VIIRS NOAA-20. Raw pull for 2016–2025 gives 140,073 detections in the bounding box. Filtering:

| Step | Kept | Rationale |
|---|---|---|
| Confidence (nominal/high) | 119,132 | Drop low-confidence anomalies |
| Daytime overpasses | 78,734 | Night detections are noisier for this target |
| Brightness ≥ 300 K | 78,734 | Guard against weak anomalies (no-op after the above) |
| **Inside forest mask** | **17,669** | **Removes agricultural burning** |

The forest-mask step is the substantive one, dropping 78% of what survived quality filtering. Ukraine's thermal anomaly signal is dominated by stubble burning on farmland, which follows harvest scheduling rather than fire weather. A model trained on the unclipped signal learns the agricultural calendar and looks deceptively good doing it.

After clipping: 1,165 days with at least one detection (31.9%), of which 468 qualify as significant fire days (12.8%). Peak fire radiative power in the decade: 2,174 MW.

### 1.4 The joined dataset

Weather and fire labels merge on date into 3,653 rows × 50 columns, one row per day for the whole forest zone.

## Phase 2 — Exploratory analysis

Ten visualisations, each answering a modelling question rather than decorating the report:

| Figure | Question it answers |
|---|---|
| Multivariate time series | Do fire clusters visually coincide with weather extremes? |
| Year × month heatmap | Is seasonality stable across years, or drifting? |
| Interactive Folium map | Are detections spread through the mask or concentrated? |
| Spearman correlation matrix | Which predictors are redundant? |
| Box plots by class | Which variables separate fire days from quiet days? |
| ACF / PACF | **How long should the input window be?** |
| Kernel densities | Do the classes overlap, or is separation clean? |
| STL decomposition | How much variance is seasonal vs. residual? |
| Monthly dynamics | When is the operational season? |
| Wind rose | Is there a directional wind regime on fire days? |

Two results carried directly into modelling. The PACF cutoff set the **14-day window**, which is why the model looks back exactly two weeks rather than an arbitrary round number. And an Augmented Dickey-Fuller test rejected the unit root for temperature (p = 3.9e-03) and relative humidity (p = 2.0e-07), confirming the inputs are stationary enough to feed the network without differencing.

Seasonality is bimodal: a spring peak after snowmelt but before green-up, when last year's dead grass is exposed and dry, and a stronger August–October peak.

## Phase 3 — Feature engineering

121 derived features, 161 predictors after selection.

**Vapour pressure deficit** (Tetens equation) measures the air's drying power directly, which neither temperature nor humidity does alone. Mean 0.381 kPa, max 1.80 kPa.

**Canadian Fire Weather Index system**, implemented from Van Wagner & Pickett (1985): FFMC → DMC → DC → ISI → BUI → FWI. The three moisture codes are recursive with time constants of roughly 1–2, 12, and 52 days, which is exactly the multi-scale memory the problem needs. Because each day depends on the last, the series is walked sequentially with carried state. Mean FWI 2.55, max 30.3. Implementation: [`../src/fwi.py`](../src/fwi.py).

**Dry spell length**: consecutive days under 1 mm. A trace of rain does not rewet litter, so the threshold is 1 mm rather than zero. Longest spell in the decade: 21 days.

**Rolling statistics** over 3/7/14 days: means and standard deviations for all base variables, plus cumulative sums for precipitation. 108 features. Standard deviation turned out to matter more than the mean, see the results discussion.

**Cyclical calendar encoding**: sine and cosine of day-of-year, so the year wraps continuously, plus a binary fire-season flag for March–October.

### Preventing leakage

Every fire-derived column is dropped before feature selection: the target, detection counts, FRP, and anything computed from them. Only weather-derived predictors survive. This is what makes the system predictive rather than descriptive, and it also means the live module can run on forecast weather alone.

### Splitting and shaping

The split is **chronological, never random**:

| Split | Days | Range | Positive rate |
|---|---|---|---|
| Train | 2,557 | 2016-01-01 → 2022-12-31 | 11.6% |
| Validation | 548 | 2023-01-01 → 2024-07-01 | 12.9% |
| Test | 548 | 2024-07-02 → 2025-12-31 | 19.3% |

Random splitting a time series would place tomorrow in the training set and today in the test set. Since weather is autocorrelated over days, that leaks the answer and inflates every metric.

`StandardScaler` is fitted on training data only, then applied to all three splits. Sliding 14-day windows produce tensors of `(2543, 14, 161)`, `(534, 14, 161)`, `(534, 14, 161)`. The first 14 rows of each split are consumed by the window, which is why sample counts fall slightly below day counts.

Class weighting sets the positive class to 7.59, the inverse of its training frequency. Without it, a model reaches 88% accuracy by predicting "no fire" forever.

Note the test period has a higher base rate (19.3%) than training (11.6%). This is honest but harder: the model faces a distribution it was not trained on, which is exactly what happens in deployment.

## Phase 4 — Modelling

Four models on identical tensors. The three classical models receive the window flattened to 2,254 features, so they see the same information as the network but with no notion that the time axis is ordered.

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline; establishes how much of the signal is linear |
| Random Forest | Non-linear interactions without temporal structure |
| CatBoost | Strong gradient boosting reference, early stopping on validation |
| **BiLSTM** | The proposal; explicitly models the time axis |

Architecture, 50,977 parameters total:

```
Input (14, 161)
Bidirectional(LSTM(32))     49,664
BatchNormalization             256
Dropout(0.3)
Dense(16, relu)              1,040
Dropout(0.2)
Dense(1, sigmoid)               17
```

Bidirectional matters here because the network is classifying a fixed window rather than streaming: reading the fortnight backwards as well as forwards lets it relate the most recent days to how the whole spell developed.

Training uses early stopping on validation AUC with best-weight restore, plus learning-rate reduction on plateau. The **classification threshold is chosen on the validation split** by maximising F1, then frozen. The test split is touched exactly once, for the final numbers.

## Phase 5 — Live inference

The live module fetches recent and forecast weather for the same 19 nodes and produces a current risk reading.

Its one non-obvious requirement is **warm-up**. FWI codes are recursive and rolling windows need history, so scoring an isolated week produces meaningless indices: the Drought Code would start from its default rather than from the actual accumulated dryness of the season. The module prepends a **400-day tail** of archived weather, recomputes the whole feature pipeline over the joined series, and keeps only the recent portion. Fire columns are stripped from that tail so no label information enters inference.

Outputs: current probability, alert tier, a 60-day forecast trace, and an animated Plotly map of risk over time.

## Evaluation choices

- **ROC-AUC** as the headline ranking metric, since it is threshold-independent and robust to imbalance.
- **Average Precision** because with a 19% positive rate, the PR curve is more informative about the rare class than ROC.
- **Recall** weighted heavily in interpretation. A missed fire day is not symmetric with a false alarm.
- **F1** for the chosen operating point.
- **Confusion matrices** so false-negative counts are visible rather than folded into an average.

## Threats to validity

- FIRMS under-detects small, brief, night-time, and cloud-obscured fires, so the label is a proxy for fire activity rather than ground truth.
- ERA5's ~25 km grid cannot resolve microclimate inside a forest patch.
- No anthropogenic ignition predictors, so the model predicts fire *weather* and relies on ignition sources being ubiquitous.
- One region, one decade. Peat behaviour makes Polissya unusual, so nothing here should be assumed to transfer.
- GPU non-determinism moves metrics by roughly a percentage point between runs even with fixed seeds.
