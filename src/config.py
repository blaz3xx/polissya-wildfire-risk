"""
Single place for the constants that define the study.

Everything here matches the values used in the notebook run reported in the
README. Changing any of them invalidates the published metrics.
"""

from __future__ import annotations

# --- Study area ------------------------------------------------------------
# Bounding box of the Ukrainian Polissya forest belt.
POLISSYA_BBOX = {
    "lat_min": 50.5,
    "lat_max": 52.5,
    "lon_min": 23.5,
    "lon_max": 34.0,
}

# OpenStreetMap tags used to build the forest mask.
FOREST_TAGS = {"landuse": "forest", "natural": "wood"}

# Forest patches smaller than this are dropped, which removes parks and
# roadside tree lines while keeping real forest blocks.
MIN_FOREST_AREA_KM2 = 2.0

# Geometry simplification tolerance in metres, applied before unioning the
# polygons. Without it the union of tens of thousands of detailed shapes
# exhausts memory in a standard Colab session.
SIMPLIFY_TOLERANCE_M = 100

# --- Period ----------------------------------------------------------------
DATE_START = "2016-01-01"
DATE_END = "2025-12-31"

# --- FIRMS filtering -------------------------------------------------------
MIN_BRIGHTNESS_K = 300.0     # discard weak thermal anomalies
DAYTIME_ONLY = True          # night overpasses are noisier for this target

# --- Modelling -------------------------------------------------------------
RANDOM_STATE = 42

# Input window length in days, chosen from the PACF cutoff of the weather
# series rather than picked as a round number.
TIME_STEPS = 14

# Chronological split, never random: shuffling a time series leaks the future.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

TARGET_COLUMN = "significant_fire"

# --- Alert tiers -----------------------------------------------------------
# Probability thresholds separating green / orange / red.
RISK_THRESHOLDS = (0.30, 0.70)

RISK_LEVELS = {
    0: {"name": "Green",  "colour": "green",  "action": "Routine monitoring"},
    1: {"name": "Orange", "colour": "orange", "action": "Restrict forest access, ready crews"},
    2: {"name": "Red",    "colour": "red",    "action": "Pre-position aviation and firefighting units"},
}

# --- Live inference --------------------------------------------------------
# History prepended before scoring live weather. Long enough for the Drought
# Code (~52-day time constant) and the 14-day rolling windows to converge.
WARMUP_DAYS = 400


def get_risk_level(probability: float) -> tuple[int, str, str]:
    """Map a predicted probability onto an alert tier.

    Args:
        probability: Model output in [0, 1].

    Returns:
        Tuple of (tier index, colour, tier name).
    """
    low, high = RISK_THRESHOLDS
    tier = 2 if probability > high else (1 if probability >= low else 0)
    level = RISK_LEVELS[tier]
    return tier, level["colour"], level["name"]
