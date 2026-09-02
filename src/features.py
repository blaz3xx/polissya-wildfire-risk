"""
Feature engineering for daily fire-weather data.

These are the transformations applied in Phase 3 of the notebook, lifted out so
they can be reused on any daily weather series. The Canadian Fire Weather Index
system lives separately in :mod:`src.fwi`.

Together these produced 121 derived features from the raw Open-Meteo variables.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["compute_vpd", "days_without_rain", "add_rolling", "add_calendar"]

# Months treated as the Polissya fire season (March through October).
FIRE_SEASON_MONTHS = (3, 4, 5, 6, 7, 8, 9, 10)


def compute_vpd(temp_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """Vapour pressure deficit in kPa, via the Tetens equation.

    VPD is the gap between how much moisture the air could hold and how much it
    actually holds, so it is a more direct measure of drying power than either
    temperature or humidity alone. Hot and humid air dries fuel slowly; hot and
    dry air strips it fast.

    Args:
        temp_c: Air temperature in degrees Celsius.
        humidity_pct: Relative humidity in percent.

    Returns:
        Vapour pressure deficit in kPa, clipped at zero.
    """
    saturation = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    actual = saturation * (humidity_pct / 100.0)
    return (saturation - actual).clip(lower=0)


def days_without_rain(precipitation: pd.Series, threshold: float = 1.0) -> pd.Series:
    """Length of the current run of days with rainfall below ``threshold``.

    Resets to zero on any day that meets or exceeds the threshold, so the value
    on day *n* is "this is the *n*-th consecutive dry day".

    Args:
        precipitation: Daily rainfall in mm, in chronological order.
        threshold: Rainfall below this counts as a dry day. 1.0 mm by default,
            since a trace of rain does not meaningfully rewet surface litter.

    Returns:
        Integer dry-spell counter aligned to the input index.
    """
    dry = (precipitation < threshold).astype(int)
    groups = (dry != dry.shift()).cumsum()
    return dry.groupby(groups).cumsum() * dry


def add_rolling(
    df: pd.DataFrame,
    columns: Iterable[str],
    windows: Sequence[int] = (3, 7, 14),
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add rolling means, standard deviations, and precipitation sums.

    Fire risk is cumulative, so the model needs memory. A single hot day means
    little; two weeks of hot days with no rain means a great deal. For rainfall
    columns a rolling sum is added as well, since total accumulated water is the
    physically meaningful quantity.

    Missing columns are skipped rather than raising, so the same column list can
    be reused across datasets with slightly different schemas.

    Args:
        df: Daily data with a date column.
        columns: Base columns to derive rolling statistics from.
        windows: Window lengths in days.
        date_col: Name of the datetime column used for ordering.

    Returns:
        A sorted copy of ``df`` with the new columns appended.
    """
    out = df.sort_values(date_col).copy()
    rain_markers = ("precip", "rain", "precipitation")

    for col in columns:
        if col not in out.columns:
            continue
        for window in windows:
            out[f"{col}_ma{window}"] = out[col].rolling(window, min_periods=1).mean()
            out[f"{col}_std{window}"] = (
                out[col].rolling(window, min_periods=2).std().fillna(0)
            )
            if any(marker in col for marker in rain_markers):
                out[f"{col}_sum{window}"] = out[col].rolling(window, min_periods=1).sum()

    return out


def add_calendar(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    """Add cyclical calendar features and a fire-season flag.

    Day-of-year is encoded as a sine/cosine pair rather than a raw integer, so
    that 31 December sits next to 1 January instead of 364 units away from it.
    Without this, a model sees the new year as a discontinuity that does not
    exist in the physical system.

    Args:
        df: Daily data with a date column.
        date_col: Name of the datetime column.

    Returns:
        A copy of ``df`` with ``day_of_year``, ``month``, ``doy_sin``,
        ``doy_cos``, and ``is_fire_season`` added.
    """
    out = df.copy()
    out["day_of_year"] = out[date_col].dt.dayofyear
    out["month"] = out[date_col].dt.month
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    out["is_fire_season"] = out["month"].isin(FIRE_SEASON_MONTHS).astype(int)
    return out
