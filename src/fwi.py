"""
Canadian Forest Fire Weather Index (FWI) System.

A direct implementation of the equations in Van Wagner & Pickett (1985),
"Equations and FORTRAN Program for the Canadian Forest Fire Weather Index
System", Forestry Technical Report 33.

The system chains six indices:

    FFMC  Fine Fuel Moisture Code   moisture of fine surface litter (1-2 day memory)
    DMC   Duff Moisture Code        moisture of loosely compacted organic layers (~12 days)
    DC    Drought Code              deep compact organic layer moisture (~52 days)
    ISI   Initial Spread Index      FFMC + wind, expected rate of spread
    BUI   Buildup Index             DMC + DC, total fuel available to burn
    FWI   Fire Weather Index        ISI + BUI, general fire intensity

The moisture codes are recursive: each day's value depends on the previous
day's. The series therefore has to be walked in order, carrying state, which
is why this is a Python loop rather than a vectorised expression.

Because DC has roughly a 52-day time constant, feeding this function a short
slice of data gives meaningless results for the first weeks. When scoring live
data, prepend a warm-up tail of at least ~400 days of history.

Inputs are daily: mean temperature (C), mean relative humidity (%), maximum
wind speed (km/h), and rainfall (mm).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = ["add_fwi", "fwi_step", "DMC_DAY_LENGTH", "DC_DAY_LENGTH_FACTOR"]

# Effective day length by month (Jan..Dec), used by the Duff Moisture Code.
DMC_DAY_LENGTH = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]

# Day length adjustment by month (Jan..Dec), used by the Drought Code.
DC_DAY_LENGTH_FACTOR = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]

# Starting values for the moisture codes on the first day of a series.
DEFAULT_START = {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0}

# Candidate column names, in priority order, for each required input.
_TEMP_CANDIDATES = [
    "temperature_2m_mean", "temperature_2m_mean_reg_max",
    "temp_mean", "temperature_2m",
]
_HUMIDITY_CANDIDATES = [
    "relative_humidity_2m_mean", "relative_humidity_2m_mean_reg_min",
    "humidity_mean", "humidity_pct",
]
_WIND_CANDIDATES = [
    "wind_speed_10m_max", "wind_speed_10m_max_reg_max", "wind_speed",
]
_RAIN_CANDIDATES = [
    "rain_sum", "precipitation_sum", "precipitation", "precipitation_sum_reg_min",
]


def fwi_step(
    temp_c: float,
    humidity_pct: float,
    wind_kmh: float,
    rain_mm: float,
    ffmc_prev: float,
    dmc_prev: float,
    dc_prev: float,
    month_index: int,
) -> tuple[float, float, float, float, float, float]:
    """Advance the FWI system by one day.

    Args:
        temp_c: Daily mean temperature, degrees Celsius.
        humidity_pct: Daily mean relative humidity, percent (clipped to 0-100).
        wind_kmh: Daily maximum wind speed, km/h.
        rain_mm: Daily rainfall, mm.
        ffmc_prev: Previous day's Fine Fuel Moisture Code.
        dmc_prev: Previous day's Duff Moisture Code.
        dc_prev: Previous day's Drought Code.
        month_index: Zero-based month (0 = January), used for day-length tables.

    Returns:
        Tuple of (FFMC, DMC, DC, ISI, BUI, FWI) for the current day.
    """
    h = max(0.0, min(100.0, humidity_pct))
    t, w, r = temp_c, wind_kmh, rain_mm

    # --- Fine Fuel Moisture Code -------------------------------------------
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

    if r > 0.5:  # rainfall below 0.5 mm is intercepted by the canopy
        rf = r - 0.5
        mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1 - math.exp(-6.93 / rf))
        if mo > 150:
            mo += 0.0015 * (mo - 150) ** 2 * math.sqrt(rf)
        mo = min(mo, 250.0)

    # Equilibrium moisture contents for drying (ed) and wetting (ew).
    ed = (0.942 * h ** 0.679 + 11.0 * math.exp((h - 100) / 10)
          + 0.18 * (21.1 - t) * (1 - math.exp(-0.115 * h)))
    ew = (0.618 * h ** 0.753 + 10.0 * math.exp((h - 100) / 10)
          + 0.18 * (21.1 - t) * (1 - math.exp(-0.115 * h)))

    if mo > ed:  # drying
        kd = ((0.424 * (1 - (h / 100) ** 1.7)
               + 0.0694 * math.sqrt(w) * (1 - (h / 100) ** 8))
              * 0.581 * math.exp(0.0365 * t))
        m = ed + (mo - ed) * 10 ** (-kd)
    elif mo < ew:  # wetting
        kw = ((0.424 * (1 - ((100 - h) / 100) ** 1.7)
               + 0.0694 * math.sqrt(w) * (1 - ((100 - h) / 100) ** 8))
              * 0.581 * math.exp(0.0365 * t))
        m = ew - (ew - mo) * 10 ** (-kw)
    else:  # between the two equilibria, moisture is unchanged
        m = mo

    ffmc = max(0.0, min(101.0, 59.5 * (250 - m) / (147.2 + m)))

    # --- Duff Moisture Code ------------------------------------------------
    if r > 1.5:
        re = 0.92 * r - 1.27
        mo_dmc = 20 + math.exp(5.6348 - dmc_prev / 43.43)
        if dmc_prev <= 33:
            b = 100 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65:
            b = 14 - 1.3 * math.log(dmc_prev)
        else:
            b = 6.2 * math.log(dmc_prev) - 17.2
        mr = mo_dmc + 1000 * re / (48.77 + b * re)
        dmc_after_rain = max(0.0, 244.72 - 43.43 * math.log(max(1e-3, mr - 20)))
    else:
        dmc_after_rain = dmc_prev

    k = (1.894 * (t + 1.1) * (100 - h) * DMC_DAY_LENGTH[month_index] * 1e-6
         if t > -1.1 else 0.0)
    dmc = dmc_after_rain + 100 * k

    # --- Drought Code ------------------------------------------------------
    if r > 2.8:
        rd = 0.83 * r - 1.27
        qo = 800 * math.exp(-dc_prev / 400)
        qr = qo + 3.937 * rd
        dc_after_rain = max(0.0, 400 * math.log(800 / max(1e-3, qr)))
    else:
        dc_after_rain = dc_prev

    dc = dc_after_rain + 0.5 * max(0.0, 0.36 * (t + 2.8) + DC_DAY_LENGTH_FACTOR[month_index])

    # --- Initial Spread Index ----------------------------------------------
    f_wind = math.exp(0.05039 * w)
    f_fuel = 91.9 * math.exp(-0.1386 * m) * (1 + (m ** 5.31) / 4.93e7)
    isi = 0.208 * f_wind * f_fuel

    # --- Buildup Index -----------------------------------------------------
    if dmc <= 0.4 * dc:
        bui = max(0.0, 0.8 * dmc * dc / (dmc + 0.4 * dc + 1e-9))
    else:
        bui = max(0.0, dmc - (1 - 0.8 * dc / (dmc + 0.4 * dc + 1e-9))
                  * (0.92 + (0.0114 * dmc) ** 1.7))

    # --- Fire Weather Index ------------------------------------------------
    f_bui = (0.626 * bui ** 0.809 + 2.0 if bui <= 80
             else 1000 / (25 + 108.64 * math.exp(-0.023 * bui)))
    s = 0.1 * isi * f_bui
    fwi = (math.exp(2.72 * (0.434 * math.log(max(1e-3, s))) ** 0.647) if s > 1 else s)

    return ffmc, dmc, dc, isi, bui, fwi


def add_fwi(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    start: dict[str, float] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute the full FWI system for a daily weather series.

    Input columns are detected automatically from a list of known aliases, so
    the function works both on the raw Open-Meteo column names and on the
    regionally aggregated ones used in this project.

    Args:
        df: Daily weather with a datetime column and temperature, humidity,
            wind, and rain columns.
        date_col: Name of the datetime column.
        start: Optional starting values, keys ``ffmc``, ``dmc``, ``dc``.
        verbose: Print which columns were selected as inputs.

    Returns:
        A sorted copy of ``df`` with FFMC, DMC, DC, ISI, BUI, and FWI added.
        If a required input is missing, those columns are filled with NaN.
    """
    columns = {
        "T": next((c for c in _TEMP_CANDIDATES if c in df.columns), None),
        "H": next((c for c in _HUMIDITY_CANDIDATES if c in df.columns), None),
        "W": next((c for c in _WIND_CANDIDATES if c in df.columns), None),
        "R": next((c for c in _RAIN_CANDIDATES if c in df.columns), None),
    }

    missing = [name for name, col in columns.items() if col is None]
    if missing:
        out = df.copy()
        for col in ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI"):
            out[col] = np.nan
        if verbose:
            print(f"[FWI] Missing required inputs {missing}; columns filled with NaN.")
        return out

    if verbose:
        print("[FWI] Inputs: " + " | ".join(f"{k}={v}" for k, v in columns.items()))

    out = df.sort_values(date_col).copy()

    temps = out[columns["T"]].fillna(10.0).to_numpy()
    humidity = out[columns["H"]].fillna(60.0).to_numpy()
    wind = out[columns["W"]].fillna(5.0).to_numpy()
    rain = out[columns["R"]].fillna(0.0).to_numpy()
    months = out[date_col].dt.month.to_numpy() - 1

    s = {**DEFAULT_START, **(start or {})}
    ffmc_prev, dmc_prev, dc_prev = s["ffmc"], s["dmc"], s["dc"]

    results: dict[str, list[float]] = {k: [] for k in ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI")}

    for i in range(len(out)):
        ffmc, dmc, dc, isi, bui, fwi = fwi_step(
            float(temps[i]), float(humidity[i]), float(wind[i]), float(rain[i]),
            ffmc_prev, dmc_prev, dc_prev, int(months[i]),
        )
        results["FFMC"].append(ffmc)
        results["DMC"].append(dmc)
        results["DC"].append(dc)
        results["ISI"].append(isi)
        results["BUI"].append(bui)
        results["FWI"].append(fwi)

        # Carry the moisture codes into the next day.
        ffmc_prev, dmc_prev, dc_prev = ffmc, dmc, dc

    for name, values in results.items():
        out[name] = values

    return out
