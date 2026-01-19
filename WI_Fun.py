import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ta_plot(
    df: pd.DataFrame,
    ta_col: str = "Ta",
    year_col: str = "Year",
    month_col: str = "Month",
    day_col: str = "Day",
    hour_col: str = "Hour",
    minute_col: str = "Minute",
    timestamp_col: str = "timestamp",
    font_family: str = "Garamond",
    font_size: int = 12,
    figsize: tuple[int, int] = (14, 6),
    alpha_hourly: float = 0.2,
    show_hourly: bool = True,
) -> pd.DataFrame:
    """
    Builds a timestamp column (if needed), sorts, drops invalid timestamps,
    computes daily min/mean/max, and plots.

    Returns a daily dataframe with columns: daily_min, daily_mean, daily_max
    """
    d = df.copy()

    # 1) Build timestamp if not already present
    if timestamp_col not in d.columns:
        d[timestamp_col] = pd.to_datetime(
            dict(
                year=d[year_col],
                month=d[month_col],
                day=d[day_col],
                hour=d[hour_col],
                minute=d[minute_col],
            ),
            errors="coerce",
        )

    # 2) Sort & drop invalid timestamps
    d = d.sort_values(timestamp_col).dropna(subset=[timestamp_col])

    # Ensure Ta is numeric
    d[ta_col] = pd.to_numeric(d[ta_col], errors="coerce")

    # Font settings
    plt.rcParams.update({"font.family": font_family, "font.size": font_size})

    # Daily stats
    s = d.set_index(timestamp_col)[ta_col]
    daily = pd.DataFrame(
        {
            "daily_min": s.resample("D").min(),
            "daily_mean": s.resample("D").mean(),
            "daily_max": s.resample("D").max(),
        }
    )

    # Plot
    plt.figure(figsize=figsize)

    if show_hourly:
        plt.plot(d[timestamp_col], d[ta_col], alpha=alpha_hourly, label="Hourly Ta")

    plt.plot(daily.index, daily["daily_max"].values, label="Daily max Ta")
    plt.plot(daily.index, daily["daily_mean"].values, label="Daily mean Ta")
    plt.plot(daily.index, daily["daily_min"].values, label="Daily min Ta")

    plt.xlabel("Time")
    plt.ylabel("Ta (°C)")
    plt.title("Hourly Air Temperature with Daily Max/Mean/Min")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return daily


def winkler_index(
    df,
    ta_col: str = "Ta",
    year_col: str = "Year",
    month_col: str = "Month",
    day_col: str = "Day",
    hour_col: str = "Hour",
    minute_col: str = "Minute",
    base_temp_c: float = 10.0,
    season_start: str = "04-01",   # MM-DD
    season_end: str = "10-31",     # MM-DD
    min_samples_per_day: int = 18,
    missing_flag: float = -9999.0,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Winkler Index using:
      GDD_d = max(0, (Tmax_d + Tmin_d)/2 - base)

    Returns:
      summary_df: per-year Winkler Index + diagnostics
      daily_df: daily Tmin/Tmax/Tmean_est + GDD + season flags
    """

    d = df.copy()

    # Clean Ta
    d[ta_col] = d[ta_col].replace(missing_flag, np.nan)
    d[ta_col] = pd.to_numeric(d[ta_col], errors="coerce")

    # Build timestamp if needed
    if timestamp_col not in d.columns:
        d[timestamp_col] = pd.to_datetime(
            dict(
                year=d[year_col],
                month=d[month_col],
                day=d[day_col],
                hour=d[hour_col],
                minute=d[minute_col],
            ),
            errors="coerce",
        )

    d = d.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

    # Daily min/max from hourly
    s = d.set_index(timestamp_col)[ta_col]
    daily = pd.DataFrame({
        "n_samples": s.resample("D").count(),
        "tmin": s.resample("D").min(),
        "tmax": s.resample("D").max(),
    })

    # Require enough hourly coverage
    good = daily["n_samples"] >= min_samples_per_day
    daily.loc[~good, ["tmin", "tmax"]] = np.nan

    # Mean estimated from Tmax/Tmin
    daily["tmean_est"] = (daily["tmax"] + daily["tmin"]) / 2.0

    # Daily GDD
    daily["gdd"] = (daily["tmean_est"] - base_temp_c).clip(lower=0)

    # Season flags per year
    sm, sd = map(int, season_start.split("-"))
    em, ed = map(int, season_end.split("-"))

    daily["year"] = daily.index.year
    daily["in_season"] = False
    for y in daily["year"].unique():
        start = pd.Timestamp(year=int(y), month=sm, day=sd)
        end = pd.Timestamp(year=int(y), month=em, day=ed)
        daily.loc[(daily.index >= start) & (daily.index <= end), "in_season"] = True

    daily["gdd_in_season"] = daily["gdd"].where(daily["in_season"], 0)

    # Per-year summary
    summary = (
        daily.groupby("year", as_index=False)
        .agg(
            winkler_index=("gdd_in_season", "sum"),
            season_days_total=("in_season", "sum"),
            season_days_used=("gdd", lambda x: x[daily.loc[x.index, "in_season"]].notna().sum()),
        )
    )
    summary["season_days_dropped"] = summary["season_days_total"] - summary["season_days_used"]

    return summary, daily
