"""Run additive and log-scale multiplicative STL decomposition."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


# Kept locally so this script runs without project_utils.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "azure_llm_one_minute.csv"
TIME_COLUMN = "minute"
TARGET_COLUMN = "total_generated_tokens"
NUMERIC_COLUMNS = [
    "code_request_count", "conversation_request_count",
    "code_context_tokens", "conversation_context_tokens",
    "code_generated_tokens", "conversation_generated_tokens",
    "total_request_count", "total_context_tokens",
    "total_generated_tokens", "total_tokens",
]

def resolve_from_root(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

def prepare_output_dir(path: str | Path) -> Path:
    directory = resolve_from_root(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def load_processed_data(path: str | Path = DEFAULT_PROCESSED_DATA) -> pd.DataFrame:
    data_path = resolve_from_root(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}. Run 01_build_dataset.py first."
        )
    frame = pd.read_csv(data_path)
    missing_columns = sorted(set([TIME_COLUMN, *NUMERIC_COLUMNS]) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], utc=True, errors="coerce")
    if frame[TIME_COLUMN].isna().any():
        raise ValueError("Invalid minute timestamps in processed dataset.")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values(TIME_COLUMN).reset_index(drop=True)
    if frame[TIME_COLUMN].duplicated().any():
        raise ValueError("Duplicate timestamps in processed dataset.")
    if frame[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError("Missing numerical values in processed dataset.")
    if (frame[NUMERIC_COLUMNS] < 0).any().any():
        raise ValueError("Negative workload counts in processed dataset.")
    if len(frame) < 3:
        raise ValueError("At least three one-minute observations are required.")
    if not frame[TIME_COLUMN].diff().dropna().eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("Time series is not sampled at uninterrupted one-minute intervals.")
    for derived, expected in {
        "total_request_count": frame["code_request_count"] + frame["conversation_request_count"],
        "total_context_tokens": frame["code_context_tokens"] + frame["conversation_context_tokens"],
        "total_generated_tokens": frame["code_generated_tokens"] + frame["conversation_generated_tokens"],
        "total_tokens": frame["total_context_tokens"] + frame["total_generated_tokens"],
    }.items():
        if not np.array_equal(frame[derived].to_numpy(), expected.to_numpy()):
            raise ValueError(f"Inconsistent derived column: {derived}")
    return frame

def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.80):
    if not 0 < train_fraction < 1:
        raise ValueError("--train-fraction must be strictly between 0 and 1.")
    boundary = int(len(frame) * train_fraction)
    if not 0 < boundary < len(frame):
        raise ValueError("Train/test split creates an empty sample.")
    return frame.iloc[:boundary].copy(), frame.iloc[boundary:].copy(), boundary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", default=str(DEFAULT_PROCESSED_DATA))
    parser.add_argument("--output-dir", default="outputs/decomposition")
    parser.add_argument("--seasonal-period", type=int, default=1440)
    return parser.parse_args()


def component_strength(component: np.ndarray, residual: np.ndarray) -> float:
    denominator = np.var(component + residual, ddof=1)
    if denominator == 0:
        return 0.0
    return float(max(0.0, 1.0 - np.var(residual, ddof=1) / denominator))


def rolling_variance_cv(series: pd.Series, window: int) -> float:
    rolling_variance = series.rolling(window).var().dropna()
    mean_variance = rolling_variance.mean()
    if rolling_variance.empty or mean_variance == 0:
        return float("nan")
    return float(rolling_variance.std() / mean_variance)


def plot_components(time, observed, trend, seasonal, residual, title, output_path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    labels = ["Observed", "Trend", "Seasonal", "Residual"]
    values = [observed, trend, seasonal, residual]
    for axis, label, values_to_plot in zip(axes, labels, values):
        axis.plot(time, values_to_plot, linewidth=0.7)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=time.dt.tz))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.seasonal_period <= 1:
        raise ValueError("--seasonal-period must be greater than one.")

    output_dir = prepare_output_dir(args.output_dir)
    frame = load_processed_data(args.input_file)
    target = frame[TARGET_COLUMN].astype(float)
    if len(target) < 2 * args.seasonal_period:
        raise ValueError("STL requires at least two complete seasonal periods.")

    # Full-sample STL is descriptive, NOT a fitted forecasting model.
    # Re-fit decomposition on training folds before estimating forecast accuracy.
    additive = STL(target, period=args.seasonal_period, robust=True).fit()
    log_target = np.log1p(target)
    multiplicative = STL(log_target, period=args.seasonal_period, robust=True).fit()
    # The multiplicative identity is for y + 1 because of log1p, not y alone.
    if not np.allclose(target, additive.trend + additive.seasonal + additive.resid):
        raise AssertionError("Additive STL reconstruction failed.")
    if not np.allclose(
        target, np.expm1(multiplicative.trend + multiplicative.seasonal + multiplicative.resid)
    ):
        raise AssertionError("Log-scale STL reconstruction failed.")

    additive_strengths = {
        "trend_strength": component_strength(additive.trend, additive.resid),
        "seasonality_strength": component_strength(additive.seasonal, additive.resid),
    }
    multiplicative_strengths = {
        "trend_strength": component_strength(multiplicative.trend, multiplicative.resid),
        "seasonality_strength": component_strength(multiplicative.seasonal, multiplicative.resid),
    }

    additive_components = pd.DataFrame(
        {
            TIME_COLUMN: frame[TIME_COLUMN],
            "observed": target,
            "trend": additive.trend,
            "seasonal": additive.seasonal,
            "residual": additive.resid,
            "detrended": target - additive.trend,
            "seasonally_adjusted": target - additive.seasonal,
        }
    )
    additive_components.to_csv(output_dir / "additive_stl_components.csv", index=False)

    multiplicative_components = pd.DataFrame(
        {
            TIME_COLUMN: frame[TIME_COLUMN],
            "observed": target,
            "log_observed": log_target,
            "log_trend": multiplicative.trend,
            "log_seasonal": multiplicative.seasonal,
            "log_residual": multiplicative.resid,
            "trend_original_scale": np.expm1(multiplicative.trend),
            "seasonal_factor": np.exp(multiplicative.seasonal),
            "residual_factor": np.exp(multiplicative.resid),
            "detrended": (target + 1) / np.exp(multiplicative.trend) - 1,
            "seasonally_adjusted": (target + 1) / np.exp(multiplicative.seasonal) - 1,
        }
    )
    multiplicative_components.to_csv(
        output_dir / "multiplicative_stl_components.csv", index=False
    )

    plot_components(
        frame[TIME_COLUMN],
        target,
        additive.trend,
        additive.seasonal,
        additive.resid,
        "Additive STL decomposition",
        output_dir / "additive_stl.png",
    )
    plot_components(
        frame[TIME_COLUMN],
        log_target,
        multiplicative.trend,
        multiplicative.seasonal,
        multiplicative.resid,
        "Multiplicative STL decomposition on log1p scale",
        output_dir / "multiplicative_stl_log_scale.png",
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(
        frame[TIME_COLUMN], additive_components["detrended"], linewidth=0.65, color="#1668a8"
    )
    axes[0].set(title="Additive detrended series", ylabel="Generated tokens")
    axes[1].plot(
        frame[TIME_COLUMN],
        additive_components["seasonally_adjusted"],
        linewidth=0.65,
        color="#27823b",
    )
    axes[1].set(
        title="Additive seasonally adjusted series",
        xlabel="Time (UTC)",
        ylabel="Generated tokens",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "detrended_and_seasonally_adjusted.png", dpi=200)
    plt.close(fig)

    additive_variance_cv = rolling_variance_cv(pd.Series(additive.resid), args.seasonal_period)
    multiplicative_variance_cv = rolling_variance_cv(
        pd.Series(multiplicative.resid), args.seasonal_period
    )
    lower_instability = (
        "multiplicative_log_scale"
        if multiplicative_variance_cv < additive_variance_cv
        else "additive"
    )
    comparison = pd.DataFrame(
        [
            {
                "model": "additive",
                **additive_strengths,
                "residual_rolling_variance_cv": additive_variance_cv,
            },
            {
                "model": "multiplicative_log_scale",
                **multiplicative_strengths,
                "residual_rolling_variance_cv": multiplicative_variance_cv,
            },
        ]
    )
    comparison["residual_units"] = ["generated_tokens", "log1p_generated_tokens"]
    comparison.to_csv(output_dir / "decomposition_comparison.csv", index=False)

    lines = [
        "CloudCast STL decomposition",
        "===========================",
        f"Seasonal period: {args.seasonal_period:,} one-minute observations",
        "",
        "Additive STL",
        "------------",
        f"Trend strength: {additive_strengths['trend_strength']:.6f}",
        f"Seasonality strength: {additive_strengths['seasonality_strength']:.6f}",
        f"Residual rolling-variance CV: {additive_variance_cv:.6f}",
        "",
        "Multiplicative STL (log1p scale)",
        "--------------------------------",
        f"Trend strength: {multiplicative_strengths['trend_strength']:.6f}",
        f"Seasonality strength: {multiplicative_strengths['seasonality_strength']:.6f}",
        f"Residual rolling-variance CV: {multiplicative_variance_cv:.6f}",
        "",
        f"Lower within-scale residual-variance CV: {lower_instability}",
        "THIS IS NOT A VALID RANKING OF FORECAST ACCURACY.",
        "The additive residuals are in tokens; log-scale residuals are in log1p units.",
        "Both STL fits use all timestamps and therefore must not be used for held-out forecasting.",
        "",
        "The residual rolling-variance coefficient of variation is a scale-free",
        "diagnostic, not a forecast-accuracy score. Final decomposition selection",
        "should also consider the component plots and residual autocorrelation.",
        "To choose based on predictive accuracy, fit candidate forecast methods",
        "using only the training segment and evaluate in common original-token units.",
    ]
    (output_dir / "decomposition_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(comparison.to_string(index=False))
    print(f"\nDecomposition outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

