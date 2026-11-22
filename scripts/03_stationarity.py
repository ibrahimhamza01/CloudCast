"""Evaluate stationarity and select a transformation for generated-token demand."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning


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
    parser.add_argument("--output-dir", default="outputs/stationarity")
    parser.add_argument("--rolling-window", type=int, default=1440)
    parser.add_argument("--seasonal-period", type=int, default=1440)
    parser.add_argument("--acf-lags", type=int, default=120)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument(
        "--evaluation-window", choices=["train", "full"], default="train",
        help="Compute stationarity tests on training data by default to avoid test leakage.",
    )
    return parser.parse_args()


def stationarity_tests(name: str, series: pd.Series) -> dict[str, object]:
    clean = series.dropna().astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        adf_result = adfuller(clean, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        kpss_result = kpss(clean, regression="c", nlags="auto")
    return {
        "transformation": name,
        "observations": len(clean),
        "adf_statistic": adf_result[0],
        "adf_p_value": adf_result[1],
        "adf_lags": adf_result[2],
        "kpss_statistic": kpss_result[0],
        "kpss_p_value": kpss_result[1],
        "kpss_lags": kpss_result[2],
        "adf_stationary_at_005": bool(adf_result[1] < 0.05),
        "kpss_stationary_at_005": bool(kpss_result[1] > 0.05),
    }


def save_rolling_plot(
    time: pd.Series,
    series: pd.Series,
    title: str,
    output_path,
    window: int,
) -> None:
    rolling_mean = series.rolling(window).mean()
    rolling_variance = series.rolling(window).var()
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(time, series, linewidth=0.65, color="#1668a8")
    axes[0].set(title=title, ylabel="Value")
    axes[1].plot(time, rolling_mean, linewidth=1.2, color="#e27602")
    axes[1].set(title=f"Rolling mean ({window:,} minutes)", ylabel="Mean")
    axes[2].plot(time, rolling_variance, linewidth=1.2, color="#27823b")
    axes[2].set(
        title=f"Rolling variance ({window:,} minutes)",
        xlabel="Time (UTC)",
        ylabel="Variance",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=time.dt.tz))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_transformation_comparison(transformations, time, output_path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    for axis, (name, series) in zip(axes.flat, transformations.items()):
        axis.plot(time, series, linewidth=0.55)
        axis.set_title(name)
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Time (UTC)")
    axes[-1, 1].set_xlabel("Time (UTC)")
    fig.suptitle("Candidate stationarity transformations", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_transformed_acf_pacf(series: pd.Series, lags: int, output_path) -> None:
    clean = series.dropna().to_numpy(dtype=float)
    max_lags = min(lags, len(clean) // 2 - 1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    plot_acf(clean, lags=max_lags, alpha=0.05, fft=True, ax=axes[0])
    axes[0].set(
        title="ACF: log plus first-differenced token demand",
        xlabel="Lag in minutes",
        ylabel="ACF",
    )
    plot_pacf(clean, lags=max_lags, alpha=0.05, method="ywm", ax=axes[1])
    axes[1].set(
        title="PACF: log plus first-differenced token demand",
        xlabel="Lag in minutes",
        ylabel="PACF",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.acf_lags < 1:
        raise ValueError("--acf-lags must be positive.")
    if args.rolling_window <= 1 or args.seasonal_period <= 1:
        raise ValueError("Rolling and seasonal windows must be greater than one.")

    output_dir = prepare_output_dir(args.output_dir)
    frame = load_processed_data(args.input_file)
    train, _, boundary = chronological_split(frame, args.train_fraction)
    analysis_frame = train if args.evaluation_window == "train" else frame
    analysis_target = analysis_frame[TARGET_COLUMN].astype(float)
    analysis_log = np.log1p(analysis_target)
    # Formal stationarity decisions are made from training data by default.
    target = frame[TARGET_COLUMN].astype(float)
    log_target = np.log1p(target)
    transformations = {
        "Raw": target,
        "Log": log_target,
        "First difference": target.diff(),
        "Log + first difference": log_target.diff(),
        "Daily seasonal difference": target.diff(args.seasonal_period),
        "Log + daily seasonal difference": log_target.diff(args.seasonal_period),
    }

    evaluation_transformations = {
        "Raw": analysis_target,
        "Log": analysis_log,
        "First difference": analysis_target.diff(),
        "Log + first difference": analysis_log.diff(),
        "Daily seasonal difference": analysis_target.diff(args.seasonal_period),
        "Log + daily seasonal difference": analysis_log.diff(args.seasonal_period),
    }
    if len(analysis_frame) <= args.seasonal_period + 10:
        raise ValueError("Too few observations for seasonal differencing and stationarity tests.")
    test_results = pd.DataFrame(
        stationarity_tests(name, series)
        for name, series in evaluation_transformations.items()
    )
    test_results.insert(0, "evaluation_window", args.evaluation_window)
    test_results.to_csv(output_dir / "stationarity_tests.csv", index=False)

    selected = transformations["Log + first difference"]
    transformed_output = pd.DataFrame(
        {
            TIME_COLUMN: frame[TIME_COLUMN],
            "log_first_difference": selected,
        }
    ).dropna()
    transformed_output.to_csv(output_dir / "log_first_differenced_target.csv", index=False)

    save_rolling_plot(
        frame[TIME_COLUMN],
        target,
        "Raw generated-token demand",
        output_dir / "raw_rolling_statistics.png",
        args.rolling_window,
    )
    save_rolling_plot(
        frame[TIME_COLUMN],
        selected,
        "Log plus first-differenced generated-token demand",
        output_dir / "transformed_rolling_statistics.png",
        args.rolling_window,
    )
    save_transformation_comparison(
        transformations,
        frame[TIME_COLUMN],
        output_dir / "transformation_comparison.png",
    )
    save_transformed_acf_pacf(
        selected,
        args.acf_lags,
        output_dir / "transformed_acf_pacf.png",
    )

    result_lines = [
        "CloudCast stationarity analysis",
        "================================",
        f"Stationarity test sample: {args.evaluation_window}",
        f"Train/test boundary index: {boundary} of {len(frame)} observations",
        "Full-series plots are exploratory; default formal tests use training observations only.",
        "",
        "ADF null hypothesis: the series has a unit root (non-stationary).",
        "KPSS null hypothesis: the series is level-stationary.",
        "",
    ]
    for row in test_results.itertuples(index=False):
        result_lines.extend(
            [
                str(row.transformation),
                "-" * len(str(row.transformation)),
                f"Observations: {row.observations:,}",
                f"ADF statistic: {row.adf_statistic:.6f}",
                f"ADF p-value: {row.adf_p_value:.6g}",
                f"KPSS statistic: {row.kpss_statistic:.6f}",
                f"KPSS p-value: {row.kpss_p_value:.6g}",
                "",
            ]
        )
    result_lines.extend(
        [
            "Selected transformation",
            "-----------------------",
            "log1p(total_generated_tokens).diff()",
            "",
            "This is a candidate transformation, not an automatic proof of stationarity.",
            "Inspect BOTH ADF and KPSS results (their null hypotheses differ) and the",
            "rolling statistics before finalizing d=1 for the forecasting model.",
            "The log1p transform reduces sensitivity to level-dependent variance.",
            "Additional differencing can cause overdifferencing and should be justified.",
        ]
    )
    (output_dir / "stationarity_summary.txt").write_text(
        "\n".join(result_lines) + "\n", encoding="utf-8"
    )

    print(test_results.to_string(index=False))
    selected_row = test_results.loc[
        test_results["transformation"].eq("Log + first difference")
    ].iloc[0]
    if not (selected_row["adf_stationary_at_005"] and selected_row["kpss_stationary_at_005"]):
        print("WARNING: ADF and KPSS do not jointly support stationarity of the chosen transform.")
    print(f"\nStationarity outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

