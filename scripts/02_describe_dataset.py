"""Create the dataset-description tables and figures required for the report."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", default=str(DEFAULT_PROCESSED_DATA))
    parser.add_argument("--output-dir", default="outputs/dataset_description")
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--acf-lags", type=int, default=1440)
    parser.add_argument("--pacf-lags", type=int, default=120)
    return parser.parse_args()


def save_target_plot(frame, split_index: int, output_dir) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(frame[TIME_COLUMN], frame[TARGET_COLUMN], linewidth=0.8, color="#1668a8")
    split_time = frame.iloc[split_index][TIME_COLUMN]
    ax.axvline(split_time, color="#c43c39", linestyle="--", linewidth=1.5, label=f"Chronological split ({split_index / len(frame):.0%} train)")
    ax.set(
        title="Azure LLM generated-token demand",
        xlabel="Time (UTC)",
        ylabel="Generated tokens per minute",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=frame[TIME_COLUMN].dt.tz))
    fig.tight_layout()
    fig.savefig(output_dir / "target_vs_time.png", dpi=200)
    plt.close(fig)


def save_acf_pacf(frame, acf_lags: int, pacf_lags: int, output_dir) -> None:
    target = frame[TARGET_COLUMN].to_numpy(dtype=float)
    max_acf = min(acf_lags, len(target) // 2 - 1)
    max_pacf = min(pacf_lags, len(target) // 2 - 1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    plot_acf(target, lags=max_acf, alpha=0.05, fft=True, ax=axes[0])
    axes[0].set(title="ACF of raw generated-token demand", xlabel="Lag in minutes", ylabel="ACF")
    plot_pacf(target, lags=max_pacf, alpha=0.05, method="ywm", ax=axes[1])
    axes[1].set(title="PACF of raw generated-token demand", xlabel="Lag in minutes", ylabel="PACF")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "raw_acf_pacf.png", dpi=200)
    plt.close(fig)


def save_correlation_heatmap(correlation, output_dir) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Pearson correlations among one-minute workload measurements")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = prepare_output_dir(args.output_dir)
    frame = load_processed_data(args.input_file)
    if args.acf_lags < 1 or args.pacf_lags < 1:
        raise ValueError("ACF and PACF lag requests must be positive.")
    train, test, split_index = chronological_split(frame, args.train_fraction)

    descriptive = frame[NUMERIC_COLUMNS].describe().T
    descriptive.to_csv(output_dir / "descriptive_statistics.csv")
    missing = frame.isna().sum().rename("missing_count").to_frame()
    missing["missing_percent"] = 100 * missing["missing_count"] / len(frame)
    missing.to_csv(output_dir / "missing_values.csv")

    correlation = frame[NUMERIC_COLUMNS].corr(method="pearson")
    correlation.to_csv(output_dir / "pearson_correlation_matrix.csv")
    # Contemporaneous total columns are retained for description only,
    # not as candidate contemporaneous predictors of the target.
    candidate_columns = [
        "code_request_count", "conversation_request_count",
        "code_context_tokens", "conversation_context_tokens",
    ]
    frame[candidate_columns].corr(method="pearson").to_csv(
        output_dir / "predictor_candidate_correlations.csv"
    )

    save_target_plot(frame, split_index, output_dir)
    save_acf_pacf(frame, args.acf_lags, args.pacf_lags, output_dir)
    save_correlation_heatmap(correlation, output_dir)

    target_correlations = (
        correlation[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    summary = {
        "rows": len(frame),
        "numerical_features": len(NUMERIC_COLUMNS),
        "first_timestamp": frame[TIME_COLUMN].iloc[0],
        "last_timestamp": frame[TIME_COLUMN].iloc[-1],
        "sampling_frequency": "1 minute",
        "missing_values_total": int(frame.isna().sum().sum()),
        "duplicate_timestamps": int(frame[TIME_COLUMN].duplicated().sum()),
        "target": TARGET_COLUMN,
        "target_min": int(frame[TARGET_COLUMN].min()),
        "target_max": int(frame[TARGET_COLUMN].max()),
        "target_mean": float(frame[TARGET_COLUMN].mean()),
        "target_standard_deviation": float(frame[TARGET_COLUMN].std()),
        "train_fraction": args.train_fraction,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_start": train[TIME_COLUMN].iloc[0],
        "train_end": train[TIME_COLUMN].iloc[-1],
        "test_start": test[TIME_COLUMN].iloc[0],
        "test_end": test[TIME_COLUMN].iloc[-1],
        "target_correlations": target_correlations.to_dict(),
        "modeling_warning": "Totals are algebraic combinations of service-level columns. Only predictors known at the forecast origin (e.g., lagged values) may be used to prevent leakage.",
    }
    write_json(output_dir / "dataset_summary.json", summary)

    report_lines = [
        "CloudCast dataset description",
        "=============================",
        f"Observations: {len(frame):,}",
        f"Numerical features: {len(NUMERIC_COLUMNS)}",
        f"Period: {frame[TIME_COLUMN].iloc[0]} through {frame[TIME_COLUMN].iloc[-1]}",
        "Sampling frequency: one minute",
        f"Missing values: {int(frame.isna().sum().sum())}",
        f"Duplicate timestamps: {int(frame[TIME_COLUMN].duplicated().sum())}",
        "",
        "Chronological split",
        "-------------------",
        f"Training observations: {len(train):,} ({100 * args.train_fraction:.0f}%)",
        f"Training period: {train[TIME_COLUMN].iloc[0]} through {train[TIME_COLUMN].iloc[-1]}",
        f"Test observations: {len(test):,} ({100 * (1 - args.train_fraction):.0f}%)",
        f"Test period: {test[TIME_COLUMN].iloc[0]} through {test[TIME_COLUMN].iloc[-1]}",
        "",
        "CAUTION: Aggregated totals are exact linear combinations; do not pass all ten columns into the same OLS model.",
        "For forecasting, use predictors observed at the forecast origin, not same-minute target components.",
        "",
        "Strongest absolute correlations with total_generated_tokens",
        "-----------------------------------------------------------",
    ]
    report_lines.extend(
        f"{name}: {value:.6f}" for name, value in target_correlations.head(6).items()
    )
    (output_dir / "dataset_description.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print("Dataset-description outputs written to:")
    print(f"  {output_dir}")
    print(f"Train/test rows: {len(train):,} / {len(test):,}")


if __name__ == "__main__":
    main()

