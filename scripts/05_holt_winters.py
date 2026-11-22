"""CloudCast Holt-Winters experiment: diagnostics, validation selection, held-out test.

Run from the project root:
    python scripts/05_holt_winters.py

Data split (one-minute observations; one seasonal period = 1,440 minutes):
    Days 1-3: fit Holt-Winters candidates
    Day 4:    select candidate by validation RMSE
    Days 1-4: refit the selected candidate
    Day 5:    evaluate ONCE on the held-out test day

This compares fixed smoothing coefficients, not an exhaustive optimized search.
The resulting one-origin 1,440-step forecasts are NOT substitutes for separate
rolling 15-, 30-, and 60-minute forecasts.

Standalone: does not import project_utils.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parents[1]
TARGET = "total_generated_tokens"

# Same six structures and three fixed coefficient settings as the validated
# CloudCast candidates experiment. Keeping the grid unchanged preserves the
# original selection method and the comparability of the reported results.
STRUCTURES = [
    ("additive_no_trend", None, "add", False),
    ("multiplicative_no_trend", None, "mul", False),
    ("additive_trend", "add", "add", False),
    ("multiplicative_trend", "add", "mul", False),
    ("additive_damped_trend", "add", "add", True),
    ("multiplicative_damped_trend", "add", "mul", True),
]
SMOOTHING_SETTINGS = [
    ("slow", 0.10, 0.01, 0.02),
    ("medium", 0.25, 0.03, 0.05),
    ("fast", 0.50, 0.06, 0.10),
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", type=Path,
        default=ROOT / "data/processed/azure_llm_one_minute.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/holt_winters",
    )
    parser.add_argument("--seasonal-period", type=int, default=1440)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_data(path: Path, period: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if not {"minute", TARGET}.issubset(df.columns):
        raise ValueError("Dataset must contain 'minute' and 'total_generated_tokens'.")
    df["minute"] = pd.to_datetime(df["minute"], utc=True, errors="raise")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")
    df = df.sort_values("minute").reset_index(drop=True)
    if len(df) != 5 * period:
        raise ValueError(f"Expected five complete periods ({5 * period} rows), got {len(df)}.")
    if df["minute"].duplicated().any():
        raise ValueError("Duplicate timestamps are not permitted.")
    if not df["minute"].diff().dropna().eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("All timestamps must be spaced exactly one minute apart.")
    if df[TARGET].isna().any() or not np.isfinite(df[TARGET].to_numpy()).all():
        raise ValueError("Target contains missing or nonfinite values.")
    if (df[TARGET] <= 0).any():
        raise ValueError("Multiplicative candidates require strictly positive demand.")
    return df


def accuracy(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "sMAPE": float(100 * np.mean(
            2 * np.abs(error) / np.maximum(np.abs(actual) + np.abs(predicted), 1e-8)
        )),
    }


def fit_and_forecast(history: np.ndarray, horizon: int, spec: dict, period: int) -> np.ndarray:
    model = ExponentialSmoothing(
        np.asarray(history, dtype=float),
        trend=spec["trend"],
        damped_trend=spec["damped"],
        seasonal=spec["seasonal"],
        seasonal_periods=period,
        initialization_method="estimated",
    )
    result = model.fit(
        optimized=False,
        smoothing_level=spec["alpha"],
        smoothing_trend=spec["beta"] if spec["trend"] else None,
        smoothing_seasonal=spec["gamma"],
        damping_trend=spec["phi"] if spec["damped"] else None,
        remove_bias=False,
    )
    forecast = np.asarray(result.forecast(horizon), dtype=float)
    if not np.isfinite(forecast).all() or (forecast <= 0).any():
        raise ValueError("Nonpositive or nonfinite forecast")
    return forecast


def diagnostics(train: np.ndarray, period: int, path: Path) -> dict:
    # Uses training data only. Hourly averaging is appropriate when there are
    # 1,440 observations/day; for a different period, compare per-position curves.
    days = train.reshape(4, period)
    if period % 24 == 0:
        profiles = days.reshape(4, 24, period // 24).mean(axis=2)
        x = np.arange(24)
        xlabel = "Hour of day (UTC)"
    else:
        profiles = days
        x = np.arange(period)
        xlabel = "Position within seasonal period"
    normalized = profiles / profiles.mean(axis=1, keepdims=True)
    correlations = np.corrcoef(normalized)

    series = pd.Series(train)
    local_window = min(60, period)
    roll_mean = series.rolling(local_window).mean()
    roll_std = series.rolling(local_window).std()
    valid = roll_mean.notna() & roll_std.notna()
    level_variability_corr = float(np.corrcoef(roll_mean[valid], roll_std[valid])[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, profile in enumerate(normalized, start=1):
        axes[0].plot(x, profile, label=f"Day {i}")
    axes[0].axhline(1, color="gray", linestyle="--", linewidth=0.9)
    axes[0].set(title="Normalized daily demand profiles", xlabel=xlabel,
                ylabel="Demand / daily mean")
    axes[0].legend()
    axes[0].grid(alpha=0.15)
    axes[1].scatter(roll_mean[valid], roll_std[valid], s=4, alpha=0.25)
    axes[1].set(title="Workload level vs variability", xlabel=f"Rolling mean ({local_window} minutes)",
                ylabel=f"Rolling standard deviation ({local_window} minutes)")
    axes[1].grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)

    return {
        "daily_profile_correlations": np.round(correlations, 6).tolist(),
        "daily_mean_generated_tokens": [float(d.mean()) for d in days],
        "rolling_mean_std_correlation": level_variability_corr,
        "rolling_window_minutes": local_window,
    }


def plot_comparisons(forecasts: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    colors = {"actual": "#1f77b4", "naive": "#ff7f0e",
              "daily_seasonal_naive": "#2ca02c", "selected_hw": "#d62728"}
    labels = {"actual": "Actual", "naive": "Naïve",
              "daily_seasonal_naive": "Daily seasonal naïve",
              "selected_hw": "Selected Holt-Winters"}
    for ax, (name, subset) in zip(axes, forecasts.groupby("evaluation", sort=False)):
        subset = subset.reset_index(drop=True)
        for key in colors:
            ax.plot(subset.index, subset[key], color=colors[key],
                    label=labels[key], linewidth=0.85 if key == "actual" else 1.0,
                    alpha=0.9)
        ax.set(title=f"{name.title()} — 24-hour forecast from one origin",
               ylabel="Generated tokens/minute")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=4, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Minutes after forecast origin")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = arguments()
    period = args.seasonal_period
    if period < 24:
        raise ValueError("--seasonal-period must be at least 24 for these daily diagnostics.")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(resolve_path(args.input_file), period)
    y = df[TARGET].to_numpy(dtype=float)

    # Four days train / one day test; last train day is validation.
    fit, validation = y[:3 * period], y[3 * period:4 * period]
    train, test = y[:4 * period], y[4 * period:]
    diagnostic_info = diagnostics(train, period, output_dir / "diagnostics.png")

    records = []
    forecasts_by_spec = {}
    specs = {}
    for model_name, trend, seasonal, damped in STRUCTURES:
        for setting, alpha, beta, gamma in SMOOTHING_SETTINGS:
            spec = {
                "model": model_name, "setting": setting, "trend": trend,
                "seasonal": seasonal, "damped": damped,
                "alpha": alpha, "beta": beta if trend else None, "gamma": gamma,
                "phi": 0.98 if damped else None,
            }
            key = (model_name, setting)
            specs[key] = spec
            try:
                forecast = fit_and_forecast(fit, period, spec, period)
                forecasts_by_spec[key] = forecast
                row = {**spec, "status": "ok", **accuracy(validation, forecast), "error": ""}
            except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
                row = {**spec, "status": "failed", "MAE": np.nan, "RMSE": np.nan,
                       "sMAPE": np.nan, "error": str(exc)}
            records.append(row)
            print(f"{model_name:30} {setting:6} {row['status']:6} RMSE={row['RMSE']:.1f}")

    candidate_results = pd.DataFrame(records).sort_values("RMSE", na_position="last")
    candidate_results.to_csv(output_dir / "candidate_validation.csv", index=False)
    successful = candidate_results[candidate_results["status"] == "ok"]
    if successful.empty:
        raise RuntimeError("All candidates failed. See candidate_validation.csv.")
    best_row = successful.iloc[0]
    best_spec = specs[(str(best_row["model"]), str(best_row["setting"]))]
    hw_validation = forecasts_by_spec[(best_spec["model"], best_spec["setting"])]
    hw_test = fit_and_forecast(train, period, best_spec, period)

    predicted_by_split = {
        "validation": {
            "actual": validation,
            "naive": np.full(period, fit[-1]),
            "daily_seasonal_naive": fit[-period:].copy(),
            "selected_hw": hw_validation,
            "minute": df["minute"].iloc[3 * period:4 * period].to_numpy(),
        },
        "test": {
            "actual": test,
            "naive": np.full(period, train[-1]),
            "daily_seasonal_naive": train[-period:].copy(),
            "selected_hw": hw_test,
            "minute": df["minute"].iloc[4 * period:5 * period].to_numpy(),
        },
    }
    metrics_rows = []
    forecast_frames = []
    for evaluation, values in predicted_by_split.items():
        for col, display in [
            ("naive", "Naive"),
            ("daily_seasonal_naive", "Daily seasonal naive"),
            ("selected_hw", "Selected Holt-Winters"),
        ]:
            metrics_rows.append({"evaluation": evaluation, "model": display,
                                 **accuracy(values["actual"], values[col])})
        forecast_frames.append(pd.DataFrame({"evaluation": evaluation, **values}))
    all_metrics = pd.DataFrame(metrics_rows)
    all_forecasts = pd.concat(forecast_frames, ignore_index=True)
    all_metrics.to_csv(output_dir / "metrics.csv", index=False)
    all_forecasts.to_csv(output_dir / "forecasts.csv", index=False)
    plot_comparisons(all_forecasts, output_dir / "forecast_comparison.png")

    def rmse(split: str, model: str) -> float:
        return float(all_metrics.loc[
            (all_metrics["evaluation"] == split) & (all_metrics["model"] == model),
            "RMSE"
        ].iloc[0])

    summary = {
        "dataset": "Azure LLM Inference Trace 2024, one-minute aggregation",
        "forecast_design": "one forecast origin; 1440-step validation and test forecasts",
        "seasonal_period": period,
        "fit_rows": len(fit),
        "validation_rows": len(validation),
        "final_refit_rows": len(train),
        "held_out_test_rows": len(test),
        "selection_criterion": "validation RMSE, fixed parameter grid",
        "selected_holt_winters": best_spec,
        "test_rmse_improvement_vs_seasonal_naive_percent": 100 * (
            1 - rmse("test", "Selected Holt-Winters") /
            rmse("test", "Daily seasonal naive")
        ),
        "validation_rmse_improvement_vs_seasonal_naive_percent": 100 * (
            1 - rmse("validation", "Selected Holt-Winters") /
            rmse("validation", "Daily seasonal naive")
        ),
        "diagnostics": diagnostic_info,
        "cautions": [
            "Model configuration selected using Day 4 only; Day 5 used only for evaluation.",
            "Smoothing coefficients were fixed, not numerically optimized.",
            "Residuals and trend/seasonality are estimated from a short four-day training span.",
            "These are one-origin day-ahead forecasts, not rolling 15/30/60-minute forecasts.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\nSelected model:", best_spec["model"], f"({best_spec['setting']})")
    print("\nValidation and held-out test metrics:\n", all_metrics.to_string(index=False))
    print("\nOutputs saved to:", output_dir)


if __name__ == "__main__":
    main()
