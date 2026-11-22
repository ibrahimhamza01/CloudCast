"""CloudCast feature selection, VIF/SVD/PCA diagnostics, and rolling-origin validation.

Standalone; never reads the held-out fifth-day target into model development.

    python scripts/06_feature_selection.py

Workflow: Days 1-3 for fitting / screening, Day 4 for validation, Day 5
untouched for the later regression script. Forecast issued after minute t has
observations at t and earlier; the response is Y[t+h] for h in {1,15,30,60}.
Validation origins advance through Day 4, using only already-observed data.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
TARGET = "total_generated_tokens"
TIME = "minute"
SEASON = 1440
HORIZONS = (1, 15, 30, 60)
RAW_COLS = [
    "code_request_count", "conversation_request_count",
    "code_context_tokens", "conversation_context_tokens",
    "code_generated_tokens", "conversation_generated_tokens",
    "total_request_count", "total_context_tokens",
    "total_generated_tokens", "total_tokens",
]
DOMAIN_FEATURES = [
    "y_now", "y_lag_15", "y_lag_60", "y_lag_1440",
    "code_request_count_now", "conversation_request_count_now",
    "target_tod_sin", "target_tod_cos",
]


def options():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-file", default=str(ROOT / "data/processed/azure_llm_one_minute.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/feature_selection"))
    ap.add_argument("--train-fraction", type=float, default=0.8)
    ap.add_argument("--validation-minutes", type=int, default=1440)
    ap.add_argument("--vif-limit", type=float, default=10.0)
    ap.add_argument("--selection-tolerance", type=float, default=0.01,
                    help="Prefer fewer features if RMSE is within this fraction of the best.")
    return ap.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_data(path):
    df = pd.read_csv(path)
    missing = set([TIME] + RAW_COLS) - set(df)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df[TIME] = pd.to_datetime(df[TIME], utc=True, errors="coerce")
    if df[TIME].isna().any() or df[TIME].duplicated().any():
        raise ValueError("Invalid or duplicate time stamps")
    df = df.sort_values(TIME).reset_index(drop=True)
    if not df[TIME].diff().dropna().eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("Expected uninterrupted one-minute sampling")
    df[RAW_COLS] = df[RAW_COLS].apply(pd.to_numeric, errors="raise")
    if df[RAW_COLS].isna().any().any() or (df[RAW_COLS] < 0).any().any():
        raise ValueError("Missing or negative workload observations")
    return df


def raw_identities(data):
    comparisons = [
        ("total_request_count", data.code_request_count + data.conversation_request_count),
        ("total_context_tokens", data.code_context_tokens + data.conversation_context_tokens),
        ("total_generated_tokens", data.code_generated_tokens + data.conversation_generated_tokens),
        ("total_tokens", data.total_context_tokens + data.total_generated_tokens),
    ]
    return pd.DataFrame([
        {"identity_checked": col, "max_absolute_difference": float((data[col] - calc).abs().max())}
        for col, calc in comparisons
    ])


def raw_svd(data):
    scaled = StandardScaler().fit_transform(data[RAW_COLS].to_numpy(dtype=float))
    singular = np.linalg.svd(scaled, compute_uv=False)
    tol = singular[0] * max(scaled.shape) * np.finfo(float).eps
    rank = int((singular > tol).sum())
    frame = pd.DataFrame({"index": np.arange(1, len(singular) + 1), "singular_value": singular})
    return frame, rank


def make_supervised(train, h):
    result = pd.DataFrame(index=train.index)
    result["target_index"] = np.arange(len(train)) + h
    result["target_y"] = train[TARGET].shift(-h)
    result["y_now"] = train[TARGET]
    for lag in (1, 5, 15, 60, 1440):
        result[f"y_lag_{lag}"] = train[TARGET].shift(lag)
    for col in ("code_request_count", "conversation_request_count",
                "code_context_tokens", "conversation_context_tokens"):
        result[f"{col}_now"] = train[col]
    future_time = train[TIME].shift(-h)
    minutes = future_time.dt.hour * 60 + future_time.dt.minute
    result["target_tod_sin"] = np.sin(2 * np.pi * minutes / SEASON)
    result["target_tod_cos"] = np.cos(2 * np.pi * minutes / SEASON)
    result["seasonal_naive"] = train[TARGET].shift(SEASON - h)
    return result.dropna().copy()


def metrics(actual, pred):
    y = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite model predictions")
    err = y - p
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "sMAPE": float(np.mean(200 * np.abs(err) / np.maximum(np.abs(y) + np.abs(p), 1e-8)))}


def calc_vif(x):
    scaled = StandardScaler().fit_transform(x)
    z = sm.add_constant(scaled, has_constant="add")
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, name in enumerate(x.columns):
            try:
                vif = float(variance_inflation_factor(z, i + 1))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                vif = float("inf")
            if np.isnan(vif):
                vif = float("inf")
            rows.append({"feature": name, "VIF": vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def vif_prune(x, limit):
    active = list(x.columns)
    steps = []
    while len(active) > 2:
        v = calc_vif(x[active])
        worst = v.iloc[0]
        if float(worst.VIF) <= limit:
            break
        steps.append({"step": len(steps) + 1, "removed": str(worst.feature),
                      "VIF_before_removal": float(worst.VIF), "remaining_features": len(active)-1})
        active.remove(str(worst.feature))
    return active, steps, calc_vif(x[active])


def backward_prune(x, y, threshold=0.05):
    active = list(x.columns)
    z = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
    steps = []
    while len(active) > 1:
        reg = sm.OLS(y, sm.add_constant(z[active], has_constant="add")).fit()
        pv = reg.pvalues.drop("const", errors="ignore")
        if pv.empty:
            break
        worst = str(pv.idxmax())
        value = float(pv.max())
        if np.isfinite(value) and value <= threshold:
            break
        steps.append({"step": len(steps) + 1, "removed": worst,
                      "p_value": value, "remaining_features": len(active)-1})
        active.remove(worst)
    return active, steps


def plot_results(out, metric_frame, pca_frame, singular_values):
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, g in metric_frame.groupby("model"):
        ax.plot(g.horizon, g.RMSE, marker="o", label=model)
    ax.set(title="Feature-selection validation: RMSE by horizon",
           xlabel="Forecast horizon (minutes)", ylabel="Validation RMSE (tokens/minute)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "validation_rmse.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pca_frame.component, pca_frame.cumulative_variance * 100, marker="o")
    ax.axhline(95, linestyle="--", color="gray", label="95% variance")
    ax.set(title="PCA on standardized candidate predictors (fit only)",
           xlabel="Number of principal components", ylabel="Cumulative explained variance (%)",
           xticks=range(1, len(pca_frame) + 1))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pca_variance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(singular_values["index"], np.maximum(singular_values.singular_value, 1e-15), marker="o")
    ax.set(title="SVD: original ten workload columns (fit only)",
           xlabel="Singular-value index", ylabel="Singular value (log scale)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "raw_singular_values.png", dpi=180)
    plt.close(fig)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def main():
    args = options()
    if not 0 < args.train_fraction < 1 or args.vif_limit <= 1 or args.selection_tolerance < 0:
        raise ValueError("Check --train-fraction, --vif-limit, and --selection-tolerance")
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(resolve(args.input_file))
    train_end = int(len(data) * args.train_fraction)
    train = data.iloc[:train_end].copy().reset_index(drop=True)
    validation_start = len(train) - args.validation_minutes
    if args.validation_minutes < max(HORIZONS) or validation_start <= SEASON + max(HORIZONS):
        raise ValueError("Not enough fitting or validation history")

    # All diagnostics and model choices are restricted to fitting period only.
    raw_fit = train.iloc[:validation_start]
    identities = raw_identities(raw_fit)
    identities.to_csv(out / "raw_identities.csv", index=False)
    svalues, rank = raw_svd(raw_fit)
    svalues.to_csv(out / "raw_singular_values.csv", index=False)

    validation, selected_rows, backward_rows, vif_rows = [], [], [], []
    vif_initial_frames, vif_final_frames, corr_rows = [], [], []
    all_model_choices = {}
    pca_summary = None
    for h in HORIZONS:
        sup = make_supervised(train, h)
        fit = sup[sup.target_index < validation_start]
        val = sup[sup.target_index >= validation_start]
        columns = [name for name in sup.columns if name not in ("target_y", "target_index", "seasonal_naive")]
        fit_x, val_x = fit[columns], val[columns]
        fit_y, val_y = fit.target_y, val.target_y
        domain = [name for name in DOMAIN_FEATURES if name in columns]
        backward, steps_backward = backward_prune(fit_x, fit_y)
        reduced, steps_vif, final_vif = vif_prune(fit_x, args.vif_limit)
        for step in steps_backward:
            backward_rows.append({"horizon": h, **step})
        for step in steps_vif:
            vif_rows.append({"horizon": h, **step})
        v0 = calc_vif(fit_x)
        v0.insert(0, "horizon", h)
        vif_initial_frames.append(v0)
        vf = final_vif.copy()
        vf.insert(0, "horizon", h)
        vif_final_frames.append(vf)
        for name in columns:
            corr_rows.append({"horizon": h, "feature": name,
                              "correlation": float(fit_x[name].corr(fit_y))})

        if h == 1:
            z = StandardScaler().fit_transform(fit_x)
            pca = PCA().fit(z)
            pca_summary = pd.DataFrame({"component": np.arange(1, len(columns)+1),
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)})
            pca_summary.to_csv(out / "pca_explained_variance.csv", index=False)
            pd.DataFrame(pca.components_.T, index=columns,
                columns=[f"PC_{i+1}" for i in range(len(columns))]).to_csv(out / "pca_loadings.csv")

        for name, forecast in (
            ("Persistence", val_x.y_now.to_numpy()),
            ("Daily seasonal naive", val.seasonal_naive.to_numpy()),
        ):
            validation.append({"horizon": h, "model": name, "n_features": 0,
                               "n_fit": len(fit), "n_validation": len(val), **metrics(val_y, forecast)})

        forecasts = {}
        for name, fields in (
            ("OLS full", columns), ("OLS backward", backward),
            ("OLS VIF-reduced", reduced), ("OLS domain subset", domain),
        ):
            reg = LinearRegression().fit(fit_x[fields], fit_y)
            pred = reg.predict(val_x[fields])
            validation.append({"horizon": h, "model": name, "n_features": len(fields),
                               "n_fit": len(fit), "n_validation": len(val), **metrics(val_y, pred)})
            forecasts[name] = {"features": list(fields), "n_features": len(fields)}

        pipeline = make_pipeline(StandardScaler(), PCA(n_components=0.95), LinearRegression())
        pipeline.fit(fit_x, fit_y)
        pred = pipeline.predict(val_x)
        n_pcs = int(pipeline.named_steps["pca"].n_components_)
        name = "OLS PCA 95%"
        validation.append({"horizon": h, "model": name, "n_features": n_pcs,
                           "n_fit": len(fit), "n_validation": len(val), **metrics(val_y, pred)})
        forecasts[name] = {"features": [f"PC{i+1}" for i in range(n_pcs)],
                           "n_features": n_pcs}

        candidate_rows = [r for r in validation if r["horizon"] == h and r["model"] in forecasts]
        best = min(r["RMSE"] for r in candidate_rows)
        eligible = [r for r in candidate_rows if r["RMSE"] <= best * (1 + args.selection_tolerance)]
        winner = min(eligible, key=lambda r: (r["n_features"], r["RMSE"], r["model"]))
        selected_rows.append({"horizon": h, "selected_model": winner["model"],
                              "validation_RMSE": winner["RMSE"],
                              "n_features": winner["n_features"],
                              "features": "; ".join(forecasts[winner["model"]]["features"])})
        all_model_choices[str(h)] = {"model": winner["model"],
            "validation_RMSE": winner["RMSE"],
            "n_features": winner["n_features"],
            "features": forecasts[winner["model"]]["features"],
            "vif_reduced_features": reduced,
            "backward_features": backward}
        print(f"Horizon {h:2d}: fit={len(fit):,} validation={len(val):,}; "
              f"selected={winner['model']} ({winner['n_features']} features) RMSE={winner['RMSE']:,.1f}")

    all_metrics = pd.DataFrame(validation).sort_values(["horizon", "model"])
    all_metrics.to_csv(out / "validation_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out / "selected_features.csv", index=False)
    pd.DataFrame(backward_rows, columns=["horizon", "step", "removed", "p_value", "remaining_features"]).to_csv(
        out / "backward_elimination.csv", index=False)
    pd.DataFrame(vif_rows, columns=["horizon", "step", "removed", "VIF_before_removal", "remaining_features"]).to_csv(
        out / "vif_elimination.csv", index=False)
    pd.concat(vif_initial_frames, ignore_index=True).to_csv(out / "vif_initial.csv", index=False)
    pd.concat(vif_final_frames, ignore_index=True).to_csv(out / "vif_final.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out / "candidate_correlations.csv", index=False)
    plot_results(out, all_metrics, pca_summary, svalues)

    summary = {
        "design": "Rolling-origin direct forecasts Y[t+h] from features available by minute t",
        "horizons": list(HORIZONS),
        "training_rows": len(train), "untouched_test_rows": len(data) - len(train),
        "fit_target_end_exclusive": validation_start,
        "validation_target_rows": args.validation_minutes,
        "raw_input_column_count": len(RAW_COLS), "raw_standardized_rank": rank,
        "raw_rank_deficient": bool(rank < len(RAW_COLS)),
        "all_raw_identities_exact": bool((identities.max_absolute_difference == 0).all()),
        "initial_1min_max_vif": float(vif_initial_frames[0].VIF.max()),
        "pca_components_95pct_one_minute": int(np.searchsorted(pca_summary.cumulative_variance.to_numpy(), .95) + 1),
        "vif_limit": args.vif_limit,
        "model_selection_rule": f"Validation RMSE within {args.selection_tolerance*100:g}% of minimum; prefer fewer features",
        "selected_by_horizon": all_model_choices,
        "cautions": [
            "Day 5 withheld; never used for feature selection or hyperparameter tuning.",
            "All validation forecasts use observed features up to each rolling forecast origin.",
            "OLS p-values from backward elimination are exploratory with autocorrelated data.",
            "PCA variance explained is not predictive accuracy, and components reduce interpretability.",
            "This selection is for rolling-origin 1/15/30/60-minute forecasting, not full-day one-origin forecasts.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nValidation RMSE:")
    print(all_metrics.pivot(index="horizon", columns="model", values="RMSE").round(1).to_string())
    print("\nOriginal columns rank:", rank, "/", len(RAW_COLS))
    print("Output folder:", out)
    print("Day 5 stays untouched. Use the later regression script for final testing and diagnostics.")


if __name__ == "__main__":
    main()
