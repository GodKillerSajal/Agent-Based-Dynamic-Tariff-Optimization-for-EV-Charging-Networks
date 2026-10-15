"""Leakage-aware, scenario-based EV tariff evaluation orchestrated with LangGraph."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)


class EvState(TypedDict, total=False):
    data_dir: str
    output_dir: str
    timestamps: pd.DatetimeIndex
    volume: pd.DataFrame
    occupancy: pd.DataFrame
    price: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    train_end: int
    test_start: int
    predictions: np.ndarray
    baseline_predictions: np.ndarray
    network_congestion_threshold: float
    row_ids: np.ndarray
    trained_model: Any
    congestion_thresholds: pd.Series
    scenario: pd.DataFrame
    high_congestion_actual: pd.DataFrame
    high_congestion_predicted: pd.DataFrame
    report: dict[str, Any]


def _load_data(state: EvState) -> dict[str, Any]:
    root = Path(state["data_dir"])
    clock = pd.read_csv(root / "time.csv")
    timestamps = pd.to_datetime(
        clock[["year", "month", "day", "hour", "minute", "second"]]
    )
    tables: dict[str, pd.DataFrame] = {}
    for name in ("volume", "occupancy", "price"):
        frame = pd.read_csv(root / f"{name}.csv")
        if len(frame) != len(clock):
            raise ValueError(f"{name}.csv has {len(frame)} rows; time.csv has {len(clock)}")
        tables[name] = frame.drop(columns=["timestamp"], errors="ignore")
    station_ids = [c for c in tables["volume"].columns if c in tables["occupancy"].columns]
    station_ids = [c for c in station_ids if c in tables["price"].columns]
    if not station_ids:
        raise ValueError("No station columns are shared by volume, occupancy, and price")
    return {
        "timestamps": pd.DatetimeIndex(timestamps),
        "volume": tables["volume"][station_ids].apply(pd.to_numeric, errors="coerce"),
        "occupancy": tables["occupancy"][station_ids].apply(pd.to_numeric, errors="coerce"),
        "price": tables["price"][station_ids].apply(pd.to_numeric, errors="coerce"),
    }


def _make_features(state: EvState) -> dict[str, Any]:
    timestamps: pd.DatetimeIndex = state["timestamps"]
    total_volume = state["volume"].sum(axis=1, min_count=1).astype(float)
    # Every feature is shifted so the target at t can use information available by t-1.
    x = pd.DataFrame(index=np.arange(len(total_volume)))
    for lag in (1, 12, 288):
        x[f"lag_{lag}"] = total_volume.shift(lag)
    x["rolling_mean_12"] = total_volume.shift(1).rolling(12, min_periods=12).mean()
    x["rolling_mean_288"] = total_volume.shift(1).rolling(288, min_periods=288).mean()
    minute_of_day = timestamps.hour * 60 + timestamps.minute
    x["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    x["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    x["weekday_sin"] = np.sin(2 * np.pi * timestamps.dayofweek / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * timestamps.dayofweek / 7)
    valid = x.notna().all(axis=1) & total_volume.notna()
    x = x.loc[valid].reset_index(drop=True)
    y = total_volume.loc[valid].reset_index(drop=True)
    row_ids = np.flatnonzero(valid.to_numpy())
    split = int(len(x) * 0.70)
    test_start = int(len(x) * 0.85)
    if test_start <= split or len(x) - test_start < 100:
        raise ValueError("Not enough valid time points for chronological train/validation/test")
    return {
        "features": x,
        "target": y,
        "row_ids": row_ids,
        "train_end": split,
        "test_start": test_start,
    }


def _fit_and_forecast(state: EvState) -> dict[str, Any]:
    x: pd.DataFrame = state["features"]
    y: pd.Series = state["target"]
    split, test_start = state["train_end"], state["test_start"]
    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.06,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(x.iloc[:split], y.iloc[:split])
    predictions = np.maximum(0.0, model.predict(x.iloc[test_start:]))
    baseline = np.maximum(0.0, x.iloc[test_start:]["lag_288"].to_numpy())
    threshold = float(y.iloc[:split].quantile(0.90))
    return {
        "predictions": predictions,
        "baseline_predictions": baseline,
        "network_congestion_threshold": threshold,
        "trained_model": model,
    }


def _derive_congestion_thresholds(state: EvState) -> dict[str, Any]:
    # Fit station thresholds only on the initial 70% of the timeline.
    valid_rows = np.asarray(state["row_ids"], dtype=int)
    train_rows = int(valid_rows[state["train_end"] - 1]) + 1
    thresholds = state["occupancy"].iloc[:train_rows].quantile(0.90).replace(0, np.nan)
    return {"congestion_thresholds": thresholds.fillna(state["occupancy"].iloc[:train_rows].median())}


def _simulate_pricing(state: EvState) -> dict[str, Any]:
    row_ids = np.asarray(state.get("row_ids", []), dtype=int)
    test_row_ids = row_ids[state["test_start"] :]
    thresholds: pd.Series = state["congestion_thresholds"]
    observed_previous = state["occupancy"].iloc[test_row_ids - 1].reset_index(drop=True)
    actual_volume = state["volume"].iloc[test_row_ids].reset_index(drop=True)
    base_price = state["price"].iloc[test_row_ids].reset_index(drop=True)
    future_occupancy = state["occupancy"].iloc[test_row_ids].reset_index(drop=True)

    # The action uses only the previous 5-minute occupancy snapshot. Risk is a
    # heuristic because the source does not contain station capacities or queue labels.
    threshold_frame = pd.DataFrame([thresholds.to_dict()])
    threshold_frame = pd.concat([threshold_frame] * len(observed_previous), ignore_index=True)
    network_high = np.asarray(state["predictions"]) >= float(state["network_congestion_threshold"])
    network_signal = pd.DataFrame(
        np.repeat(network_high[:, None], observed_previous.shape[1], axis=1),
        columns=observed_previous.columns,
    )
    high = observed_previous.ge(threshold_frame * 1.05) & network_signal
    moderate = observed_previous.ge(threshold_frame * 0.90) & network_signal & ~high
    multiplier = pd.DataFrame(1.0, index=actual_volume.index, columns=actual_volume.columns)
    multiplier = multiplier.mask(moderate, 1.03).mask(high, 1.07)

    high_actual = future_occupancy.ge(threshold_frame)
    scenario_rows: list[dict[str, float]] = []
    for elasticity in (-0.2, -0.5, -0.8):
        demand_factor = (1.0 + elasticity * (multiplier - 1.0)).clip(lower=0.0)
        scenario_volume = actual_volume * demand_factor
        baseline_revenue = (actual_volume * base_price).sum().sum()
        scenario_revenue = (scenario_volume * base_price * multiplier).sum().sum()
        peak_baseline = actual_volume.where(high_actual).sum(axis=1, min_count=1)
        peak_scenario = scenario_volume.where(high_actual).sum(axis=1, min_count=1)
        baseline_peak = float(peak_baseline.quantile(0.95))
        scenario_peak = float(peak_scenario.quantile(0.95))
        scenario_rows.append(
            {
                "elasticity_assumption": elasticity,
                "simulated_revenue_change_pct": float(
                    (scenario_revenue / baseline_revenue - 1.0) * 100.0
                ),
                "simulated_p95_high_congestion_volume_change_pct": float(
                    (scenario_peak / baseline_peak - 1.0) * 100.0
                ),
            }
        )
    scenario = pd.DataFrame(scenario_rows)
    scenario["interpretation"] = "counterfactual scenario; elasticity is assumed, not observed"
    return {"scenario": scenario, "high_congestion_actual": high_actual, "high_congestion_predicted": high}


def _summarize(state: EvState) -> dict[str, Any]:
    row_ids = np.asarray(state.get("row_ids", []), dtype=int)
    target = state["target"].to_numpy()[state["test_start"] :]
    pred = np.asarray(state["predictions"])
    naive = np.asarray(state["baseline_predictions"])
    epsilon = 1e-6
    report: dict[str, Any] = {
        "dataset": "UrbanEV Shenzhen, 5-minute station-level matrices",
        "total_timestamps": int(len(state["timestamps"])),
        "station_count": int(state["volume"].shape[1]),
        "valid_forecast_samples": int(len(state["features"])),
        "train_samples": int(state["train_end"]),
        "validation_samples": int(state["test_start"] - state["train_end"]),
        "test_samples": int(len(target)),
        "forecast": {
            "model": "HistGradientBoostingRegressor",
            "split": "chronological 70/15/15; all features shifted to avoid look-ahead",
            "mae": float(mean_absolute_error(target, pred)),
            "rmse": float(np.sqrt(mean_squared_error(target, pred))),
            "smape_pct": float(
                np.mean(2 * np.abs(target - pred) / (np.abs(target) + np.abs(pred) + epsilon)) * 100
            ),
            "seasonal_naive_lag_288_mae": float(mean_absolute_error(target, naive)),
            "seasonal_naive_lag_288_rmse": float(np.sqrt(mean_squared_error(target, naive))),
        },
        "congestion_classifier": {
            "definition": "next-step station occupancy at/above train-only station p90; action uses prior station occupancy plus predicted network demand",
        },
        "pricing_scenarios": state["scenario"].to_dict(orient="records"),
        "limitations": [
            "No station-capacity or observed queue labels are supplied; congestion is a train-percentile proxy.",
            "Revenue and load changes are counterfactual simulations under assumed constant elasticities, not causal estimates.",
            "The benchmark contains one UrbanEV observation window; reported forecast scores are held-out within that window.",
        ],
    }
    if "high_congestion_actual" in state:
        y_true = np.asarray(state["high_congestion_actual"], dtype=bool).ravel()
        y_pred = np.asarray(state["high_congestion_predicted"], dtype=bool).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        report["congestion_classifier"].update(
            {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
        )
    return {"report": report}


def build_graph():
    graph = StateGraph(EvState)
    graph.add_node("load_and_validate", _load_data)
    graph.add_node("build_lagged_features", _make_features)
    graph.add_node("forecast_demand", _fit_and_forecast)
    graph.add_node("learn_congestion_thresholds", _derive_congestion_thresholds)
    graph.add_node("simulate_tariff_policy", _simulate_pricing)
    graph.add_node("evaluate_and_report", _summarize)
    graph.add_edge(START, "load_and_validate")
    graph.add_edge("load_and_validate", "build_lagged_features")
    graph.add_edge("build_lagged_features", "forecast_demand")
    graph.add_edge("forecast_demand", "learn_congestion_thresholds")
    graph.add_edge("learn_congestion_thresholds", "simulate_tariff_policy")
    graph.add_edge("simulate_tariff_policy", "evaluate_and_report")
    graph.add_edge("evaluate_and_report", END)
    return graph.compile()


def run(data_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    state = build_graph().invoke({"data_dir": str(data_dir), "output_dir": str(output)})
    report = state["report"]
    report["pipeline_runtime_seconds"] = float(time.perf_counter() - started)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    state["scenario"].to_csv(output / "tariff_scenarios.csv", index=False)
    import joblib

    joblib.dump(state["trained_model"], output / "demand_model.joblib")

    # Optional local experiment tracking; evaluation remains usable without MLflow.
    try:
        import mlflow

        tracking = f"sqlite:///{(output / 'mlflow.db').resolve().as_posix()}"
        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment("ev-tariff-langgraph")
        with mlflow.start_run(run_name="chronological-holdout"):
            mlflow.log_params({"split": "70/15/15 chronological", "lookback_5min": 288})
            for key, value in report["forecast"].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"forecast.{key}", float(value))
            for row in report["pricing_scenarios"]:
                prefix = f"elasticity_{str(row['elasticity_assumption']).replace('-', 'minus').replace('.', '_')}"
                mlflow.log_metric(f"{prefix}.revenue_change_pct", row["simulated_revenue_change_pct"])
                mlflow.log_metric(
                    f"{prefix}.p95_high_congestion_volume_change_pct",
                    row["simulated_p95_high_congestion_volume_change_pct"],
                )
            # The pickle/joblib artifact is created locally above. Log it as an
            # opaque run artifact; MLflow's sklearn flavor rejects this model's
            # internal predictor type under its safe-load policy.
            mlflow.log_artifact(str(output / "demand_model.joblib"))
            mlflow.log_artifact(str(output / "evaluation.json"))
    except ImportError:
        report["mlflow_tracking"] = "not installed; evaluation outputs were still written"
    return report

