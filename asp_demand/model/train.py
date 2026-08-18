"""Train LightGBM regressors for hourly and daily request-count demand."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from asp_demand.config import (
    QUANTILES,
    features_path,
    metrics_path,
    model_path,
    parquet_path,
)
from asp_demand.features.build import TARGET, build_features
from asp_demand.features.calendar import load_calendar

# Base params; objective/metric/alpha are set per quantile in train_model.
PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def train_model(
    granularity: str,
    run_dir: str | Path,
    valid_fraction: float = 0.2,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, float]:
    """Train P50/P90/P95 quantile models for ``granularity`` in ``run_dir``; return metrics.

    One LightGBM quantile-regression model is fit per quantile (``QUANTILES``); the saved
    metrics are computed on the median (P50) over the time-ordered validation split.
    """
    rd = Path(run_dir)
    frame = pd.read_parquet(parquet_path(rd, granularity))
    calendar = load_calendar()
    df, features = build_features(granularity, frame, calendar)

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    if len(df) < 4:
        raise ValueError(
            f"not enough {granularity} rows to train ({len(df)}); ingest more days of logs"
        )

    split = max(1, int(len(df) * (1 - valid_fraction)))
    train_df, valid_df = df.iloc[:split], df.iloc[split:]
    if valid_df.empty:
        valid_df = train_df

    train_set = lgb.Dataset(train_df[features], label=train_df[TARGET])
    valid_set = lgb.Dataset(valid_df[features], label=valid_df[TARGET], reference=train_set)
    base = dict(PARAMS if params is None else params)

    rd.mkdir(parents=True, exist_ok=True)
    total = len(QUANTILES)
    if progress_cb:
        progress_cb(0, total)  # one model per quantile (P50/P90/P95)
    median_pred: np.ndarray | None = None
    for done, (label, alpha) in enumerate(QUANTILES.items(), start=1):
        q_params = {**base, "objective": "quantile", "alpha": alpha, "metric": "quantile"}
        booster = lgb.train(
            q_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        booster.save_model(str(model_path(rd, granularity, label)))
        if label == "p50":
            median_pred = np.asarray(booster.predict(valid_df[features]), dtype=float)
        if progress_cb:
            progress_cb(done, total)

    assert median_pred is not None  # QUANTILES always contains p50
    metrics = _metrics(valid_df[TARGET].to_numpy(dtype=float), median_pred)
    features_path(rd, granularity).write_text(json.dumps(features, indent=2))
    metrics_path(rd, granularity).write_text(json.dumps(metrics, indent=2))
    return metrics
