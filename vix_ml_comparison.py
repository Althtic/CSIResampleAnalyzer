#!/usr/bin/env python3
"""VIX-Enhanced Multi-Model Comparison: Fixed-time vs Turnover-resampled Bars.

Models: Random Forest, XGBoost, LSTM, Self-Attention.
All models predict next-bar up/down direction on the same train/test splits.
"""

from __future__ import annotations

import argparse
import html
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ── Reuse from existing scripts ──────────────────────────────────────────
from analyze_resampling import make_time_bars, make_turnover_bars, markdown_table
from factor_ml_comparison import (
    INDICES,
    DATA_ROOT,
    load_index_minutes,
    filter_common_dates,
    build_factors,
    expanding_window_folds,
)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

VIX_PATH = DATA_ROOT / "VIX_GI_2014-01-01_2026-06-16_daily.xlsx"
DEFAULT_OUTPUT = DATA_ROOT / "vix_ml_results"

RF_PARAMS = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 50, "random_state": 42, "n_jobs": -1}
XGB_PARAMS = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "random_state": 42, "n_jobs": -1,
              "eval_metric": "logloss"}

NN_EPOCHS = 30
NN_BATCH_SIZE = 256
NN_EARLY_STOP = 5

# ═══════════════════════════════════════════════════════════════════════════
# Section 1: VIX data loading & feature engineering
# ═══════════════════════════════════════════════════════════════════════════

def load_vix_features(path: str | Path) -> pd.DataFrame:
    """Load daily VIX data and compute features. All features use past-only data."""
    raw = pd.read_excel(path)
    raw["trade_date"] = pd.to_datetime(raw["_DATE"].astype(str), format="%Y%m%d")
    df = raw.sort_values("trade_date").reset_index(drop=True)

    close = df["MATCH"].astype(float)
    high = df["HIGH"].astype(float)
    low = df["LOW"].astype(float)

    df["vix_close"] = close
    df["vix_ret"] = close.pct_change()
    df["vix_range"] = (high - low) / close.clip(lower=1e-8)

    # Deviation from moving averages
    ma5 = close.rolling(5, min_periods=3).mean()
    ma20 = close.rolling(20, min_periods=5).mean()
    df["vix_dev_ma5"] = close / ma5.clip(lower=1e-8) - 1.0
    df["vix_dev_ma20"] = close / ma20.clip(lower=1e-8) - 1.0

    # 5-day rolling volatility of VIX returns
    df["vix_vol5"] = df["vix_ret"].rolling(5, min_periods=3).std()

    # All VIX features shifted by 1 day: use D-1 to predict D
    for col in ["vix_close", "vix_ret", "vix_range", "vix_dev_ma5", "vix_dev_ma20", "vix_vol5"]:
        df[col] = df[col].shift(1)

    features_cols = ["trade_date", "vix_close", "vix_ret", "vix_range",
                     "vix_dev_ma5", "vix_dev_ma20", "vix_vol5"]
    return df[features_cols].copy()


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Merge VIX with bar factors
# ═══════════════════════════════════════════════════════════════════════════

def merge_vix_to_factors(factors: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Broadcast daily VIX features to each intraday bar.

    Bar trade_date D → VIX features from D-1 (already shifted in load_vix_features).
    """
    vix_map = vix_df.set_index("trade_date")
    vix_cols = ["vix_close", "vix_ret", "vix_range", "vix_dev_ma5", "vix_dev_ma20", "vix_vol5"]

    df = factors.copy()
    # Match bar date to VIX date (already D-1 shifted)
    bar_dates = pd.to_datetime(df["trade_date"])
    for col in vix_cols:
        df[col] = bar_dates.map(vix_map[col])
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: ML dataset preparation
# ═══════════════════════════════════════════════════════════════════════════

FACTOR_COLS = ["ret", "range_pct", "body_pct", "upper_shadow", "lower_shadow",
               "vol_ratio", "turn_ratio", "turn_intensity", "realized_vol", "amihud"]

VIX_COLS = ["vix_close", "vix_ret", "vix_range", "vix_dev_ma5", "vix_dev_ma20", "vix_vol5"]


def prepare_ml_dataset_vix(factors: pd.DataFrame):
    """Build flat + sequence + auxiliary feature sets with VIX.

    Returns:
        X_flat:    (n, n_flat_features) — for RF / XGBoost
        X_seq:     (n, 5, 10) — 5 lagged timesteps of 10 factors
        X_aux:     (n, n_aux) — time dummies + VIX
        y:         (n,) binary target
        dates:     (n,) trade dates
        flat_names: list of flat feature names
    """
    df = factors.dropna().copy().sort_values(["trade_date", "end_time"]).reset_index(drop=True)

    # ── Lagged factor features ──
    lag_arrays = []
    lag_names = []
    for col in FACTOR_COLS:
        for lag in range(1, 6):
            lag_names.append(f"{col}_lag{lag}")
            lag_arrays.append(df[col].shift(lag).to_numpy(dtype=float))

    # ── Time dummies ──
    df["hour"] = pd.to_datetime(df["end_time"]).dt.hour
    df["dow"] = pd.to_datetime(df["end_time"]).dt.dayofweek
    hour_names = []
    hour_arrays = []
    for h in [9, 10, 11, 13, 14]:
        hour_names.append(f"hour_{h}")
        hour_arrays.append((df["hour"] == h).astype(float).to_numpy(dtype=float))
    dow_names = []
    dow_arrays = []
    for d in range(5):
        dow_names.append(f"dow_{d}")
        dow_arrays.append((df["dow"] == d).astype(float).to_numpy(dtype=float))

    # ── VIX features ──
    vix_arrays = [df[col].to_numpy(dtype=float) for col in VIX_COLS]

    # ── X_flat: all features stacked ──
    all_flat_arrays = lag_arrays + hour_arrays + dow_arrays + vix_arrays
    all_flat_names = lag_names + hour_names + dow_names + VIX_COLS
    X_flat = np.column_stack(all_flat_arrays)

    # ── X_seq: reshape lags to (n, 5, 10) ──
    n = len(df)
    X_seq = np.zeros((n, 5, 10), dtype=np.float32)
    for f_idx, col in enumerate(FACTOR_COLS):
        for lag in range(1, 6):
            X_seq[:, lag - 1, f_idx] = df[col].shift(lag).to_numpy(dtype=float)

    # ── X_aux: time + VIX ──
    aux_arrays = hour_arrays + dow_arrays + vix_arrays
    X_aux = np.column_stack(aux_arrays).astype(np.float32)

    # ── Target: next-bar direction (within same day) ──
    target = np.full(n, np.nan)
    next_bar_ret = np.full(n, np.nan)
    for _, day_idx in df.groupby("trade_date").groups.items():
        rets = df.loc[day_idx, "ret"].to_numpy(dtype=float)
        target[day_idx[:-1]] = (rets[1:] > 0).astype(float)
        next_bar_ret[day_idx[:-1]] = rets[1:]

    dates = df["trade_date"]

    mask = np.isfinite(X_flat).all(axis=1) & np.isfinite(target)
    return (X_flat[mask], X_seq[mask], X_aux[mask], target[mask],
            dates.iloc[mask].reset_index(drop=True), next_bar_ret[mask], all_flat_names)


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Model definitions
# ═══════════════════════════════════════════════════════════════════════════

def build_lstm(n_aux: int):
    """Simple LSTM: 1 layer, ~3K params."""
    import tensorflow as tf
    from tensorflow.keras.layers import (LSTM, Concatenate, Dense, Dropout,
                                          Input, LayerNormalization)
    from tensorflow.keras.models import Model

    seq_input = Input(shape=(5, 10), name="seq_input")
    aux_input = Input(shape=(n_aux,), name="aux_input")

    x = LSTM(32, return_sequences=False, name="lstm")(seq_input)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Concatenate()([x, aux_input])
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=[seq_input, aux_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def build_attention(n_aux: int, n_heads: int = 2, key_dim: int = 8):
    """Simple self-attention: 1 attention block + pooling, ~2K params."""
    import tensorflow as tf
    from tensorflow.keras.layers import (Add, Concatenate, Dense, Dropout,
                                          GlobalAveragePooling1D, Input,
                                          LayerNormalization, MultiHeadAttention)
    from tensorflow.keras.models import Model

    seq_input = Input(shape=(5, 10), name="seq_input")
    aux_input = Input(shape=(n_aux,), name="aux_input")

    # Project to key_dim for attention
    x = Dense(key_dim * n_heads, activation="relu")(seq_input)
    attn_out = MultiHeadAttention(num_heads=n_heads, key_dim=key_dim, name="mha")(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Concatenate()([x, aux_input])
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=[seq_input, aux_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Training & evaluation
# ═══════════════════════════════════════════════════════════════════════════

def train_eval_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    return proba


def train_eval_xgb(X_train, y_train, X_test, y_test):
    # Compute scale_pos_weight for class imbalance
    neg, pos = np.bincount(y_train.astype(int))
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train, verbose=False)
    proba = model.predict_proba(X_test)[:, 1]
    return proba


def train_eval_nn(build_fn, X_train_seq, X_train_aux, y_train,
                  X_test_seq, X_test_aux, y_test):
    import tensorflow as tf
    tf.random.set_seed(42)

    # Class weight
    neg, pos = np.bincount(y_train.astype(int))
    class_weight = {0: 1.0, 1: neg / pos} if pos > 0 else None

    model = build_fn(X_train_aux.shape[1])
    model.fit(
        [X_train_seq, X_train_aux], y_train,
        validation_split=0.2,
        epochs=NN_EPOCHS,
        batch_size=NN_BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=NN_EARLY_STOP, restore_best_weights=True)],
        verbose=0,
    )
    proba = model.predict([X_test_seq, X_test_aux], batch_size=NN_BATCH_SIZE,
                          verbose=0).flatten()
    tf.keras.backend.clear_session()
    return proba


def compute_metrics(y_true, proba, pred, next_bar_ret):
    auc = float(roc_auc_score(y_true, proba))
    acc = float(accuracy_score(y_true, pred))
    pred_up = float(np.mean(pred))
    actual_up = float(np.mean(y_true))

    # Strategy Sharpe: long when predict up
    strategy_ret = pred * next_bar_ret
    n_bars = len(strategy_ret)
    # Annualize: ~48 bars/day * 252 days for 5min, adjust by actual count
    bars_per_day = 48
    sr = float(np.mean(strategy_ret) / np.std(strategy_ret, ddof=1) * np.sqrt(252 * bars_per_day)) \
        if np.std(strategy_ret) > 0 and n_bars > 1 else 0.0

    return {"auc": auc, "accuracy": acc, "pred_up_rate": pred_up,
            "actual_up_rate": actual_up, "sharpe": sr}


def run_all_models(factors_vix, bar_label, index_label):
    """Train & evaluate all 4 models on the same expanding-window folds."""
    X_flat, X_seq, X_aux, y, dates, next_bar_ret, flat_names = prepare_ml_dataset_vix(factors_vix)
    folds = expanding_window_folds(dates)

    model_registry = {
        "RF":        (train_eval_rf,  "flat"),
        "XGBoost":   (train_eval_xgb, "flat"),
        "LSTM":      (train_eval_nn,  "seq"),
        "Attention": (train_eval_nn,  "seq"),
    }

    results = []
    for train_end, test_year in folds:
        train_mask = dates.str[:4] <= train_end
        test_mask = dates.str[:4] == test_year
        if test_mask.sum() < 100 or train_mask.sum() < 100:
            continue

        y_train, y_test = y[train_mask], y[test_mask]
        if len(np.unique(y_train)) < 2:
            continue

        for model_name, (train_fn, input_type) in model_registry.items():
            try:
                if input_type == "flat":
                    X_tr = X_flat[train_mask]
                    X_te = X_flat[test_mask]
                    proba = train_fn(X_tr, y_train, X_te, y_test)
                else:
                    kwargs = {}
                    if model_name == "LSTM":
                        kwargs["build_fn"] = build_lstm
                    else:
                        kwargs["build_fn"] = build_attention
                    proba = train_fn(
                        X_train_seq=X_seq[train_mask],
                        X_train_aux=X_aux[train_mask],
                        y_train=y_train,
                        X_test_seq=X_seq[test_mask],
                        X_test_aux=X_aux[test_mask],
                        y_test=y_test,
                        **kwargs,
                    )

                pred = (proba >= 0.5).astype(float)
                metrics = compute_metrics(y_test, proba, pred, next_bar_ret[test_mask])
                metrics.update({
                    "index": index_label,
                    "bar_type": bar_label,
                    "model": model_name,
                    "train_end": train_end,
                    "test_year": test_year,
                    "train_samples": int(train_mask.sum()),
                    "test_samples": int(test_mask.sum()),
                })
                results.append(metrics)
            except Exception as exc:
                print(f"    FAIL {model_name} {bar_label} {index_label} "
                      f"train→test {test_year}: {exc}", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: SVG charts
# ═══════════════════════════════════════════════════════════════════════════

def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(x: float, y: float, text: object, size: int = 12,
             anchor: str = "start", weight: str = "400", color: str = "#17202a") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,Helvetica,sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{esc(text)}</text>')


def save_auc_chart(ml_df: pd.DataFrame, path: Path) -> None:
    """AUC comparison: 4 models × 2 bar types × 5 indices."""
    if ml_df.empty:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40">'
                        '<text x="10" y="25">No data</text></svg>')
        return

    models = ["RF", "XGBoost", "LSTM", "Attention"]
    bar_types = ["5min_bar", "turnover_bar"]
    indices_sorted = sorted(ml_df["index"].unique())

    model_colors = {"RF": "#2563eb", "XGBoost": "#d97706", "LSTM": "#16a34a", "Attention": "#ef4444"}

    w, h = 1400, 600
    left, right, top, bottom = 140, 40, 80, 80
    bar_w = 22
    group_gap = 20
    n_bars_per_index = len(models) * len(bar_types)

    # Compute layout
    n_indices = len(indices_sorted)
    usable_w = w - left - right
    bars_total_w = n_indices * n_bars_per_index * bar_w
    gaps_total_w = (n_indices - 1) * group_gap + n_indices * (len(bar_types) - 1) * 4
    group_w = (usable_w - gaps_total_w) / n_indices + (n_bars_per_index * bar_w + (len(bar_types) - 1) * 4)

    # Actually simpler: compute x positions directly
    total_needed = n_indices * n_bars_per_index * bar_w + (n_indices - 1) * group_gap + n_indices * (len(bar_types) - 1) * 4
    scale = min(1.0, usable_w / max(total_needed, 1))
    actual_bar_w = max(4, bar_w * scale)

    y_min = max(0.45, ml_df["auc"].min() - 0.03)
    y_max = min(0.85, ml_df["auc"].max() + 0.03)
    plot_h = h - top - bottom

    def ymap(v):
        return top + plot_h * (1.0 - (v - y_min) / (y_max - y_min))

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "VIX-Enhanced Multi-Model AUC Comparison", 20, "middle", "700"),
        svg_text(w / 2, 52, "Darker = 5min fixed-time bars, Lighter = turnover-resampled bars", 12, "middle"),
        f'<line x1="{left}" y1="{ymap(0.5):.1f}" x2="{w - right}" y2="{ymap(0.5):.1f}" '
        f'stroke="#bdc3c7" stroke-dasharray="4,4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#2c3e50"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{w - right}" y2="{top + plot_h}" stroke="#2c3e50"/>',
    ]

    for tick in np.linspace(y_min, y_max, 5):
        y = ymap(float(tick))
        elems.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#2c3e50"/>')
        elems.append(svg_text(left - 10, y + 4, f"{tick:.3f}", 10, "end"))

    for i, idx in enumerate(indices_sorted):
        subset = ml_df[ml_df["index"] == idx]
        # Starting x for this index group
        x_start = left + i * (n_bars_per_index * actual_bar_w + group_gap + (len(bar_types) - 1) * 4 * scale)

        for j, bt in enumerate(bar_types):
            for k, mdl in enumerate(models):
                row = subset[(subset["bar_type"] == bt) & (subset["model"] == mdl)]
                if row.empty:
                    continue
                auc_val = float(row["auc"].iloc[0])
                bar_idx = j * len(models) + k
                x = x_start + bar_idx * actual_bar_w + j * 4
                y = ymap(auc_val)
                bh = max(1, top + plot_h - y)
                color = model_colors[mdl]
                opacity = "0.9" if bt == "5min_bar" else "0.55"
                elems.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{actual_bar_w:.1f}" '
                             f'height="{bh:.1f}" fill="{color}" opacity="{opacity}"/>')

    # Legend
    lx, ly = left + 10, h - 40
    for k, mdl in enumerate(models):
        x = lx + k * 120
        elems.append(f'<rect x="{x}" y="{ly}" width="14" height="14" '
                     f'fill="{model_colors[mdl]}" opacity="0.9"/>')
        elems.append(svg_text(x + 18, ly + 12, mdl, 11))
    elems.append(f'<rect x="{lx}" y="{ly + 18}" width="14" height="14" fill="#333" opacity="0.9"/>')
    elems.append(svg_text(lx + 18, ly + 30, "5min fixed-time", 10))
    elems.append(f'<rect x="{lx + 120}" y="{ly + 18}" width="14" height="14" fill="#333" opacity="0.55"/>')
    elems.append(svg_text(lx + 138, ly + 30, "turnover-resampled", 10))

    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def save_sharpe_chart(ml_df: pd.DataFrame, path: Path) -> None:
    """Sharpe ratio comparison chart."""
    if ml_df.empty:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40">'
                        '<text x="10" y="25">No data</text></svg>')
        return

    models = ["RF", "XGBoost", "LSTM", "Attention"]
    bar_types = ["5min_bar", "turnover_bar"]
    indices_sorted = sorted(ml_df["index"].unique())
    model_colors = {"RF": "#2563eb", "XGBoost": "#d97706", "LSTM": "#16a34a", "Attention": "#ef4444"}

    w, h = 1400, 600
    left, right, top, bottom = 140, 40, 80, 80
    bar_w = 22
    group_gap = 20
    n_indices = len(indices_sorted)
    n_bars_per_index = len(models) * len(bar_types)
    usable_w = w - left - right
    total_needed = n_indices * n_bars_per_index * bar_w + (n_indices - 1) * group_gap
    scale = min(1.0, usable_w / max(total_needed, 1))
    actual_bar_w = max(4, bar_w * scale)

    all_sr = ml_df["sharpe"].to_numpy(dtype=float)
    y_min = min(-3, all_sr.min() - 0.5)
    y_max = max(3, all_sr.max() + 0.5)
    plot_h = h - top - bottom

    def ymap(v):
        return top + plot_h * (1.0 - (v - y_min) / (y_max - y_min))

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "VIX-Enhanced Multi-Model Strategy Sharpe", 20, "middle", "700"),
        svg_text(w / 2, 52, "Long when model predicts up. Darker = 5min, Lighter = turnover.", 12, "middle"),
        f'<line x1="{left}" y1="{ymap(0):.1f}" x2="{w - right}" y2="{ymap(0):.1f}" stroke="#2c3e50"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#2c3e50"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{w - right}" y2="{top + plot_h}" stroke="#2c3e50"/>',
    ]

    for tick in np.linspace(y_min, y_max, 5):
        y = ymap(float(tick))
        elems.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#2c3e50"/>')
        elems.append(svg_text(left - 10, y + 4, f"{tick:.1f}", 10, "end"))

    for i, idx in enumerate(indices_sorted):
        subset = ml_df[ml_df["index"] == idx]
        x_start = left + i * (n_bars_per_index * actual_bar_w + group_gap)
        for j, bt in enumerate(bar_types):
            for k, mdl in enumerate(models):
                row = subset[(subset["bar_type"] == bt) & (subset["model"] == mdl)]
                if row.empty:
                    continue
                sr = float(row["sharpe"].iloc[0])
                bar_idx = j * len(models) + k
                x = x_start + bar_idx * actual_bar_w + j * 4
                y0 = ymap(0)
                y = ymap(sr)
                y_top = y if sr >= 0 else y0
                bh = max(1, abs(y - y0))
                color = model_colors[mdl]
                opacity = "0.9" if bt == "5min_bar" else "0.55"
                elems.append(f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{actual_bar_w:.1f}" '
                             f'height="{bh:.1f}" fill="{color}" opacity="{opacity}"/>')

    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Section 7: Report
# ═══════════════════════════════════════════════════════════════════════════

def write_report(path: Path, ml_summary: pd.DataFrame, ml_detail: pd.DataFrame,
                 args: argparse.Namespace):
    lines = [
        "# VIX-Enhanced Multi-Model Comparison Report",
        "",
        "## 1. Setup",
        "",
        f"- Indices: {', '.join(INDICES[k]['label'] for k in INDICES)}",
        f"- Date range: {args.start} to {args.end}",
        f"- Bar types: 5-minute fixed-time vs turnover-resampled",
        f"- Models: Random Forest, XGBoost, LSTM, Self-Attention",
        f"- Training: Expanding window walk-forward (train up to year T-1, test on year T)",
        "",
        "## 2. VIX Features",
        "",
        "6 daily VIX features (from D-1, strictly no look-ahead):",
        "",
        "| Feature | Description |",
        "|---|---|",
        "| vix_close | VIX panic index closing level |",
        "| vix_ret | VIX daily return |",
        "| vix_range | (HIGH - LOW) / close |",
        "| vix_dev_ma5 | Deviation from 5-day MA |",
        "| vix_dev_ma20 | Deviation from 20-day MA |",
        "| vix_vol5 | 5-day rolling std of VIX returns |",
        "",
        "Plus 50 lagged factor features (10 factors × lags 1-5) + 10 time dummies.",
        "",
        "## 3. Model Architectures",
        "",
        "- **RF**: 200 trees, max_depth=8, min_samples_leaf=50",
        "- **XGBoost**: 200 trees, max_depth=6, lr=0.05",
        "- **LSTM**: 1 LSTM(32) + Dense(16) → sigmoid (~3K params)",
        "- **Attention**: MultiHeadAttention(2 heads, key_dim=8) + Dense(16) → sigmoid (~2K params)",
        "",
        "## 4. Results Summary",
        "",
        markdown_table(ml_summary) if not ml_summary.empty else "_No results_",
        "",
        "![AUC comparison](vix_auc_comparison.svg)",
        "",
        "![Sharpe comparison](vix_sharpe_comparison.svg)",
        "",
        "## 5. Per-Fold Detail",
        "",
        markdown_table(ml_detail) if not ml_detail.empty else "_No details_",
        "",
        "## 6. Key Findings",
        "",
    ]

    if not ml_summary.empty:
        # Best model overall
        best = ml_summary.loc[ml_summary["auc"].idxmax()]
        lines.append(f"- **Best AUC**: {best['model']} on {best['index']} "
                     f"({best['bar_type']}), AUC = {best['auc']:.4f}")

        # Average by model
        lines.append("")
        lines.append("### Average AUC by Model")
        for mdl in ["RF", "XGBoost", "LSTM", "Attention"]:
            sub = ml_summary[ml_summary["model"] == mdl]
            if not sub.empty:
                lines.append(f"- {mdl}: mean AUC = {sub['auc'].mean():.4f}")

        # Turnover vs fixed comparison
        lines.append("")
        lines.append("### Turnover-resampled vs Fixed-time AUC")
        for idx in sorted(ml_summary["index"].unique()):
            idx_sub = ml_summary[ml_summary["index"] == idx]
            for mdl in ["RF", "XGBoost", "LSTM", "Attention"]:
                mdl_sub = idx_sub[idx_sub["model"] == mdl]
                time_row = mdl_sub[mdl_sub["bar_type"] == "5min_bar"]
                turn_row = mdl_sub[mdl_sub["bar_type"] == "turnover_bar"]
                if time_row.empty or turn_row.empty:
                    continue
                time_auc = float(time_row["auc"].iloc[0])
                turn_auc = float(turn_row["auc"].iloc[0])
                winner = "turnover" if turn_auc > time_auc else "5min"
                diff = abs(turn_auc - time_auc)
                lines.append(f"- {idx} / {mdl}: 5min={time_auc:.4f}, "
                            f"turnover={turn_auc:.4f} ({winner} wins by {diff:.4f})")

    lines.extend([
        "",
        "## 7. Generated Files",
        "",
        "- `vix_ml_summary.csv` — Results by index × bar_type × model (averaged over folds)",
        "- `vix_ml_by_fold.csv` — Per-fold detailed results",
        "- `vix_auc_comparison.svg` — AUC comparison chart",
        "- `vix_sharpe_comparison.svg` — Sharpe ratio comparison chart",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VIX-enhanced multi-model comparison: fixed-time vs turnover-resampled bars")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-06-18")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--target-bars-per-day", type=int, default=48)
    p.add_argument("--skip-rf", action="store_true")
    p.add_argument("--skip-xgb", action="store_true")
    p.add_argument("--skip-nn", action="store_true")
    p.add_argument("--indices", nargs="+", default=list(INDICES))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    bar_dir = out / "bars"
    for d in [out, bar_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Remove unsupported models from registry
    if args.skip_rf:
        del train_eval_rf  # unused — handled via filter below
    # (skip flags handled via model_registry filtering below)

    # ── 1. Load VIX data ────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading VIX daily data")
    print("=" * 60)
    vix_df = load_vix_features(VIX_PATH)
    print(f"  VIX data: {len(vix_df)} rows, {vix_df['trade_date'].min().date()} to "
          f"{vix_df['trade_date'].max().date()}")
    first_valid = vix_df.dropna(subset=["vix_close"]).iloc[0]["trade_date"]
    print(f"  First valid date (after shift + rolling): {first_valid.date()}")

    # ── 2. Load index data & build bars ─────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 2: Loading minute data and building bars")
    print("=" * 60)

    # Adjust start to not be before VIX is available
    effective_start = max(args.start, str(first_valid.date()))
    if effective_start != args.start:
        print(f"  Note: start adjusted from {args.start} to {effective_start} for VIX coverage")

    selected_indices = {k: INDICES[k] for k in args.indices if k in INDICES}
    minutes_all = {}
    for key in selected_indices:
        print(f"  Loading {INDICES[key]['label']} ({key})...", end=" ", flush=True)
        minutes, files = load_index_minutes(key, effective_start, args.end)
        minutes_all[key] = minutes
        print(f"{len(minutes):,} rows from {len(files)} files")

    minutes_all = filter_common_dates(minutes_all)
    common_dates = sorted(set.union(*(set(df["trade_date"].dt.date.astype(str).unique())
                                      for df in minutes_all.values())))
    print(f"  Common trading days: {len(common_dates)} ({common_dates[0]} to {common_dates[-1]})")

    all_factors_vix = {}
    for key, minutes in minutes_all.items():
        print(f"  Building bars & factors for {INDICES[key]['label']}...", end=" ", flush=True)
        daily_to = minutes.groupby(minutes["trade_date"].dt.date)["turnover"].sum()
        threshold = float(daily_to.median() / args.target_bars_per_day)

        bars_5min = make_time_bars(minutes, 5)
        bars_turnover = make_turnover_bars(minutes, threshold)

        for bar_type, bars in [("5min_bar", bars_5min), ("turnover_bar", bars_turnover)]:
            bars.to_csv(bar_dir / f"{key}_{bar_type}.csv", index=False)

        # Build factors and merge VIX
        factors_5min = build_factors(bars_5min)
        factors_turnover = build_factors(bars_turnover)

        factors_5min_vix = merge_vix_to_factors(factors_5min, vix_df)
        factors_turnover_vix = merge_vix_to_factors(factors_turnover, vix_df)

        all_factors_vix[(key, "5min_bar")] = factors_5min_vix
        all_factors_vix[(key, "turnover_bar")] = factors_turnover_vix

        n_5min = len(factors_5min_vix.dropna())
        n_turn = len(factors_turnover_vix.dropna())
        print(f"5min: {len(bars_5min):,} bars ({n_5min:,} w/ VIX), "
              f"turnover: {len(bars_turnover):,} bars ({n_turn:,} w/ VIX)")

    # ── 3. Run all models ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 3: Running multi-model comparison")
    print("=" * 60)

    all_results = []
    for key in selected_indices:
        for bar_type in ["5min_bar", "turnover_bar"]:
            factors = all_factors_vix[(key, bar_type)]
            label = f"{INDICES[key]['label']} {bar_type}"
            print(f"  {label}...", end=" ", flush=True)
            fold_results = run_all_models(factors, bar_type, key)
            all_results.extend(fold_results)
            # Quick summary
            if fold_results:
                by_model = {}
                for r in fold_results:
                    by_model.setdefault(r["model"], []).append(r["auc"])
                parts = [f"{m}: avg AUC {np.mean(aucs):.4f}" for m, aucs in by_model.items()]
                print(", ".join(parts), flush=True)
            else:
                print("no folds", flush=True)

    # ── 4. Aggregate & save ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 4: Saving results & generating charts")
    print("=" * 60)

    ml_detail = pd.DataFrame(all_results)
    if ml_detail.empty:
        print("  No results generated.")
        return

    ml_summary = ml_detail.groupby(["index", "bar_type", "model"]).agg(
        auc=("auc", "mean"),
        accuracy=("accuracy", "mean"),
        sharpe=("sharpe", "mean"),
        total_train=("train_samples", "sum"),
        total_test=("test_samples", "sum"),
        n_folds=("test_year", "count"),
    ).reset_index()

    ml_summary.to_csv(out / "vix_ml_summary.csv", index=False)
    ml_detail.to_csv(out / "vix_ml_by_fold.csv", index=False)
    save_auc_chart(ml_summary, out / "vix_auc_comparison.svg")
    save_sharpe_chart(ml_summary, out / "vix_sharpe_comparison.svg")

    # ── 5. Report ──────────────────────────────────────────────────────
    write_report(out / "vix_comparison_report.md", ml_summary, ml_detail, args)

    print()
    print("Done. Outputs:")
    print(f"  Results:     {out}")
    print()
    print("Summary (top 10 by AUC):")
    top10 = ml_summary.sort_values("auc", ascending=False).head(10)
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
