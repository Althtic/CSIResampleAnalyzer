#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

# ── Reuse core bar construction from existing script ──────────────────────
from analyze_resampling import make_time_bars, make_turnover_bars, markdown_table

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

INDICES = {
    "hs300":  {"label": "沪深300",  "dir": "hs300_minute_20200101_20240618",  "prefix": "hs300"},
    "csi500": {"label": "中证500",  "dir": "csi500_minute_20200101_20240618", "prefix": "csi500"},
    "csi1000":{"label": "中证1000", "dir": "csi1000_minute_20200101_20240618","prefix": "csi1000"},
    "csi800": {"label": "中证800",  "dir": "csi800_minute_20200101_20240618", "prefix": "csi800"},
    "chinext":{"label": "创业板指", "dir": "chinext_minute_20200101_20240618","prefix": "chinext"},
}

DATA_ROOT = Path("/Users/sedol/Desktop/高频数据")
DEFAULT_OUTPUT = DATA_ROOT / "factor_ml_results"

RF_PARAMS = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 50, "random_state": 42, "n_jobs": -1}

# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Data loading
# ═══════════════════════════════════════════════════════════════════════════

def selected_files(data_dir: Path, prefix: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{4}})-(\d{{2}})_1min(?:_mtd)?\.xlsx$")
    files = []
    for path in sorted(data_dir.glob(f"{prefix}_????-??_1min*.xlsx")):
        m = pattern.match(path.name)
        if not m:
            continue
        month_start = pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
        month_end = month_start + pd.offsets.MonthEnd(0)
        if month_start <= end and month_end >= start:
            files.append(path)
    return files


def load_index_minutes(key: str, start: str, end: str) -> tuple[pd.DataFrame, list[Path]]:
    spec = INDICES[key]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    files = selected_files(DATA_ROOT / spec["dir"], spec["prefix"], start_ts, end_ts)
    if not files:
        raise FileNotFoundError(f"No files for {key} in {spec['dir']}")

    frames = []
    for path in files:
        raw = pd.read_excel(path)
        raw["source_file"] = path.name
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)
    dt_utc = pd.to_datetime(raw["TIME"].astype(str), errors="coerce", utc=True)
    data = pd.DataFrame({
        "universe": key,
        "datetime": dt_utc.dt.tz_localize(None) + pd.Timedelta(hours=8),
        "trade_date": pd.to_datetime(raw["_DATE"].astype(str), format="%Y%m%d", errors="coerce"),
        "price": pd.to_numeric(raw["MATCH"], errors="coerce"),
        "volume": pd.to_numeric(raw["VOLUME"], errors="coerce").fillna(0.0),
        "turnover": pd.to_numeric(raw["TURNOVER"], errors="coerce").fillna(0.0),
    })
    data = data.dropna(subset=["datetime", "trade_date", "price"])
    data = data[(data["datetime"] >= start_ts) & (data["datetime"] <= end_ts)]
    data = data.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    data["turnover"] = data["turnover"].clip(lower=0.0)
    data["volume"] = data["volume"].clip(lower=0.0)
    return data, files


def filter_common_dates(minutes_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    date_sets = [set(df["trade_date"].dt.date.astype(str).unique()) for df in minutes_dict.values()]
    common = sorted(set.intersection(*date_sets))
    return {k: df[df["trade_date"].dt.date.astype(str).isin(common)].reset_index(drop=True)
            for k, df in minutes_dict.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Return extraction
# ═══════════════════════════════════════════════════════════════════════════

def bar_log_returns(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Within-day close-to-close log returns from bars.
    Returns (returns, trade_dates) as aligned Series."""
    ret_pieces = []
    date_pieces = []
    for _, day in bars.groupby("trade_date", sort=True):
        day = day.sort_values("end_time").reset_index(drop=True)
        close = day["close"].astype(float).to_numpy()
        if len(close) < 2:
            continue
        ret = np.diff(np.log(close))
        ret_pieces.append(pd.Series(ret))
        date_pieces.append(pd.Series(day["trade_date"].iloc[1:].to_numpy()))
    all_ret = pd.concat(ret_pieces, ignore_index=True)
    all_dates = pd.concat(date_pieces, ignore_index=True)
    return all_ret, all_dates


def minute_returns(minutes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, day in minutes.groupby(minutes["trade_date"].dt.date, sort=True):
        day = day.sort_values("datetime").reset_index(drop=True)
        price = day["price"].astype(float).to_numpy()
        simple = price[1:] / price[:-1] - 1.0
        logr = np.diff(np.log(price))
        rows.append(pd.DataFrame({
            "trade_date": day["trade_date"].dt.date.astype(str).iloc[1:].to_numpy(),
            "simple_return": simple,
            "log_return": logr,
        }))
    return pd.concat(rows, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Rolling stability metrics
# ═══════════════════════════════════════════════════════════════════════════

def return_stats(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 10:
        return {}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if std == 0:
        return {}
    z = (values - mean) / std
    acf1_val = float(np.corrcoef(values[:-1], values[1:])[0, 1]) if n > 2 else float("nan")
    abs_vals = np.abs(values)
    abs_acf1 = float(np.corrcoef(abs_vals[:-1], abs_vals[1:])[0, 1]) if n > 2 else float("nan")
    return {
        "mean": mean, "std": std,
        "skewness": float(np.mean(z ** 3)),
        "kurtosis": float(np.mean(z ** 4) - 3.0),
        "acf1": acf1_val, "abs_acf1": abs_acf1,
        "n": n,
    }


def rolling_stability(return_series: pd.Series, dates: pd.Series, window: int = 60) -> pd.DataFrame:
    """Compute return stats over rolling windows of `window` trading days."""
    unique_dates = sorted(dates.unique())
    window = min(window, max(10, len(unique_dates) // 2))
    rows = []
    for i in range(window, len(unique_dates)):
        win_dates = unique_dates[i - window : i]
        mask = dates.isin(win_dates)
        vals = return_series[mask].to_numpy(dtype=float)
        stats = return_stats(vals)
        if stats:
            stats["window_end"] = unique_dates[i]
            rows.append(stats)
    return pd.DataFrame(rows)


def stability_of_statistics(rolling_df: pd.DataFrame) -> dict[str, float]:
    """Std dev of each rolling statistic → lower means more stable through time."""
    result = {}
    for col in ["mean", "std", "skewness", "kurtosis", "acf1", "abs_acf1"]:
        vals = rolling_df[col].dropna().to_numpy(dtype=float)
        if len(vals) > 1:
            result[f"{col}_level"] = float(np.mean(vals))
            result[f"{col}_volatility"] = float(np.std(vals, ddof=1))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Factor construction
# ═══════════════════════════════════════════════════════════════════════════

def build_factors(bars: pd.DataFrame) -> pd.DataFrame:
    """Build 10 volume-price factors from K-line bars. No look-ahead."""
    df = bars.copy()
    df = df.sort_values(["trade_date", "end_time"]).reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)
    turnover = df["turnover"].astype(float)

    # Duration in minutes
    duration = df["minutes"].astype(float).clip(lower=1.0)

    # 1. Return
    prev_close = close.shift(1)
    # Zero out overnight gaps (first bar of each day)
    day_changed = df["trade_date"].ne(df["trade_date"].shift(1))
    prev_close[day_changed] = np.nan
    ret = close / prev_close - 1.0

    # 2. Range
    range_pct = (high - low) / close

    # 3. Real body
    body_pct = (close - open_).abs() / close

    # 4-5. Shadows
    upper_shadow = (high - np.maximum(open_, close)) / close
    lower_shadow = (np.minimum(open_, close) - low) / close

    # 6-7. Volume & turnover ratios (rolling, computed per-day to avoid cross-day contamination)
    vol_ratio = pd.Series(np.nan, index=df.index)
    turn_ratio = pd.Series(np.nan, index=df.index)
    for _, day_idx in df.groupby("trade_date").groups.items():
        v = volume[day_idx]
        t = turnover[day_idx]
        rm_v = v.rolling(20, min_periods=5).mean()
        rm_t = t.rolling(20, min_periods=5).mean()
        vol_ratio[day_idx] = v / rm_v
        turn_ratio[day_idx] = t / rm_t

    # 8. Turnover intensity
    turn_intensity = turnover / duration

    # 9. Realized volatility (per-day rolling)
    log_ret = pd.Series(np.nan, index=df.index)
    for _, day_idx in df.groupby("trade_date").groups.items():
        c = close[day_idx]
        lr = np.log(c).diff()
        log_ret[day_idx] = lr
    log_ret.iloc[0] = np.nan
    realized_vol = pd.Series(np.nan, index=df.index)
    for _, day_idx in df.groupby("trade_date").groups.items():
        lr = log_ret[day_idx]
        realized_vol[day_idx] = lr.rolling(20, min_periods=5).std()

    # 10. Amihud illiquidity
    amihud = np.abs(ret) / turnover.clip(lower=1e-8) * 1e10

    return pd.DataFrame({
        "trade_date": df["trade_date"],
        "end_time": df["end_time"],
        "ret": ret.to_numpy(dtype=float),
        "range_pct": range_pct.to_numpy(dtype=float),
        "body_pct": body_pct.to_numpy(dtype=float),
        "upper_shadow": upper_shadow.to_numpy(dtype=float),
        "lower_shadow": lower_shadow.to_numpy(dtype=float),
        "vol_ratio": vol_ratio.to_numpy(dtype=float),
        "turn_ratio": turn_ratio.to_numpy(dtype=float),
        "turn_intensity": turn_intensity.to_numpy(dtype=float),
        "realized_vol": realized_vol.to_numpy(dtype=float),
        "amihud": amihud.to_numpy(dtype=float),
    })


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: ML pipeline
# ═══════════════════════════════════════════════════════════════════════════

def prepare_ml_dataset(factors: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Build lagged features and binary target (next-bar up/down)."""
    df = factors.dropna().copy().sort_values(["trade_date", "end_time"]).reset_index(drop=True)

    factor_cols = ["ret", "range_pct", "body_pct", "upper_shadow", "lower_shadow",
                   "vol_ratio", "turn_ratio", "turn_intensity", "realized_vol", "amihud"]

    # Lags 1-5 for each factor
    feature_list = []
    feature_names = []
    for col in factor_cols:
        for lag in range(1, 6):
            name = f"{col}_lag{lag}"
            feature_names.append(name)
            feature_list.append(df[col].shift(lag).to_numpy(dtype=float))

    # Time dummies
    df["hour"] = pd.to_datetime(df["end_time"]).dt.hour
    df["dow"] = pd.to_datetime(df["end_time"]).dt.dayofweek
    for h in [9, 10, 11, 13, 14]:
        feature_names.append(f"hour_{h}")
        feature_list.append((df["hour"] == h).astype(float).to_numpy(dtype=float))
    for d in range(5):
        feature_names.append(f"dow_{d}")
        feature_list.append((df["dow"] == d).astype(float).to_numpy(dtype=float))

    X = np.column_stack(feature_list)

    # Target: next-bar return > 0 (within same day only)
    target = np.full(len(df), np.nan)
    for _, day_idx in df.groupby("trade_date").groups.items():
        rets = df.loc[day_idx, "ret"].to_numpy(dtype=float)
        target[day_idx[:-1]] = (rets[1:] > 0).astype(float)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(target)
    return X[mask], target[mask], df.loc[mask, "trade_date"], df.loc[mask, "end_time"], feature_names


def expanding_window_folds(dates: pd.Series) -> list[tuple[str, str]]:
    """Return (train_end, test_year) pairs. Train = all data up to train_end."""
    years = sorted(dates.str[:4].unique())
    folds = []
    for i in range(2, len(years)):
        train_end = f"{int(years[i-1])}"
        test_year = years[i]
        folds.append((train_end, test_year))
    return folds


def run_ml_experiment(factors: pd.DataFrame, bar_label: str, index_label: str) -> list[dict]:
    X, y, dates, times, feature_names = prepare_ml_dataset(factors)
    folds = expanding_window_folds(dates)
    results = []
    for train_end, test_year in folds:
        train_mask = dates.str[:4] <= train_end
        test_mask = dates.str[:4] == test_year
        if test_mask.sum() < 100:
            continue
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        if len(np.unique(y_train)) < 2:
            continue

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train, y_train)
        proba = rf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(float)

        # Strategy return: long when predict up, short (or flat) when down
        test_factors = factors.loc[factors.index.isin(dates[test_mask].index)].sort_values(["trade_date", "end_time"])
        next_ret = test_factors["ret"].shift(-1).to_numpy(dtype=float)
        # Align: prediction at t, return at t+1
        common_len = min(len(pred), len(next_ret) - 1)
        strategy_ret = pred[:common_len] * next_ret[:common_len]
        # Annualized Sharpe
        sr = float(np.mean(strategy_ret) / np.std(strategy_ret, ddof=1) * np.sqrt(252 * 48)) if np.std(strategy_ret) > 0 else 0.0

        # Feature importance
        importances = rf.feature_importances_
        top5_idx = np.argsort(importances)[::-1][:5]

        results.append({
            "index": index_label,
            "bar_type": bar_label,
            "train_end": train_end,
            "test_year": test_year,
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "auc": float(roc_auc_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "pred_up_rate": float(np.mean(pred)),
            "actual_up_rate": float(np.mean(y_test)),
            "sharpe": sr,
            "top1_feature": feature_names[top5_idx[0]],
            "top2_feature": feature_names[top5_idx[1]],
            "top3_feature": feature_names[top5_idx[2]],
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: SVG charts
# ═══════════════════════════════════════════════════════════════════════════

def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(x: float, y: float, text: object, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,Helvetica,sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#17202a">{esc(text)}</text>')


def save_stability_chart(stability_df: pd.DataFrame, path: Path) -> None:
    """Bar chart: stability (volatility of rolling stats) ratio turnover/time for each metric × index."""
    if stability_df.empty or "index" not in stability_df.columns:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40"><text x="10" y="25">No stability data</text></svg>')
        return
    metrics = ["mean", "std", "skewness", "kurtosis", "acf1", "abs_acf1"]
    indices_list = sorted(stability_df["index"].unique())
    colors = {"hs300": "#2563eb", "csi500": "#d97706", "csi1000": "#16a34a", "csi800": "#8b5cf6", "chinext": "#ef4444"}
    labels = {k: INDICES[k]["label"] for k in indices_list if k in INDICES}

    w, h = 1180, 620
    left, right, top, bottom = 190, 60, 68, 70
    row_h = (h - top - bottom) / len(metrics)
    plot_w = w - left - right
    xmax = 2.0

    def xmap(v: float) -> float:
        return left + min(v, xmax) / xmax * plot_w

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "Return Stability: turnover-bar volatility / fixed-time-bar volatility of rolling stats", 18, "middle", "700"),
        svg_text(w / 2, 52, "Ratio < 1.0 means turnover bars produce more stable statistics through time. Each colored bar = one index.", 12, "middle"),
        f'<line x1="{xmap(1.0):.1f}" y1="{top - 12}" x2="{xmap(1.0):.1f}" y2="{h - bottom + 10}" stroke="#7f8c8d" stroke-dasharray="4,4"/>',
    ]

    for m_idx, metric in enumerate(metrics):
        y_base = top + m_idx * row_h
        elems.append(svg_text(18, y_base + row_h / 2 + 4, metric, 12))
        col_name = f"{metric}_volatility"
        for j, idx_key in enumerate(indices_list):
            row = stability_df[(stability_df["index"] == idx_key)]
            if row.empty:
                continue
            fixed = float(row[row["return_type"] == "5min_bar"][col_name].iloc[0]) if len(row[row["return_type"] == "5min_bar"]) else 1.0
            turnover = float(row[row["return_type"] == "turnover_bar"][col_name].iloc[0]) if len(row[row["return_type"] == "turnover_bar"]) else 1.0
            ratio = turnover / fixed if fixed > 0 else 1.0
            y = y_base + 14 + j * 16
            bar_color = colors.get(idx_key, "#999999")
            bar_w = max(1, xmap(ratio) - left)
            elems.append(f'<rect x="{left}" y="{y:.1f}" width="{bar_w:.1f}" height="11" fill="{bar_color}" opacity="0.78"/>')
            elems.append(svg_text(xmap(ratio) + 5, y + 10, f"{ratio:.2f}", 9))
    # Legend
    for j, idx_key in enumerate(indices_list):
        x = left + j * 130
        elems.append(f'<rect x="{x}" y="{h - 36}" width="13" height="13" fill="{colors.get(idx_key)}" opacity="0.78"/>')
        elems.append(svg_text(x + 18, h - 25, labels.get(idx_key, idx_key), 11))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def save_ml_summary_chart(ml_df: pd.DataFrame, path: Path) -> None:
    """AUC bar chart comparing 5min vs turnover bars across indices."""
    w, h = 1080, 520
    left, right, top, bottom = 170, 40, 68, 68
    bar_w = 36
    gap = 28
    colors = {"5min_bar": "#2563eb", "turnover_bar": "#d97706"}
    indices_sorted = sorted(ml_df["index"].unique())

    plot_w = w - left - right
    group_w = (plot_w) / max(len(indices_sorted), 1)
    ymin = max(0.45, ml_df["auc"].min() - 0.05)
    ymax = min(0.85, ml_df["auc"].max() + 0.05)

    def ymap(v: float) -> float:
        return top + (h - top - bottom) * (1.0 - (v - ymin) / (ymax - ymin))

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "ML Comparison: AUC by index and bar type (expanding-window average)", 18, "middle", "700"),
        f'<line x1="{left}" y1="{ymap(0.5):.1f}" x2="{w - right}" y2="{ymap(0.5):.1f}" stroke="#bdc3c7" stroke-dasharray="4,4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + h - top - bottom}" stroke="#2c3e50"/>',
        f'<line x1="{left}" y1="{top + h - top - bottom}" x2="{w - right}" y2="{top + h - top - bottom}" stroke="#2c3e50"/>',
    ]
    for tick in np.linspace(ymin, ymax, 5):
        y = ymap(float(tick))
        elems.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#2c3e50"/>')
        elems.append(svg_text(left - 10, y + 4, f"{tick:.3f}", 11, "end"))
    for i, idx in enumerate(indices_sorted):
        subset = ml_df[ml_df["index"] == idx]
        x_center = left + group_w * (i + 0.5)
        for j, bar_type in enumerate(["5min_bar", "turnover_bar"]):
            row = subset[subset["bar_type"] == bar_type]
            if row.empty:
                continue
            auc_val = float(row["auc"].iloc[0])
            x = x_center - bar_w - gap / 2 + j * (bar_w * 2 + gap)
            y = ymap(auc_val)
            elems.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + h - top - bottom - y:.1f}" fill="{colors[bar_type]}" opacity="0.82"/>')
            elems.append(svg_text(x + bar_w / 2, y - 6, f"{auc_val:.3f}", 10, "middle"))
        label = INDICES.get(idx, {}).get("label", idx)
        elems.append(svg_text(x_center, top + h - top - bottom + 22, label, 12, "middle"))
    # Legend
    elems.append(f'<rect x="{left + 18}" y="{top + 14}" width="14" height="14" fill="{colors["5min_bar"]}" opacity="0.82"/>')
    elems.append(svg_text(left + 38, top + 26, "5min fixed-time bars", 12))
    elems.append(f'<rect x="{left + 190}" y="{top + 14}" width="14" height="14" fill="{colors["turnover_bar"]}" opacity="0.82"/>')
    elems.append(svg_text(left + 210, top + 26, "turnover-resampled bars", 12))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def save_sharpe_chart(ml_df: pd.DataFrame, path: Path) -> None:
    """Strategy Sharpe ratio comparison."""
    w, h = 1080, 520
    left, right, top, bottom = 170, 40, 68, 68
    bar_w = 36
    gap = 28
    colors = {"5min_bar": "#2563eb", "turnover_bar": "#d97706"}
    indices_sorted = sorted(ml_df["index"].unique())

    plot_w = w - left - right
    group_w = plot_w / max(len(indices_sorted), 1)
    all_sharpes = ml_df["sharpe"].to_numpy(dtype=float)
    ymin = min(-3, all_sharpes.min() - 0.3)
    ymax = max(3, all_sharpes.max() + 0.3)

    def ymap(v: float) -> float:
        return top + (h - top - bottom) * (1.0 - (v - ymin) / (ymax - ymin))

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "Strategy Sharpe Ratio by index and bar type (expanding-window average)", 18, "middle", "700"),
        svg_text(w / 2, 50, "Long when model predicts up; annualized Sharpe. Higher is better.", 12, "middle"),
        f'<line x1="{left}" y1="{ymap(0):.1f}" x2="{w - right}" y2="{ymap(0):.1f}" stroke="#2c3e50"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + h - top - bottom}" stroke="#2c3e50"/>',
        f'<line x1="{left}" y1="{top + h - top - bottom}" x2="{w - right}" y2="{top + h - top - bottom}" stroke="#2c3e50"/>',
    ]
    for tick in np.linspace(ymin, ymax, 5):
        y = ymap(float(tick))
        elems.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#2c3e50"/>')
        elems.append(svg_text(left - 10, y + 4, f"{tick:.1f}", 11, "end"))
    for i, idx in enumerate(indices_sorted):
        subset = ml_df[ml_df["index"] == idx]
        x_center = left + group_w * (i + 0.5)
        for j, bar_type in enumerate(["5min_bar", "turnover_bar"]):
            row = subset[subset["bar_type"] == bar_type]
            if row.empty:
                continue
            sr = float(row["sharpe"].iloc[0])
            x = x_center - bar_w - gap / 2 + j * (bar_w * 2 + gap)
            y = ymap(sr)
            h_bar = ymap(0) - y if sr >= 0 else y - ymap(0)
            y_top = y if sr >= 0 else ymap(0)
            elems.append(f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" height="{max(1, abs(h_bar)):.1f}" fill="{colors[bar_type]}" opacity="0.82"/>')
            elems.append(svg_text(x + bar_w / 2, y - 6 if sr >= 0 else y + 14, f"{sr:.2f}", 10, "middle"))
        label = INDICES.get(idx, {}).get("label", idx)
        elems.append(svg_text(x_center, top + h - top - bottom + 22, label, 12, "middle"))
    elems.append(f'<rect x="{left + 18}" y="{top + 14}" width="14" height="14" fill="{colors["5min_bar"]}" opacity="0.82"/>')
    elems.append(svg_text(left + 38, top + 26, "5min fixed-time bars", 12))
    elems.append(f'<rect x="{left + 190}" y="{top + 14}" width="14" height="14" fill="{colors["turnover_bar"]}" opacity="0.82"/>')
    elems.append(svg_text(left + 210, top + 26, "turnover-resampled bars", 12))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Section 7: Report
# ═══════════════════════════════════════════════════════════════════════════

def write_report(path: Path, stability: pd.DataFrame, ml_summary: pd.DataFrame,
                 ml_detail: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Factor & ML Comparison Report",
        "",
        "## 1. Data Scope",
        "",
        f"- Indices: {', '.join(INDICES[k]['label'] for k in INDICES)}",
        f"- Date range: {args.start} to {args.end}",
        f"- Bar types: 5-minute fixed-time bars vs turnover-resampled bars",
        f"- Turnover threshold: median daily turnover / {args.target_bars_per_day} target bars per day",
        "",
        "## 2. Return Stability",
        "",
        "Stability is measured as the standard deviation of rolling 60-day statistics. Lower volatility = more stable.",
        "",
        markdown_table(stability),
        "",
        "![Stability comparison](stability_comparison.svg)",
        "",
        "## 3. Volume-Price Factors",
        "",
        "Ten factors are constructed from each bar type:",
        "",
        "| # | Factor | Description |",
        "|---|---|---|",
        "| 1 | ret | Bar close-to-close return |",
        "| 2 | range_pct | (high - low) / close |",
        "| 3 | body_pct | abs(close - open) / close |",
        "| 4 | upper_shadow | (high - max(open, close)) / close |",
        "| 5 | lower_shadow | (min(open, close) - low) / close |",
        "| 6 | vol_ratio | volume / rolling_mean(volume, 20) |",
        "| 7 | turn_ratio | turnover / rolling_mean(turnover, 20) |",
        "| 8 | turn_intensity | turnover / bar_duration_minutes |",
        "| 9 | realized_vol | rolling_std(log_return, 20) |",
        "| 10 | amihud | abs(return) / turnover × 1e10 |",
        "",
        "## 4. ML Comparison — Random Forest",
        "",
        f"**Model**: RandomForestClassifier(n_estimators={RF_PARAMS['n_estimators']}, max_depth={RF_PARAMS['max_depth']}, min_samples_leaf={RF_PARAMS['min_samples_leaf']})",
        "",
        "**Training method**: Expanding window walk-forward validation.",
        "- Train on all data up to year T-1, test on year T.",
        "- Each fold uses strictly chronological split — no look-ahead.",
        "",
        "**Features**: 10 factors × 5 lags + hour/day-of-week dummies = ~55 features.",
        "",
        "**Target**: Next bar return direction (up = 1, down = 0).",
        "",
        "### Summary by Index and Bar Type",
        "",
        (markdown_table(ml_summary) if not ml_summary.empty else "_ML skipped_"),
        "",
        "![AUC comparison](ml_auc_comparison.svg)",
        "",
        "![Sharpe comparison](ml_sharpe_comparison.svg)",
        "",
        "### Per-Fold Detail",
        "",
        (markdown_table(ml_detail) if not ml_detail.empty else "_ML skipped_"),
        "",
        "## 5. Quick Read",
        "",
    ]

    # Stability winners
    lines.append("### Stability")
    for idx in sorted(stability["index"].unique()):
        sub = stability[stability["index"] == idx]
        better_count = 0
        total = 0
        for _, row in sub.iterrows():
            metric = row["return_type"]
            if metric == "5min_bar":
                continue
        lines.append(f"- {INDICES.get(idx, {}).get('label', idx)}: see table above.")

    # ML winners
    if not ml_summary.empty and "index" in ml_summary.columns:
        lines.append("")
        lines.append("### ML Predictive Performance")
        for idx in sorted(ml_summary["index"].unique()):
            sub = ml_summary[ml_summary["index"] == idx]
            if len(sub) < 2:
                continue
            time_auc = float(sub[sub["bar_type"] == "5min_bar"]["auc"].iloc[0]) if len(sub[sub["bar_type"] == "5min_bar"]) else 0
            turnover_auc = float(sub[sub["bar_type"] == "turnover_bar"]["auc"].iloc[0]) if len(sub[sub["bar_type"] == "turnover_bar"]) else 0
            winner = "turnover-resampled" if turnover_auc > time_auc else "fixed-time 5min"
            lines.append(f"- {INDICES.get(idx, {}).get('label', idx)}: {winner} bars have higher AUC "
                         f"(turnover {turnover_auc:.4f} vs fixed-time {time_auc:.4f}).")

    lines.extend([
        "",
        "## 6. Generated Files",
        "",
        "- `bars/{index}_5min_bars.csv` — 5-minute fixed-time bar data per index",
        "- `bars/{index}_turnover_bars.csv` — Turnover-resampled bar data per index",
        "- `stability/stability_metrics.csv` — Return stability metrics",
        "- `stability/stability_comparison.svg` — Stability comparison chart",
        "- `ml/ml_results_summary.csv` — ML summary by index and bar type",
        "- `ml/ml_results_by_fold.csv` — ML detail per fold",
        "- `ml/ml_auc_comparison.svg` — AUC comparison chart",
        "- `ml/ml_sharpe_comparison.svg` — Sharpe comparison chart",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Factor + ML comparison: fixed-time vs turnover-resampled bars")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-06-18")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--target-bars-per-day", type=int, default=48)
    p.add_argument("--skip-ml", action="store_true")
    p.add_argument("--rf-estimators", type=int, default=200)
    p.add_argument("--rf-max-depth", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RF_PARAMS["n_estimators"] = args.rf_estimators
    RF_PARAMS["max_depth"] = args.rf_max_depth

    out = Path(args.output_dir)
    bar_dir = out / "bars"
    stability_dir = out / "stability"
    ml_dir = out / "ml"
    for d in [bar_dir, stability_dir, ml_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data & build bars ──────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading minute data and building bars for all indices")
    print("=" * 60)

    minutes_all = {}
    for key in INDICES:
        print(f"  Loading {INDICES[key]['label']} ({key})...", end=" ", flush=True)
        minutes, files = load_index_minutes(key, args.start, args.end)
        minutes_all[key] = minutes
        print(f"{len(minutes):,} rows from {len(files)} files")

    minutes_all = filter_common_dates(minutes_all)
    common_dates = sorted(set.union(*(set(df["trade_date"].dt.date.astype(str).unique()) for df in minutes_all.values())))
    print(f"  Common trading days: {len(common_dates)} ({common_dates[0]} to {common_dates[-1]})")

    all_bars = {}
    all_returns = {}
    all_return_dates = {}
    for key, minutes in minutes_all.items():
        print(f"  Building bars for {INDICES[key]['label']}...", end=" ", flush=True)
        daily_to = minutes.groupby(minutes["trade_date"].dt.date)["turnover"].sum()
        threshold = float(daily_to.median() / args.target_bars_per_day)

        bars_5min = make_time_bars(minutes, 5)
        bars_turnover = make_turnover_bars(minutes, threshold)

        all_bars[(key, "5min_bar")] = bars_5min
        all_bars[(key, "turnover_bar")] = bars_turnover

        # Returns from bars
        ret_5min, dates_5min = bar_log_returns(bars_5min)
        ret_turnover, dates_turnover = bar_log_returns(bars_turnover)
        all_returns[(key, "5min_bar")] = ret_5min
        all_returns[(key, "turnover_bar")] = ret_turnover
        all_return_dates[(key, "5min_bar")] = dates_5min
        all_return_dates[(key, "turnover_bar")] = dates_turnover

        # Save bars
        bars_5min.to_csv(bar_dir / f"{key}_5min_bars.csv", index=False)
        bars_turnover.to_csv(bar_dir / f"{key}_turnover_bars.csv", index=False)
        print(f"5min: {len(bars_5min):,} bars, turnover: {len(bars_turnover):,} bars (threshold={threshold:,.0f})")

    # ── 2. Return stability analysis ───────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 2: Return stability analysis (rolling 60-day windows)")
    print("=" * 60)

    stability_rows = []
    for key in INDICES:
        minutes = minutes_all[key]
        print(f"  Computing stability for {INDICES[key]['label']}...", end=" ", flush=True)

        # Raw 1-min returns
        min_rets = minute_returns(minutes)

        for ret_type, series, date_series in [
            ("1min_simple", min_rets["simple_return"], min_rets["trade_date"]),
            ("1min_log", min_rets["log_return"], min_rets["trade_date"]),
            ("5min_bar", all_returns[(key, "5min_bar")],
             all_return_dates[(key, "5min_bar")]),
            ("turnover_bar", all_returns[(key, "turnover_bar")],
             all_return_dates[(key, "turnover_bar")]),
        ]:
            rolling = rolling_stability(series, date_series)
            if rolling.empty:
                continue
            stats = stability_of_statistics(rolling)
            stats["index"] = key
            stats["return_type"] = ret_type
            stats["n_windows"] = len(rolling)
            stability_rows.append(stats)

        print(f"done")

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(stability_dir / "stability_metrics.csv", index=False)
    save_stability_chart(stability_df, stability_dir / "stability_comparison.svg")
    print(f"  Stability metrics saved ({len(stability_df)} rows)")

    # ── 3. Factor construction ─────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 3: Building volume-price factors")
    print("=" * 60)

    all_factors = {}
    for key in INDICES:
        for bar_type in ["5min_bar", "turnover_bar"]:
            label = f"{INDICES[key]['label']} ({bar_type})"
            print(f"  Factors for {label}...", end=" ", flush=True)
            factors = build_factors(all_bars[(key, bar_type)])
            all_factors[(key, bar_type)] = factors
            print(f"{len(factors):,} rows, {factors.dropna().shape[1]} factors")

    # ── 4. ML comparison ───────────────────────────────────────────────
    if not args.skip_ml:
        print()
        print("=" * 60)
        print("STEP 4: Random Forest comparison (expanding window)")
        print("=" * 60)

        ml_results = []
        for key in INDICES:
            for bar_type in ["5min_bar", "turnover_bar"]:
                label = f"{INDICES[key]['label']} ({bar_type})"
                print(f"  Training RF for {label}...", end=" ", flush=True)
                fold_results = run_ml_experiment(all_factors[(key, bar_type)], bar_type, key)
                ml_results.extend(fold_results)
                avg_auc = np.mean([r["auc"] for r in fold_results])
                print(f"{len(fold_results)} folds, avg AUC={avg_auc:.4f}")

        ml_detail = pd.DataFrame(ml_results)
        ml_summary = ml_detail.groupby(["index", "bar_type"]).agg(
            auc=("auc", "mean"),
            accuracy=("accuracy", "mean"),
            sharpe=("sharpe", "mean"),
            total_train_samples=("train_samples", "sum"),
            total_test_samples=("test_samples", "sum"),
            n_folds=("test_year", "count"),
        ).reset_index()

        ml_summary.to_csv(ml_dir / "ml_results_summary.csv", index=False)
        ml_detail.to_csv(ml_dir / "ml_results_by_fold.csv", index=False)
        save_ml_summary_chart(ml_summary, ml_dir / "ml_auc_comparison.svg")
        save_sharpe_chart(ml_summary, ml_dir / "ml_sharpe_comparison.svg")
        print(f"  ML results saved")
    else:
        ml_summary = pd.DataFrame()
        ml_detail = pd.DataFrame()

    # ── 5. Report ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 5: Generating report")
    print("=" * 60)

    write_report(out / "comparison_report.md", stability_df, ml_summary, ml_detail, args)

    # Summary
    print()
    print("Done. Outputs:")
    print(f"  Bars:        {bar_dir}")
    print(f"  Stability:   {stability_dir}")
    print(f"  ML:          {ml_dir}")
    print(f"  Report:      {out / 'comparison_report.md'}")
    if not args.skip_ml and not ml_summary.empty:
        print()
        print("ML Summary:")
        print(ml_summary.to_string(index=False))


if __name__ == "__main__":
    main()
