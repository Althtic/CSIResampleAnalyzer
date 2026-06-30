#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


FIELD_DICTIONARY = [
    ("MATCH", "Minute matched/last index level. Used as the minute close price."),
    ("AVGPRICE", "Average price field. In the inspected HS300 files it is INVALID, so it is not used."),
    ("VOLUME", "Minute trading volume."),
    ("TURNOVER", "Minute trading amount/value. Used to build equal-turnover bars."),
    ("TIME", "Minute timestamp, stored as ISO time with +08:00 timezone."),
    ("_DATE", "Trading date in YYYYMMDD integer format."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed-time bars with equal-turnover bars for HS300 1-minute data."
    )
    parser.add_argument("--data-dir", default="hs300_minute_20200101_20240618")
    parser.add_argument("--output-dir", default="resampling_results")
    parser.add_argument("--start", default="2024-04-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument("--time-step", type=int, default=5, help="Minutes per fixed-time bar.")
    parser.add_argument(
        "--target-bars-per-day",
        type=int,
        default=48,
        help="Used to infer turnover threshold: median daily turnover / target bars.",
    )
    parser.add_argument(
        "--turnover-threshold",
        type=float,
        default=None,
        help="Optional fixed turnover threshold. If omitted, inferred from the sample.",
    )
    return parser.parse_args()


def selected_month_files(data_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    files: list[Path] = []
    for path in sorted(data_dir.glob("hs300_????-??_1min.xlsx")):
        match = re.search(r"hs300_(\d{4})-(\d{2})_1min\.xlsx$", path.name)
        if not match:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        month_start = pd.Timestamp(year=year, month=month, day=1)
        month_end = month_start + pd.offsets.MonthEnd(0)
        if month_start <= end and month_end >= start:
            files.append(path)
    return files


def load_minutes(data_dir: Path, start: str, end: str) -> tuple[pd.DataFrame, list[Path]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    files = selected_month_files(data_dir, start_ts, end_ts)
    if not files:
        raise FileNotFoundError(f"No monthly HS300 files found in {data_dir} for {start} to {end}")

    frames = []
    for path in files:
        frame = pd.read_excel(path)
        frame["source_file"] = path.name
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    dt_utc = pd.to_datetime(raw["TIME"].astype(str), errors="coerce", utc=True)
    data = pd.DataFrame(
        {
            "datetime": dt_utc.dt.tz_localize(None) + pd.Timedelta(hours=8),
            "trade_date": pd.to_datetime(raw["_DATE"].astype(str), format="%Y%m%d", errors="coerce"),
            "price": pd.to_numeric(raw["MATCH"], errors="coerce"),
            "volume": pd.to_numeric(raw["VOLUME"], errors="coerce").fillna(0.0),
            "turnover": pd.to_numeric(raw["TURNOVER"], errors="coerce").fillna(0.0),
            "source_file": raw["source_file"],
        }
    )
    data = data.dropna(subset=["datetime", "trade_date", "price"])
    data = data[(data["datetime"] >= start_ts) & (data["datetime"] <= end_ts)]
    data = data.sort_values(["datetime"]).drop_duplicates("datetime").reset_index(drop=True)
    data["turnover"] = data["turnover"].clip(lower=0.0)
    data["volume"] = data["volume"].clip(lower=0.0)
    return data, files


def finalize_bar(part: pd.DataFrame, kind: str, day_bar: int, partial: bool) -> dict[str, object]:
    prices = part["price"].astype(float)
    return {
        "bar_type": kind,
        "trade_date": part["trade_date"].iloc[0].date().isoformat(),
        "day_bar": day_bar,
        "start_time": part["datetime"].iloc[0],
        "end_time": part["datetime"].iloc[-1],
        "open": float(prices.iloc[0]),
        "high": float(prices.max()),
        "low": float(prices.min()),
        "close": float(prices.iloc[-1]),
        "volume": float(part["volume"].sum()),
        "turnover": float(part["turnover"].sum()),
        "minutes": int(len(part)),
        "partial": bool(partial),
    }


def make_time_bars(minutes: pd.DataFrame, step: int) -> pd.DataFrame:
    bars = []
    for _, day in minutes.groupby(minutes["trade_date"].dt.date, sort=True):
        day = day.sort_values("datetime").reset_index(drop=True)
        groups = np.arange(len(day)) // step
        for day_bar, part in day.groupby(groups, sort=True):
            partial = len(part) < step
            bars.append(finalize_bar(part, f"{step}min_time", int(day_bar), partial))
    return pd.DataFrame(bars)


def make_turnover_bars(minutes: pd.DataFrame, threshold: float) -> pd.DataFrame:
    bars = []
    for _, day in minutes.groupby(minutes["trade_date"].dt.date, sort=True):
        day = day.sort_values("datetime").reset_index(drop=True)
        start_idx = 0
        acc = 0.0
        day_bar = 0
        for idx, turnover in enumerate(day["turnover"].to_numpy(dtype=float)):
            acc += turnover
            if acc >= threshold and idx >= start_idx:
                part = day.iloc[start_idx : idx + 1]
                bars.append(finalize_bar(part, "turnover", day_bar, False))
                start_idx = idx + 1
                day_bar += 1
                acc = 0.0
        if start_idx < len(day):
            part = day.iloc[start_idx:]
            bars.append(finalize_bar(part, "turnover", day_bar, True))
    return pd.DataFrame(bars)


def returns_from_bars(bars: pd.DataFrame) -> np.ndarray:
    close = bars["close"].astype(float).to_numpy()
    returns = np.diff(np.log(close))
    return returns[np.isfinite(returns)]


def autocorr(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= lag + 2:
        return float("nan")
    a = values[:-lag]
    b = values[lag:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def distribution_stats(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4 or np.std(values) == 0:
        return float("nan"), float("nan")
    z = (values - values.mean()) / values.std(ddof=0)
    skew = float(np.mean(z**3))
    excess_kurtosis = float(np.mean(z**4) - 3.0)
    return skew, excess_kurtosis


def summarize_bars(name: str, bars: pd.DataFrame) -> dict[str, float | str]:
    returns = returns_from_bars(bars)
    abs_returns = np.abs(returns)
    skew, excess = distribution_stats(returns)
    jb = len(returns) / 6.0 * (skew**2 + (excess**2) / 4.0) if np.isfinite(skew + excess) else float("nan")
    daily_counts = bars.groupby("trade_date").size()
    return {
        "sample": name,
        "bars": int(len(bars)),
        "trading_days": int(daily_counts.size),
        "bars_per_day_mean": float(daily_counts.mean()),
        "bars_per_day_std": float(daily_counts.std(ddof=0)),
        "minutes_mean": float(bars["minutes"].mean()),
        "minutes_std": float(bars["minutes"].std(ddof=0)),
        "turnover_mean": float(bars["turnover"].mean()),
        "turnover_std": float(bars["turnover"].std(ddof=0)),
        "turnover_cv": float(bars["turnover"].std(ddof=0) / bars["turnover"].mean()),
        "return_count": int(len(returns)),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns, ddof=1)),
        "return_skew": skew,
        "return_excess_kurtosis": excess,
        "jarque_bera": float(jb),
        "jarque_bera_per_return": float(jb / len(returns)) if len(returns) else float("nan"),
        "return_acf1": autocorr(returns, 1),
        "return_acf5": autocorr(returns, 5),
        "abs_return_acf1": autocorr(abs_returns, 1),
        "abs_return_acf5": autocorr(abs_returns, 5),
    }


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)

    def fmt(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            if math.isnan(float(value)):
                return ""
            return f"{float(value):.6g}"
        return str(value)

    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def svg_text(x: float, y: float, text: object, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#17202a">{esc(text)}</text>'
    )


def save_return_histogram(time_returns: np.ndarray, turnover_returns: np.ndarray, path: Path) -> None:
    w, h = 980, 540
    left, right, top, bottom = 78, 28, 52, 72
    plot_w, plot_h = w - left - right, h - top - bottom
    combined = np.concatenate([time_returns, turnover_returns]) * 10000.0
    lo, hi = np.nanpercentile(combined, [1, 99])
    lo, hi = float(lo), float(hi)
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    bins = np.linspace(lo, hi, 45)
    t_hist, edges = np.histogram(time_returns * 10000.0, bins=bins, density=True)
    v_hist, _ = np.histogram(turnover_returns * 10000.0, bins=bins, density=True)
    ymax = max(float(np.nanmax(t_hist)), float(np.nanmax(v_hist))) * 1.15

    def xmap(x: float) -> float:
        return left + (x - lo) / (hi - lo) * plot_w

    def ymap(y: float) -> float:
        return top + plot_h - y / ymax * plot_h

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 28, "Log return distribution, clipped to 1st-99th percentile", 18, "middle", "700"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#2c3e50"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#2c3e50"/>',
    ]
    for hist, color, dx in [(t_hist, "#2f80ed", 0.0), (v_hist, "#d35400", 0.42)]:
        for i, y in enumerate(hist):
            x0, x1 = xmap(float(edges[i])), xmap(float(edges[i + 1]))
            bar_w = (x1 - x0) * 0.56
            x = x0 + (x1 - x0) * dx
            elems.append(
                f'<rect x="{x:.2f}" y="{ymap(float(y)):.2f}" width="{bar_w:.2f}" '
                f'height="{top + plot_h - ymap(float(y)):.2f}" fill="{color}" opacity="0.62"/>'
            )
    for tick in np.linspace(lo, hi, 7):
        x = xmap(float(tick))
        elems.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#2c3e50"/>')
        elems.append(svg_text(x, top + plot_h + 24, f"{tick:.1f}", 11, "middle"))
    elems.append(svg_text(left + plot_w / 2, h - 24, "log return, basis points", 12, "middle"))
    elems.append(f'<rect x="{left + 18}" y="{top + 12}" width="16" height="16" fill="#2f80ed" opacity="0.7"/>')
    elems.append(svg_text(left + 42, top + 25, "5-minute time bars", 12))
    elems.append(f'<rect x="{left + 190}" y="{top + 12}" width="16" height="16" fill="#d35400" opacity="0.7"/>')
    elems.append(svg_text(left + 214, top + 25, "turnover bars", 12))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def save_metric_ratios(stats: pd.DataFrame, path: Path) -> None:
    time_row = stats[stats["sample"].str.contains("time")].iloc[0]
    turn_row = stats[stats["sample"].eq("turnover")].iloc[0]
    metrics = [
        ("Turnover CV", "turnover_cv", "lower is more even sampling"),
        ("|Return ACF1|", "return_acf1", "closer to zero is better"),
        ("Abs return ACF1", "abs_return_acf1", "lower means less volatility clustering"),
        ("|Excess kurtosis|", "return_excess_kurtosis", "closer to zero is more Gaussian-like"),
        ("JB / return", "jarque_bera_per_return", "lower is more Gaussian-like"),
    ]
    rows = []
    for label, column, note in metrics:
        base = abs(float(time_row[column])) if "ACF" in label or "kurtosis" in label else float(time_row[column])
        value = abs(float(turn_row[column])) if "ACF" in label or "kurtosis" in label else float(turn_row[column])
        if np.isfinite(base) and base != 0 and np.isfinite(value):
            rows.append((label, value / base, value, base, note))

    w, h = 980, 430
    left, right, top, bottom = 230, 190, 58, 46
    row_h = 54
    plot_w = w - left - right
    xmax = max(1.55, max(r[1] for r in rows) * 1.15)

    def xmap(x: float) -> float:
        return left + x / xmax * plot_w

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "Turnover bars / time bars diagnostic ratios", 18, "middle", "700"),
        svg_text(w / 2, 50, "Values below 1.0 favor turnover bars for these diagnostics.", 12, "middle"),
        f'<line x1="{xmap(1.0):.1f}" y1="{top - 10}" x2="{xmap(1.0):.1f}" y2="{top + row_h * len(rows) + 8}" stroke="#7f8c8d" stroke-dasharray="4,4"/>',
    ]
    for tick in np.linspace(0, xmax, 6):
        x = xmap(float(tick))
        elems.append(f'<line x1="{x:.1f}" y1="{top + row_h * len(rows) + 4}" x2="{x:.1f}" y2="{top + row_h * len(rows) + 10}" stroke="#2c3e50"/>')
        elems.append(svg_text(x, top + row_h * len(rows) + 28, f"{tick:.2f}", 11, "middle"))
    for idx, (label, ratio, value, base, note) in enumerate(rows):
        y = top + idx * row_h
        color = "#1f9d55" if ratio < 1.0 else "#c0392b"
        elems.append(svg_text(22, y + 27, label, 13))
        elems.append(svg_text(w - 18, y + 27, note, 11, "end"))
        elems.append(f'<rect x="{left}" y="{y + 10}" width="{max(1, xmap(ratio) - left):.1f}" height="24" rx="3" fill="{color}" opacity="0.78"/>')
        elems.append(svg_text(xmap(ratio) + 7, y + 28, f"{ratio:.2f}x", 12))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def save_abs_return_acf(time_returns: np.ndarray, turnover_returns: np.ndarray, path: Path) -> None:
    lags = np.arange(1, 21)
    time_acf = np.array([autocorr(np.abs(time_returns), int(lag)) for lag in lags])
    turn_acf = np.array([autocorr(np.abs(turnover_returns), int(lag)) for lag in lags])
    all_vals = np.concatenate([time_acf[np.isfinite(time_acf)], turn_acf[np.isfinite(turn_acf)], np.array([0.0])])
    ymin, ymax = float(min(0.0, all_vals.min())), float(max(0.05, all_vals.max()))
    pad = (ymax - ymin) * 0.12
    ymin, ymax = ymin - pad, ymax + pad

    w, h = 980, 500
    left, right, top, bottom = 76, 28, 54, 64
    plot_w, plot_h = w - left - right, h - top - bottom

    def xmap(x: float) -> float:
        return left + (x - 1) / 19 * plot_w

    def ymap(y: float) -> float:
        return top + plot_h - (y - ymin) / (ymax - ymin) * plot_h

    def polyline(vals: np.ndarray, color: str) -> str:
        pts = " ".join(f"{xmap(float(l)):.1f},{ymap(float(v)):.1f}" for l, v in zip(lags, vals) if np.isfinite(v))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>'

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "Absolute-return autocorrelation by lag", 18, "middle", "700"),
        f'<line x1="{left}" y1="{ymap(0):.1f}" x2="{left + plot_w}" y2="{ymap(0):.1f}" stroke="#bdc3c7"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#2c3e50"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#2c3e50"/>',
        polyline(time_acf, "#2f80ed"),
        polyline(turn_acf, "#d35400"),
    ]
    for lag in [1, 5, 10, 15, 20]:
        x = xmap(float(lag))
        elems.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#2c3e50"/>')
        elems.append(svg_text(x, top + plot_h + 24, lag, 11, "middle"))
    for yv in np.linspace(ymin, ymax, 5):
        y = ymap(float(yv))
        elems.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#2c3e50"/>')
        elems.append(svg_text(left - 10, y + 4, f"{yv:.2f}", 11, "end"))
    elems.append(svg_text(left + plot_w / 2, h - 20, "lag", 12, "middle"))
    elems.append(f'<rect x="{left + 18}" y="{top + 12}" width="16" height="16" fill="#2f80ed" opacity="0.85"/>')
    elems.append(svg_text(left + 42, top + 25, "5-minute time bars", 12))
    elems.append(f'<rect x="{left + 190}" y="{top + 12}" width="16" height="16" fill="#d35400" opacity="0.85"/>')
    elems.append(svg_text(left + 214, top + 25, "turnover bars", 12))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def write_field_dictionary(path: Path) -> None:
    rows = ["# Field dictionary", "", "| Column | Meaning |", "|---|---|"]
    for column, meaning in FIELD_DICTIONARY:
        rows.append(f"| `{column}` | {meaning} |")
    rows.append("")
    rows.append("The A-share `stock_list.csv` found in the workspace has columns:")
    rows.append("")
    rows.append("| Column | Meaning |")
    rows.append("|---|---|")
    rows.append("| `ts_code` | Tushare stock code, including exchange suffix, such as `000001.SZ`. |")
    rows.append("| `symbol` | Six-digit stock ticker. |")
    rows.append("| `name` | Stock short name. |")
    rows.append("| `list_date` | Listing date in YYYYMMDD format. |")
    path.write_text("\n".join(rows), encoding="utf-8")


def write_report(
    path: Path,
    files: list[Path],
    minutes: pd.DataFrame,
    time_bars: pd.DataFrame,
    turnover_bars: pd.DataFrame,
    threshold: float,
    stats: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# HS300 1-minute resampling prototype",
        "",
        "## Data",
        "",
        f"- Files: {', '.join(p.name for p in files)}",
        f"- Sample window: {args.start} to {args.end}",
        f"- Minute rows after cleaning: {len(minutes):,}",
        f"- Trading days: {minutes['trade_date'].dt.date.nunique()}",
        "",
        "## Method",
        "",
        f"- Time bars: every {args.time_step} minute rows within each trading day.",
        f"- Turnover bars: close a bar when accumulated turnover reaches {threshold:,.0f}.",
        f"- Threshold rule: median daily turnover / {args.target_bars_per_day} target bars per day.",
        "- Overnight bars are not allowed; both methods reset at each trading day.",
        "",
        "## Key diagnostics",
        "",
        markdown_table(stats),
        "",
        "## Quick read",
        "",
    ]
    time_row = stats[stats["sample"].str.contains("time")].iloc[0]
    turn_row = stats[stats["sample"].eq("turnover")].iloc[0]
    checks = [
        ("turnover_cv", "turnover per bar is more even"),
        ("abs_return_acf1", "absolute-return autocorrelation is lower"),
        ("jarque_bera_per_return", "return distribution is closer to normal by JB/return"),
    ]
    for column, label in checks:
        better = float(turn_row[column]) < float(time_row[column])
        verdict = "yes" if better else "no"
        lines.append(f"- {label}: {verdict} ({float(turn_row[column]):.6g} vs {float(time_row[column]):.6g}).")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `field_dictionary.md`",
            "- `diagnostics.csv`",
            "- `time_bars.csv`",
            "- `turnover_bars.csv`",
            "- `return_histogram.svg`",
            "- `diagnostic_ratios.svg`",
            "- `abs_return_acf.svg`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    minutes, files = load_minutes(data_dir, args.start, args.end)
    daily_turnover = minutes.groupby(minutes["trade_date"].dt.date)["turnover"].sum()
    threshold = (
        float(args.turnover_threshold)
        if args.turnover_threshold is not None
        else float(daily_turnover.median() / args.target_bars_per_day)
    )

    time_bars = make_time_bars(minutes, args.time_step)
    turnover_bars = make_turnover_bars(minutes, threshold)
    stats = pd.DataFrame(
        [
            summarize_bars(f"{args.time_step}min_time", time_bars),
            summarize_bars("turnover", turnover_bars),
        ]
    )

    write_field_dictionary(output_dir / "field_dictionary.md")
    minutes.to_csv(output_dir / "clean_minutes_sample.csv", index=False)
    time_bars.to_csv(output_dir / "time_bars.csv", index=False)
    turnover_bars.to_csv(output_dir / "turnover_bars.csv", index=False)
    stats.to_csv(output_dir / "diagnostics.csv", index=False)

    time_returns = returns_from_bars(time_bars)
    turnover_returns = returns_from_bars(turnover_bars)
    save_return_histogram(time_returns, turnover_returns, output_dir / "return_histogram.svg")
    save_metric_ratios(stats, output_dir / "diagnostic_ratios.svg")
    save_abs_return_acf(time_returns, turnover_returns, output_dir / "abs_return_acf.svg")
    write_report(output_dir / "resampling_report.md", files, minutes, time_bars, turnover_bars, threshold, stats, args)

    print(f"Loaded {len(minutes):,} minute rows from {len(files)} files.")
    print(f"Turnover threshold: {threshold:,.0f}")
    print(stats.to_string(index=False))
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
