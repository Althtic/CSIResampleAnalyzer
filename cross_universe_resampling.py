#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_resampling import make_time_bars, make_turnover_bars, markdown_table


UNIVERSES = {
    "hs300": {
        "label": "HS300",
        "data_dir": "hs300_minute_20200101_20240618",
        "prefix": "hs300",
        "color": "#2563eb",
    },
    "csi500": {
        "label": "CSI500",
        "data_dir": "csi500_minute_20200101_20240618",
        "prefix": "csi500",
        "color": "#d97706",
    },
    "csi1000": {
        "label": "CSI1000",
        "data_dir": "csi1000_minute_20200101_20240618",
        "prefix": "csi1000",
        "color": "#16a34a",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-universe fixed-time vs turnover-resampled K-line comparison."
    )
    parser.add_argument("--start", default="2024-04-01")
    parser.add_argument("--end", default="2024-06-18")
    parser.add_argument("--output-dir", default="cross_universe_resampling_results")
    parser.add_argument("--time-step", type=int, default=5)
    parser.add_argument("--target-bars-per-day", type=int, default=48)
    parser.add_argument("--plot-date", default="2024-04-01")
    return parser.parse_args()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(x: float, y: float, text: object, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#17202a">{esc(text)}</text>'
    )


def normal_cdf(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(values / math.sqrt(2.0)))


def selected_files(data_dir: Path, prefix: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    files = []
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{4}})-(\d{{2}})_1min(?:_mtd)?\.xlsx$")
    for path in sorted(data_dir.glob(f"{prefix}_????-??_1min*.xlsx")):
        match = pattern.match(path.name)
        if not match:
            continue
        month_start = pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=1)
        month_end = month_start + pd.offsets.MonthEnd(0)
        if month_start <= end and month_end >= start:
            files.append(path)
    return files


def load_universe_minutes(universe_key: str, start: str, end: str) -> tuple[pd.DataFrame, list[Path]]:
    spec = UNIVERSES[universe_key]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    files = selected_files(Path(spec["data_dir"]), spec["prefix"], start_ts, end_ts)
    if not files:
        raise FileNotFoundError(f"No files for {universe_key} in {spec['data_dir']}")

    frames = []
    for path in files:
        raw = pd.read_excel(path)
        raw["source_file"] = path.name
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)
    dt_utc = pd.to_datetime(raw["TIME"].astype(str), errors="coerce", utc=True)
    data = pd.DataFrame(
        {
            "universe": universe_key,
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
    data = data.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    data["turnover"] = data["turnover"].clip(lower=0.0)
    data["volume"] = data["volume"].clip(lower=0.0)
    return data, files


def filter_common_dates(minutes_by_universe: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    date_sets = [
        set(frame["trade_date"].dt.date.astype(str).unique())
        for frame in minutes_by_universe.values()
    ]
    common_dates = sorted(set.intersection(*date_sets))
    filtered = {}
    for key, frame in minutes_by_universe.items():
        filtered[key] = frame[frame["trade_date"].dt.date.astype(str).isin(common_dates)].reset_index(drop=True)
    return filtered, common_dates


def bar_returns(bars: pd.DataFrame, kind: str) -> pd.DataFrame:
    pieces = []
    for _, day in bars.groupby("trade_date", sort=True):
        day = day.sort_values("end_time").reset_index(drop=True)
        close = day["close"].astype(float).to_numpy()
        if len(close) < 2:
            continue
        pieces.append(
            pd.DataFrame(
                {
                    "sample": kind,
                    "trade_date": day["trade_date"].iloc[1:].to_numpy(),
                    "simple_return": close[1:] / close[:-1] - 1.0,
                    "log_return": np.diff(np.log(close)),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


def autocorr(values: np.ndarray, lag: int = 1) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= lag + 2:
        return float("nan")
    a, b = values[:-lag], values[lag:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def distribution_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    z = (values - mean) / std
    skew = float(np.mean(z**3))
    excess = float(np.mean(z**4) - 3.0)
    jb = float(n / 6.0 * (skew**2 + excess**2 / 4.0))
    sorted_z = np.sort(z)
    empirical = np.arange(1, n + 1) / n
    ks = float(np.max(np.abs(empirical - normal_cdf(sorted_z))))
    normal_3sigma = 2.0 * (1.0 - 0.9986501019683699)
    tail3 = float(np.mean(np.abs(z) > 3.0))
    return {
        "return_count": int(n),
        "return_mean": mean,
        "return_std": std,
        "return_skew": skew,
        "return_excess_kurtosis": excess,
        "jarque_bera_per_obs": jb / n,
        "ks_distance": ks,
        "tail_prob_abs_gt_3sigma": tail3,
        "tail_ratio_3sigma_vs_normal": tail3 / normal_3sigma,
        "abs_return_acf1": autocorr(np.abs(values), 1),
    }


def summarize_bars(universe: str, sample: str, bars: pd.DataFrame, returns: pd.DataFrame) -> dict[str, float | str]:
    daily_counts = bars.groupby("trade_date").size()
    turnover_mean = float(bars["turnover"].mean())
    stats = distribution_stats(returns["log_return"].to_numpy())
    return {
        "universe": universe,
        "sample": sample,
        "bars": int(len(bars)),
        "trading_days": int(daily_counts.size),
        "bars_per_day_mean": float(daily_counts.mean()),
        "bars_per_day_std": float(daily_counts.std(ddof=0)),
        "minutes_mean": float(bars["minutes"].mean()),
        "minutes_std": float(bars["minutes"].std(ddof=0)),
        "turnover_mean": turnover_mean,
        "turnover_std": float(bars["turnover"].std(ddof=0)),
        "turnover_cv": float(bars["turnover"].std(ddof=0) / turnover_mean),
        **stats,
    }


def build_improvement_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for universe in summary["universe"].unique():
        fixed = summary[(summary["universe"].eq(universe)) & (summary["sample"].eq("fixed_time_5min"))].iloc[0]
        resampled = summary[(summary["universe"].eq(universe)) & (summary["sample"].eq("turnover_resampled"))].iloc[0]
        metrics = [
            ("turnover_cv", "Turnover CV", "lower"),
            ("abs_return_acf1", "Abs return ACF1", "lower"),
            ("return_excess_kurtosis", "Abs excess kurtosis", "abs_lower"),
            ("jarque_bera_per_obs", "JB per obs", "lower"),
            ("ks_distance", "KS distance", "lower"),
            ("tail_ratio_3sigma_vs_normal", "3-sigma tail ratio", "lower"),
        ]
        for column, label, direction in metrics:
            fixed_value = abs(float(fixed[column])) if direction == "abs_lower" else float(fixed[column])
            resampled_value = abs(float(resampled[column])) if direction == "abs_lower" else float(resampled[column])
            ratio = resampled_value / fixed_value if fixed_value != 0 else float("nan")
            rows.append(
                {
                    "universe": universe,
                    "metric": label,
                    "fixed_time_value": fixed_value,
                    "turnover_resampled_value": resampled_value,
                    "resampled_to_fixed_ratio": ratio,
                    "improved": bool(ratio < 1.0),
                }
            )
    return pd.DataFrame(rows)


def save_kline_comparison(
    universe: str,
    label: str,
    fixed_bars: pd.DataFrame,
    resampled_bars: pd.DataFrame,
    plot_date: str,
    path: Path,
) -> None:
    panels = [
        ("FIXED-TIME 5min K bars (not resampled)", fixed_bars[fixed_bars["trade_date"].eq(plot_date)], "#2563eb"),
        ("TURNOVER-RESAMPLED K bars", resampled_bars[resampled_bars["trade_date"].eq(plot_date)], "#d97706"),
    ]
    panels = [(title, data.reset_index(drop=True), color) for title, data, color in panels if not data.empty]
    if len(panels) != 2:
        raise ValueError(f"{universe}: plot date {plot_date} is missing one bar type")

    w, h = 1180, 760
    left, right, top, bottom = 78, 42, 76, 58
    panel_gap = 72
    panel_h = (h - top - bottom - panel_gap) / 2
    plot_w = w - left - right
    prices = pd.concat([data[["open", "high", "low", "close"]] for _, data, _ in panels]).to_numpy(dtype=float)
    ymin, ymax = float(np.nanmin(prices)), float(np.nanmax(prices))
    pad = max((ymax - ymin) * 0.08, 1.0)
    ymin -= pad
    ymax += pad

    def ymap(value: float, panel_top: float) -> float:
        return panel_top + panel_h - (value - ymin) / (ymax - ymin) * panel_h

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, f"{label}: fixed-time vs TURNOVER-RESAMPLED K lines on {plot_date}", 19, "middle", "700"),
        svg_text(w / 2, 53, "Blue panel is not resampled; orange panel is turnover-resampled.", 12, "middle"),
    ]
    for panel_idx, (title, data, accent) in enumerate(panels):
        panel_top = top + panel_idx * (panel_h + panel_gap)
        elems.append(svg_text(left, panel_top - 16, title, 15, "start", "700"))
        elems.append(f'<rect x="{left}" y="{panel_top:.1f}" width="{plot_w}" height="{panel_h:.1f}" fill="#ffffff" stroke="#d7dee8"/>')
        for tick in np.linspace(ymin, ymax, 5):
            y = ymap(float(tick), panel_top)
            elems.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#edf1f5"/>')
            elems.append(svg_text(left - 10, y + 4, f"{tick:.1f}", 11, "end"))
        slot = plot_w / max(len(data), 1)
        candle_w = max(3.0, min(10.0, slot * 0.58))
        for idx, row in data.iterrows():
            x = left + slot * (idx + 0.5)
            o, hi, lo, c = (float(row[col]) for col in ["open", "high", "low", "close"])
            up = c >= o
            fill = "#ffffff" if up else accent
            stroke = "#16a34a" if up else accent
            y_hi, y_lo = ymap(hi, panel_top), ymap(lo, panel_top)
            y_o, y_c = ymap(o, panel_top), ymap(c, panel_top)
            elems.append(f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="{stroke}" stroke-width="1.2"/>')
            elems.append(
                f'<rect x="{x - candle_w / 2:.1f}" y="{min(y_o, y_c):.1f}" width="{candle_w:.1f}" '
                f'height="{max(abs(y_c - y_o), 1.2):.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
            )
        elems.append(svg_text(left + plot_w / 2, panel_top + panel_h + 26, f"{len(data)} bars", 11, "middle"))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def save_improvement_chart(improvements: pd.DataFrame, path: Path) -> None:
    metrics = [
        "Turnover CV",
        "Abs return ACF1",
        "Abs excess kurtosis",
        "JB per obs",
        "KS distance",
        "3-sigma tail ratio",
    ]
    universes = ["hs300", "csi500", "csi1000"]
    colors = {"hs300": "#2563eb", "csi500": "#d97706", "csi1000": "#16a34a"}
    labels = {key: UNIVERSES[key]["label"] for key in universes}
    w, h = 1180, 650
    left, right, top, bottom = 210, 36, 74, 70
    row_h = (h - top - bottom) / len(metrics)
    plot_w = w - left - right
    xmax = max(1.4, float(improvements["resampled_to_fixed_ratio"].replace([np.inf, -np.inf], np.nan).max()) * 1.12)

    def xmap(value: float) -> float:
        return left + value / xmax * plot_w

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "Cross-universe turnover-resampling benefit ratios", 19, "middle", "700"),
        svg_text(w / 2, 52, "Each bar is TURNOVER-RESAMPLED / fixed-time. Values below 1.0 indicate improvement from resampling.", 12, "middle"),
        f'<line x1="{xmap(1.0):.1f}" y1="{top - 12}" x2="{xmap(1.0):.1f}" y2="{h - bottom + 10}" stroke="#7f8c8d" stroke-dasharray="4,4"/>',
    ]
    for idx, metric in enumerate(metrics):
        y_base = top + idx * row_h
        elems.append(svg_text(22, y_base + row_h / 2 + 4, metric, 12))
        for j, universe in enumerate(universes):
            row = improvements[(improvements["universe"].eq(universe)) & (improvements["metric"].eq(metric))].iloc[0]
            ratio = float(row["resampled_to_fixed_ratio"])
            y = y_base + 12 + j * 16
            elems.append(
                f'<rect x="{left}" y="{y:.1f}" width="{max(1, xmap(ratio) - left):.1f}" height="11" '
                f'fill="{colors[universe]}" opacity="0.78"/>'
            )
            elems.append(svg_text(xmap(ratio) + 5, y + 10, f"{ratio:.2f}", 10))
    for idx, universe in enumerate(universes):
        x = left + idx * 150
        elems.append(f'<rect x="{x}" y="{h - 34}" width="14" height="14" fill="{colors[universe]}" opacity="0.78"/>')
        elems.append(svg_text(x + 20, h - 22, labels[universe], 12))
    for tick in np.linspace(0, xmax, 6):
        x = xmap(float(tick))
        elems.append(f'<line x1="{x:.1f}" y1="{h - bottom + 12}" x2="{x:.1f}" y2="{h - bottom + 18}" stroke="#2c3e50"/>')
        elems.append(svg_text(x, h - bottom + 36, f"{tick:.2f}", 10, "middle"))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    common_dates: list[str],
    thresholds: dict[str, float],
    files_by_universe: dict[str, list[Path]],
    summary: pd.DataFrame,
    improvements: pd.DataFrame,
) -> None:
    pass_rate = (
        improvements.groupby("metric")["improved"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "universes_improved", "count": "universes_total"})
    )
    lines = [
        "# Cross-Universe Turnover Resampling Report",
        "",
        "## Scope",
        "",
        f"- Universes: HS300, CSI500, CSI1000.",
        f"- Common date window: `{common_dates[0]}` to `{common_dates[-1]}`.",
        f"- Trading days in common: `{len(common_dates)}`.",
        f"- Fixed-time bars: `{args.time_step}` minute bars.",
        f"- Turnover-resampled threshold rule: each universe uses its own median daily turnover / `{args.target_bars_per_day}`.",
        "- This keeps the target bar count comparable while respecting different turnover scales across universes.",
        "",
        "## Thresholds",
        "",
        "| Universe | Turnover threshold | Files |",
        "|---|---:|---|",
    ]
    for key, spec in UNIVERSES.items():
        lines.append(
            f"| {spec['label']} | {thresholds[key]:,.0f} | "
            f"{', '.join(path.name for path in files_by_universe[key])} |"
        )
    lines.extend(
        [
            "",
            "## Summary Statistics",
            "",
            markdown_table(summary),
            "",
            "## Resampling Improvement Ratios",
            "",
            "Ratios are `turnover-resampled / fixed-time`; below 1.0 means resampling improved that metric.",
            "",
            markdown_table(improvements),
            "",
            "## Universality Check",
            "",
            markdown_table(pass_rate),
            "",
            "## Quick Read",
            "",
        ]
    )
    for _, row in pass_rate.iterrows():
        lines.append(
            f"- {row['metric']}: improved in `{int(row['universes_improved'])}/{int(row['universes_total'])}` universes."
        )
    lines.extend(
        [
            "",
            "Overall: turnover resampling delivers a very robust improvement in bar-level turnover uniformity across all three universes. "
            "In this common window, the normality and tail metrics also improve in all three universes. "
            "The only exception among the selected diagnostics is absolute-return ACF1, where CSI1000 becomes slightly worse after resampling.",
            "",
            "## Figures",
            "",
            "![Cross-universe improvement ratios](cross_universe_improvement_ratios.svg)",
            "",
        ]
    )
    for key, spec in UNIVERSES.items():
        lines.extend(
            [
                f"### {spec['label']} K-line Comparison",
                "",
                f"![{spec['label']} K-line comparison](kline_comparison_{key}_{args.plot_date}.svg)",
                "",
            ]
        )
    lines.extend(
        [
            "## Generated Files",
            "",
            "- `cross_universe_summary.csv`",
            "- `cross_universe_improvement.csv`",
            "- `cross_universe_improvement_ratios.svg`",
            "- `kline_comparison_<universe>_<date>.svg`",
            "- `<universe>_fixed_time_bars.csv`",
            "- `<universe>_turnover_resampled_bars.csv`",
        ]
    )
    (output_dir / "cross_universe_resampling_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    minutes_by_universe = {}
    files_by_universe = {}
    for universe in UNIVERSES:
        minutes, files = load_universe_minutes(universe, args.start, args.end)
        minutes_by_universe[universe] = minutes
        files_by_universe[universe] = files
    minutes_by_universe, common_dates = filter_common_dates(minutes_by_universe)

    summaries = []
    thresholds = {}
    for universe, minutes in minutes_by_universe.items():
        daily_turnover = minutes.groupby(minutes["trade_date"].dt.date)["turnover"].sum()
        threshold = float(daily_turnover.median() / args.target_bars_per_day)
        thresholds[universe] = threshold

        fixed_bars = make_time_bars(minutes, args.time_step)
        fixed_bars["universe"] = universe
        fixed_bars["sample"] = "fixed_time_5min"
        turnover_bars = make_turnover_bars(minutes, threshold)
        turnover_bars["universe"] = universe
        turnover_bars["sample"] = "turnover_resampled"

        fixed_returns = bar_returns(fixed_bars, "fixed_time_5min")
        turnover_returns = bar_returns(turnover_bars, "turnover_resampled")
        summaries.append(summarize_bars(universe, "fixed_time_5min", fixed_bars, fixed_returns))
        summaries.append(summarize_bars(universe, "turnover_resampled", turnover_bars, turnover_returns))

        fixed_bars.to_csv(output_dir / f"{universe}_fixed_time_bars.csv", index=False)
        turnover_bars.to_csv(output_dir / f"{universe}_turnover_resampled_bars.csv", index=False)
        fixed_returns.to_csv(output_dir / f"{universe}_fixed_time_returns.csv", index=False)
        turnover_returns.to_csv(output_dir / f"{universe}_turnover_resampled_returns.csv", index=False)

        plot_date = args.plot_date if args.plot_date in common_dates else common_dates[0]
        save_kline_comparison(
            universe,
            UNIVERSES[universe]["label"],
            fixed_bars,
            turnover_bars,
            plot_date,
            output_dir / f"kline_comparison_{universe}_{plot_date}.svg",
        )

    summary = pd.DataFrame(summaries)
    improvements = build_improvement_table(summary)
    summary.to_csv(output_dir / "cross_universe_summary.csv", index=False)
    improvements.to_csv(output_dir / "cross_universe_improvement.csv", index=False)
    save_improvement_chart(improvements, output_dir / "cross_universe_improvement_ratios.svg")
    write_report(output_dir, args, common_dates, thresholds, files_by_universe, summary, improvements)

    print(f"Common dates: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} trading days)")
    print(summary.to_string(index=False))
    print(f"Report: {(output_dir / 'cross_universe_resampling_report.md').resolve()}")


if __name__ == "__main__":
    main()
