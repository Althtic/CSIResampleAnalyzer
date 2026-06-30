#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_resampling import load_minutes, make_time_bars, make_turnover_bars, markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare conditional probability models on fixed-time bars and turnover-resampled bars."
    )
    parser.add_argument("--data-dir", default="hs300_minute_20200101_20240618")
    parser.add_argument("--output-dir", default="conditional_probability_results")
    parser.add_argument("--start", default="2024-05-01")
    parser.add_argument("--end", default="2024-05-31")
    parser.add_argument("--time-step", type=int, default=5)
    parser.add_argument("--target-bars-per-day", type=int, default=48)
    parser.add_argument("--turnover-threshold", type=float, default=None)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--grid-size", type=int, default=501)
    return parser.parse_args()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(x: float, y: float, text: object, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#17202a">{esc(text)}</text>'
    )


def normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def prepare_model_data(bars: pd.DataFrame, label: str, vol_window: int) -> pd.DataFrame:
    pieces = []
    bars_per_day = bars.groupby("trade_date").size().mean()
    for _, day in bars.groupby("trade_date", sort=True):
        day = day.sort_values("end_time").reset_index(drop=True)
        close = day["close"].astype(float)
        simple_return = close.pct_change()
        log_return = np.log(close).diff()
        part = pd.DataFrame(
            {
                "sample": label,
                "trade_date": day["trade_date"].astype(str),
                "timestamp": pd.to_datetime(day["end_time"]),
                "simple_return": simple_return,
                "log_return": log_return,
            }
        ).dropna()
        pieces.append(part)

    data = pd.concat(pieces, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    data["rolling_vol"] = data["log_return"].rolling(vol_window, min_periods=vol_window).std() * math.sqrt(bars_per_day)
    data["target_log_return"] = data.groupby("trade_date")["log_return"].shift(-1)
    data["target_simple_return"] = data.groupby("trade_date")["simple_return"].shift(-1)
    data = data.dropna(subset=["rolling_vol", "target_log_return", "target_simple_return"]).reset_index(drop=True)
    data["vol_bps"] = data["rolling_vol"] * 10000.0
    data["target_return_bps"] = data["target_log_return"] * 10000.0
    return data


@dataclass
class ConditionalKDE:
    returns_bps: np.ndarray
    vol_bps: np.ndarray
    h_return: float
    h_vol: float
    return_grid: np.ndarray

    @classmethod
    def fit(cls, train: pd.DataFrame, grid_size: int) -> "ConditionalKDE":
        returns = train["target_return_bps"].to_numpy(dtype=float)
        vol = train["vol_bps"].to_numpy(dtype=float)
        n = len(train)
        h_return = bandwidth(returns, n)
        h_vol = bandwidth(vol, n)
        lo, hi = np.quantile(returns, [0.001, 0.999])
        pad = max((hi - lo) * 0.35, h_return * 6.0)
        grid = np.linspace(float(lo - pad), float(hi + pad), grid_size)
        return cls(returns, vol, h_return, h_vol, grid)

    def weights(self, vol_value: float) -> np.ndarray:
        z = (self.vol_bps - vol_value) / self.h_vol
        weights = np.exp(-0.5 * z * z)
        if not np.isfinite(weights).all() or weights.sum() <= 1e-14:
            weights = np.ones_like(self.vol_bps)
        return weights

    def density_at(self, return_value: float, vol_value: float) -> float:
        weights = self.weights(vol_value)
        z = (return_value - self.returns_bps) / self.h_return
        density = np.sum(weights * normal_pdf(z)) / (weights.sum() * self.h_return)
        return float(max(density, 1e-300))

    def cdf_at(self, return_value: float, vol_value: float) -> float:
        weights = self.weights(vol_value)
        z = (return_value - self.returns_bps) / self.h_return
        cdf = np.sum(weights * normal_cdf(z)) / weights.sum()
        return float(np.clip(cdf, 0.0, 1.0))

    def density_grid(self, vol_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = self.weights(vol_value)
        z = (self.return_grid[:, None] - self.returns_bps[None, :]) / self.h_return
        pdf = np.sum(normal_pdf(z) * weights[None, :], axis=1) / (weights.sum() * self.h_return)
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        area = np.trapezoid(pdf, self.return_grid)
        if np.isfinite(area) and area > 0:
            pdf = pdf / area
        cdf = np.cumsum((pdf[:-1] + pdf[1:]) * np.diff(self.return_grid) / 2.0)
        cdf = np.r_[0.0, cdf]
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        return self.return_grid, pdf, cdf


def bandwidth(values: np.ndarray, n: int) -> float:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values, ddof=1))
    q75, q25 = np.quantile(values, [0.75, 0.25])
    robust = float((q75 - q25) / 1.349)
    scale = min(std, robust) if robust > 0 else std
    if scale <= 0 or not np.isfinite(scale):
        scale = max(float(np.mean(np.abs(values))), 1.0)
    return max(1.06 * scale * (n ** (-1.0 / 5.0)), 1e-6)


def train_test_dates(minutes: pd.DataFrame, train_ratio: float) -> tuple[list[str], list[str]]:
    dates = sorted(minutes["trade_date"].dt.date.astype(str).unique())
    split = int(len(dates) * train_ratio)
    split = min(max(split, 5), len(dates) - 3)
    return dates[:split], dates[split:]


def evaluate_model(model: ConditionalKDE, train: pd.DataFrame, test: pd.DataFrame, sample: str) -> tuple[dict[str, float | str], pd.DataFrame]:
    observed = test["target_return_bps"].to_numpy(dtype=float)
    vol = test["vol_bps"].to_numpy(dtype=float)
    log_density = np.array([math.log(model.density_at(r, v)) for r, v in zip(observed, vol)])
    pit = np.array([model.cdf_at(r, v) for r, v in zip(observed, vol)])
    sorted_pit = np.sort(pit)
    n = len(pit)
    pit_ks = float(np.max(np.abs(sorted_pit - (np.arange(1, n + 1) - 0.5) / n)))
    tail_cut = float(np.quantile(train["target_return_bps"], 0.10))

    q05, q95, width, prob_up, prob_tail, crps = [], [], [], [], [], []
    for r, v in zip(observed, vol):
        grid, _, cdf = model.density_grid(v)
        low = float(np.interp(0.05, cdf, grid))
        high = float(np.interp(0.95, cdf, grid))
        p_up = 1.0 - float(np.interp(0.0, grid, cdf))
        p_tail = float(np.interp(tail_cut, grid, cdf))
        indicator = (grid >= r).astype(float)
        crps_value = float(np.trapezoid((cdf - indicator) ** 2, grid))
        q05.append(low)
        q95.append(high)
        width.append(high - low)
        prob_up.append(p_up)
        prob_tail.append(p_tail)
        crps.append(crps_value)

    q05_arr = np.array(q05)
    q95_arr = np.array(q95)
    prob_up_arr = np.array(prob_up)
    prob_tail_arr = np.array(prob_tail)
    up_actual = (observed > 0).astype(float)
    tail_actual = (observed <= tail_cut).astype(float)
    diagnostics = test[["sample", "trade_date", "timestamp", "vol_bps", "target_return_bps"]].copy()
    diagnostics["pit"] = pit
    diagnostics["log_density"] = log_density
    diagnostics["q05"] = q05_arr
    diagnostics["q95"] = q95_arr
    diagnostics["prob_up"] = prob_up_arr
    diagnostics["prob_left_tail_10pct"] = prob_tail_arr

    metrics = {
        "sample": sample,
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "avg_log_score": float(np.mean(log_density)),
        "median_log_score": float(np.median(log_density)),
        "crps_bps": float(np.mean(crps)),
        "pit_ks": pit_ks,
        "pit_mean": float(np.mean(pit)),
        "pit_std": float(np.std(pit, ddof=1)),
        "interval90_coverage": float(np.mean((observed >= q05_arr) & (observed <= q95_arr))),
        "interval90_abs_error": float(abs(np.mean((observed >= q05_arr) & (observed <= q95_arr)) - 0.90)),
        "interval90_avg_width_bps": float(np.mean(width)),
        "direction_accuracy": float(np.mean((prob_up_arr >= 0.5) == (up_actual > 0))),
        "direction_brier": float(np.mean((prob_up_arr - up_actual) ** 2)),
        "left_tail_brier": float(np.mean((prob_tail_arr - tail_actual) ** 2)),
        "left_tail_realized_rate": float(np.mean(tail_actual)),
        "left_tail_avg_prob": float(np.mean(prob_tail_arr)),
    }
    return metrics, diagnostics


def save_conditional_slices(models: dict[str, ConditionalKDE], train_sets: dict[str, pd.DataFrame], path: Path) -> None:
    w, h = 1180, 520
    left, right, top, bottom = 72, 32, 72, 58
    gap = 68
    panel_w = (w - left - right - gap) / 2
    panel_h = h - top - bottom
    colors = ["#2563eb", "#d97706", "#16a34a"]
    labels = {"time_5min": "FIXED-TIME 5-minute bars (not resampled)", "turnover": "TURNOVER-RESAMPLED bars"}
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "Conditional density slices: fixed-time vs TURNOVER-RESAMPLED", 19, "middle", "700"),
        svg_text(w / 2, 52, "Each curve fixes volatility at the train-set 25%, 50%, or 75% quantile.", 12, "middle"),
    ]
    all_density_max = 0.0
    cache: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {}
    for name, model in models.items():
        vols = np.quantile(train_sets[name]["vol_bps"], [0.25, 0.50, 0.75])
        curves = []
        for vol in vols:
            grid, pdf, _ = model.density_grid(float(vol))
            curves.append((float(vol), grid, pdf))
            all_density_max = max(all_density_max, float(np.max(pdf)))
        cache[name] = curves
    all_x = np.concatenate([curve[1] for curves in cache.values() for curve in curves])
    xmin, xmax = float(np.quantile(all_x, 0.01)), float(np.quantile(all_x, 0.99))
    ymax = all_density_max * 1.12

    for idx, name in enumerate(["time_5min", "turnover"]):
        x0 = left + idx * (panel_w + gap)
        y0 = top

        def xmap(value: float) -> float:
            return x0 + (value - xmin) / (xmax - xmin) * panel_w

        def ymap(value: float) -> float:
            return y0 + panel_h - value / ymax * panel_h

        elems.append(svg_text(x0, y0 - 16, labels[name], 15, "start", "700"))
        elems.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" fill="#ffffff" stroke="#d7dee8"/>')
        for tick in np.linspace(0, ymax, 5):
            y = ymap(float(tick))
            elems.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + panel_w:.1f}" y2="{y:.1f}" stroke="#edf1f5"/>')
        for color, (vol, grid, pdf) in zip(colors, cache[name]):
            mask = (grid >= xmin) & (grid <= xmax)
            points = " ".join(f"{xmap(float(x)):.1f},{ymap(float(y)):.1f}" for x, y in zip(grid[mask], pdf[mask]))
            elems.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"/>')
            legend_y = y0 + 18 + colors.index(color) * 20
            elems.append(f'<line x1="{x0 + 18:.1f}" y1="{legend_y:.1f}" x2="{x0 + 48:.1f}" y2="{legend_y:.1f}" stroke="{color}" stroke-width="2.4"/>')
            elems.append(svg_text(x0 + 56, legend_y + 4, f"vol={vol:.2f} bps", 11))
        for tick in np.linspace(xmin, xmax, 5):
            x = xmap(float(tick))
            elems.append(f'<line x1="{x:.1f}" y1="{y0 + panel_h:.1f}" x2="{x:.1f}" y2="{y0 + panel_h + 5:.1f}" stroke="#2c3e50"/>')
            elems.append(svg_text(x, y0 + panel_h + 22, f"{tick:.1f}", 10, "middle"))
        elems.append(svg_text(x0 + panel_w / 2, y0 + panel_h + 43, "next log return, bps", 11, "middle"))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def save_pit_histogram(diagnostics: pd.DataFrame, path: Path) -> None:
    w, h = 980, 430
    left, right, top, bottom = 70, 28, 62, 58
    gap = 58
    panel_w = (w - left - right - gap) / 2
    panel_h = h - top - bottom
    labels = {"time_5min": "FIXED-TIME 5-minute bars (not resampled)", "turnover": "TURNOVER-RESAMPLED bars"}
    colors = {"time_5min": "#2563eb", "turnover": "#d97706"}
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "PIT calibration histogram: fixed-time vs TURNOVER-RESAMPLED", 18, "middle", "700"),
        svg_text(w / 2, 50, "A well-calibrated conditional distribution should be close to uniform.", 12, "middle"),
    ]
    bins = np.linspace(0.0, 1.0, 11)
    hist_cache = {}
    ymax = 0.0
    for name in ["time_5min", "turnover"]:
        hist, _ = np.histogram(diagnostics.loc[diagnostics["sample"].eq(name), "pit"], bins=bins, density=True)
        hist_cache[name] = hist
        ymax = max(ymax, float(hist.max()), 1.0)
    ymax *= 1.15
    for idx, name in enumerate(["time_5min", "turnover"]):
        x0 = left + idx * (panel_w + gap)
        y0 = top

        def xmap(value: float) -> float:
            return x0 + value * panel_w

        def ymap(value: float) -> float:
            return y0 + panel_h - value / ymax * panel_h

        elems.append(svg_text(x0, y0 - 16, labels[name], 14, "start", "700"))
        elems.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" fill="#ffffff" stroke="#d7dee8"/>')
        elems.append(f'<line x1="{x0:.1f}" y1="{ymap(1.0):.1f}" x2="{x0 + panel_w:.1f}" y2="{ymap(1.0):.1f}" stroke="#111827" stroke-dasharray="4,4"/>')
        for i, value in enumerate(hist_cache[name]):
            bx0, bx1 = xmap(float(bins[i])), xmap(float(bins[i + 1]))
            y = ymap(float(value))
            elems.append(f'<rect x="{bx0 + 2:.1f}" y="{y:.1f}" width="{max(1, bx1 - bx0 - 4):.1f}" height="{y0 + panel_h - y:.1f}" fill="{colors[name]}" opacity="0.68"/>')
        for tick in np.linspace(0, 1, 6):
            x = xmap(float(tick))
            elems.append(f'<line x1="{x:.1f}" y1="{y0 + panel_h:.1f}" x2="{x:.1f}" y2="{y0 + panel_h + 5:.1f}" stroke="#2c3e50"/>')
            elems.append(svg_text(x, y0 + panel_h + 22, f"{tick:.1f}", 10, "middle"))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def save_metric_comparison(metrics: pd.DataFrame, path: Path) -> None:
    time_row = metrics[metrics["sample"].eq("time_5min")].iloc[0]
    turn_row = metrics[metrics["sample"].eq("turnover")].iloc[0]
    rows = [
        ("Avg log score", "avg_log_score", False),
        ("CRPS", "crps_bps", True),
        ("PIT KS", "pit_ks", True),
        ("90% coverage error", "interval90_abs_error", True),
        ("Direction Brier", "direction_brier", True),
        ("Left-tail Brier", "left_tail_brier", True),
    ]
    ratios = []
    for label, column, lower_better in rows:
        time_value = float(time_row[column])
        turn_value = float(turn_row[column])
        ratio = turn_value / time_value if lower_better else time_value / turn_value
        ratios.append((label, ratio, turn_value, time_value))
    w, h = 1060, 460
    left, right, top = 220, 230, 64
    row_h = 56
    plot_w = w - left - right
    xmax = max(1.3, max(r[1] for r in ratios if np.isfinite(r[1])) * 1.15)

    def xmap(value: float) -> float:
        return left + value / xmax * plot_w

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "Conditional probability model metrics", 18, "middle", "700"),
        svg_text(w / 2, 50, "Bars below 1.0 favor TURNOVER-RESAMPLED bars. For log score, ratio is fixed-time / resampled.", 12, "middle"),
        f'<line x1="{xmap(1.0):.1f}" y1="{top - 8}" x2="{xmap(1.0):.1f}" y2="{top + row_h * len(rows)}" stroke="#7f8c8d" stroke-dasharray="4,4"/>',
    ]
    for idx, (label, ratio, turn_value, time_value) in enumerate(ratios):
        y = top + idx * row_h
        color = "#1f9d55" if ratio < 1.0 else "#c0392b"
        elems.append(svg_text(24, y + 28, label, 12))
        elems.append(f'<rect x="{left}" y="{y + 10}" width="{max(1, xmap(ratio) - left):.1f}" height="24" rx="3" fill="{color}" opacity="0.78"/>')
        elems.append(svg_text(xmap(ratio) + 8, y + 28, f"{ratio:.2f}x", 12))
        elems.append(svg_text(w - 18, y + 28, f"resampled {turn_value:.4g} | fixed-time {time_value:.4g}", 11, "end"))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def write_report(
    path: Path,
    args: argparse.Namespace,
    files: list[Path],
    threshold: float,
    train_dates: list[str],
    test_dates: list[str],
    metrics: pd.DataFrame,
    bars_summary: pd.DataFrame,
) -> None:
    time_row = metrics[metrics["sample"].eq("time_5min")].iloc[0]
    turn_row = metrics[metrics["sample"].eq("turnover")].iloc[0]
    winner_lines = []
    checks = [
        ("avg_log_score", "higher", "average log score"),
        ("crps_bps", "lower", "CRPS"),
        ("pit_ks", "lower", "PIT KS distance"),
        ("interval90_abs_error", "lower", "90% interval coverage error"),
        ("direction_brier", "lower", "direction Brier score"),
        ("left_tail_brier", "lower", "left-tail Brier score"),
    ]
    for column, direction, label in checks:
        time_value = float(time_row[column])
        turn_value = float(turn_row[column])
        turnover_better = turn_value > time_value if direction == "higher" else turn_value < time_value
        winner = "turnover-resampled bars" if turnover_better else "fixed-time bars"
        winner_lines.append(
            f"- {label}: {winner} "
            f"(turnover `{turn_value:.6g}` vs fixed-time `{time_value:.6g}`)."
        )

    lines = [
        "# Conditional Probability K-line Experiment",
        "",
        "## Objective",
        "",
        "Use the same conditional probability distribution idea as `test.ipynb`: estimate the joint distribution of return and volatility, then normalize a volatility slice to obtain `f(next return | current volatility)`.",
        "",
        "## Data And Bars",
        "",
        f"- Data window: `{args.start}` to `{args.end}`.",
        f"- Files used: {', '.join(p.name for p in files)}.",
        f"- Fixed-time K line: `{args.time_step}` minute bars.",
        f"- Turnover K line: close a bar when cumulative turnover reaches `{threshold:,.0f}`.",
        f"- Rolling volatility window: `{args.vol_window}` bars, scaled by sqrt(mean bars per day).",
        f"- Train dates: `{train_dates[0]}` to `{train_dates[-1]}`.",
        f"- Test dates: `{test_dates[0]}` to `{test_dates[-1]}`.",
        "",
        "## Bar Sample Summary",
        "",
        markdown_table(bars_summary),
        "",
        "## Modeling Method",
        "",
        "- For each K-line type, compute close-to-close log returns and rolling volatility.",
        "- Use current rolling volatility as the conditioning variable.",
        "- Use the next bar log return as the modeled target.",
        "- Fit a two-dimensional Gaussian-kernel conditional density without look-ahead: training days only.",
        "- Evaluate all metrics on the same test trading days.",
        "",
        "## Evaluation Metrics",
        "",
        markdown_table(metrics),
        "",
        "Metric interpretation: higher is better for `avg_log_score`; lower is better for `crps_bps`, `pit_ks`, `interval90_abs_error`, `direction_brier`, and `left_tail_brier`.",
        "",
        "## Quick Read",
        "",
        *winner_lines,
        "",
        "## Figures",
        "",
        "![Conditional density slices](conditional_density_slices.svg)",
        "",
        "![PIT histogram](conditional_pit_histogram.svg)",
        "",
        "![Metric comparison](conditional_metric_comparison.svg)",
        "",
        "## Files Generated",
        "",
        "- `conditional_model_metrics.csv`",
        "- `conditional_model_diagnostics.csv`",
        "- `conditional_bar_summary.csv`",
        "- `conditional_probability_report.md`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    minutes, files = load_minutes(Path(args.data_dir), args.start, args.end)
    daily_turnover = minutes.groupby(minutes["trade_date"].dt.date)["turnover"].sum()
    threshold = (
        float(args.turnover_threshold)
        if args.turnover_threshold is not None
        else float(daily_turnover.median() / args.target_bars_per_day)
    )
    time_bars = make_time_bars(minutes, args.time_step)
    turnover_bars = make_turnover_bars(minutes, threshold)
    train_dates, test_dates = train_test_dates(minutes, args.train_ratio)
    train_set = set(train_dates)
    test_set = set(test_dates)

    datasets = {
        "time_5min": prepare_model_data(time_bars, "time_5min", args.vol_window),
        "turnover": prepare_model_data(turnover_bars, "turnover", args.vol_window),
    }
    train_sets = {name: data[data["trade_date"].isin(train_set)].copy() for name, data in datasets.items()}
    test_sets = {name: data[data["trade_date"].isin(test_set)].copy() for name, data in datasets.items()}
    models = {name: ConditionalKDE.fit(train, args.grid_size) for name, train in train_sets.items()}

    metric_rows = []
    diagnostic_frames = []
    for name in ["time_5min", "turnover"]:
        metrics, diagnostics = evaluate_model(models[name], train_sets[name], test_sets[name], name)
        metric_rows.append(metrics)
        diagnostic_frames.append(diagnostics)
    metrics_df = pd.DataFrame(metric_rows)
    diagnostics_df = pd.concat(diagnostic_frames, ignore_index=True)

    bars_summary = pd.DataFrame(
        [
            {
                "sample": "time_5min",
                "bars": int(len(time_bars)),
                "mean_bars_per_day": float(time_bars.groupby("trade_date").size().mean()),
                "mean_minutes_per_bar": float(time_bars["minutes"].mean()),
                "mean_turnover_per_bar": float(time_bars["turnover"].mean()),
            },
            {
                "sample": "turnover",
                "bars": int(len(turnover_bars)),
                "mean_bars_per_day": float(turnover_bars.groupby("trade_date").size().mean()),
                "mean_minutes_per_bar": float(turnover_bars["minutes"].mean()),
                "mean_turnover_per_bar": float(turnover_bars["turnover"].mean()),
            },
        ]
    )

    metrics_df.to_csv(output_dir / "conditional_model_metrics.csv", index=False)
    diagnostics_df.to_csv(output_dir / "conditional_model_diagnostics.csv", index=False)
    bars_summary.to_csv(output_dir / "conditional_bar_summary.csv", index=False)
    time_bars.to_csv(output_dir / "time_5min_bars.csv", index=False)
    turnover_bars.to_csv(output_dir / "turnover_bars.csv", index=False)

    save_conditional_slices(models, train_sets, output_dir / "conditional_density_slices.svg")
    save_pit_histogram(diagnostics_df, output_dir / "conditional_pit_histogram.svg")
    save_metric_comparison(metrics_df, output_dir / "conditional_metric_comparison.svg")
    write_report(
        output_dir / "conditional_probability_report.md",
        args,
        files,
        threshold,
        train_dates,
        test_dates,
        metrics_df,
        bars_summary,
    )

    print(f"Loaded {len(minutes):,} minute rows from {len(files)} file(s).")
    print(f"Built {len(time_bars):,} fixed-time bars and {len(turnover_bars):,} turnover bars.")
    print(f"Train dates: {train_dates[0]} to {train_dates[-1]}; test dates: {test_dates[0]} to {test_dates[-1]}.")
    print(metrics_df.to_string(index=False))
    print(f"Report: {(output_dir / 'conditional_probability_report.md').resolve()}")


if __name__ == "__main__":
    main()
