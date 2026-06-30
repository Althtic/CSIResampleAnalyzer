#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_resampling import (
    FIELD_DICTIONARY,
    load_minutes,
    make_time_bars,
    make_turnover_bars,
    markdown_table,
    returns_from_bars,
    save_abs_return_acf,
    save_metric_ratios,
    save_return_histogram,
    summarize_bars,
    write_field_dictionary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed-time K bars and equal-turnover K bars from HS300 1-minute data, "
            "then save CSV diagnostics, SVG charts, and an inspection report."
        )
    )
    parser.add_argument("--data-dir", default="hs300_minute_20200101_20240618")
    parser.add_argument("--output-dir", default="resampling_results")
    parser.add_argument("--start", default="2024-04-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument("--time-step", type=int, default=5, help="Minutes per normal time bar.")
    parser.add_argument(
        "--target-bars-per-day",
        type=int,
        default=48,
        help="Infer turnover threshold as median daily turnover / target bars.",
    )
    parser.add_argument(
        "--turnover-threshold",
        type=float,
        default=None,
        help="Use a fixed turnover threshold instead of inferring one from the sample.",
    )
    parser.add_argument(
        "--plot-date",
        default=None,
        help="Trading date to plot as YYYY-MM-DD. Defaults to the first available trading day.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of trading days used to fit the GARCH models.",
    )
    parser.add_argument(
        "--skip-garch",
        action="store_true",
        help="Only build resampling outputs; skip the GARCH(1,1) comparison.",
    )
    return parser.parse_args()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(x: float, y: float, text: object, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#17202a">{esc(text)}</text>'
    )


def pick_plot_date(minutes: pd.DataFrame, requested: str | None) -> str:
    available = sorted(minutes["trade_date"].dt.date.astype(str).unique())
    if not available:
        raise ValueError("No trading dates available after filtering.")
    if requested is None:
        return available[0]
    requested_date = pd.Timestamp(requested).date().isoformat()
    if requested_date not in set(available):
        raise ValueError(f"plot date {requested_date} is not in the sample. First available date is {available[0]}.")
    return requested_date


def save_kline_comparison(time_bars: pd.DataFrame, turnover_bars: pd.DataFrame, plot_date: str, path: Path) -> None:
    panels = [
        (f"FIXED-TIME {time_bars['bar_type'].iloc[0]} K bars (not resampled)", time_bars[time_bars["trade_date"] == plot_date], "#2563eb"),
        ("TURNOVER-RESAMPLED reconstructed K bars", turnover_bars[turnover_bars["trade_date"] == plot_date], "#d97706"),
    ]
    panels = [(title, data.reset_index(drop=True), color) for title, data, color in panels if not data.empty]
    if len(panels) != 2:
        raise ValueError(f"Could not find both bar types for plot date {plot_date}.")

    w, h = 1180, 760
    left, right, top, bottom = 78, 42, 72, 58
    panel_gap = 68
    panel_h = (h - top - bottom - panel_gap) / 2
    plot_w = w - left - right
    all_prices = pd.concat([data[["open", "high", "low", "close"]] for _, data, _ in panels]).to_numpy(dtype=float)
    ymin = float(np.nanmin(all_prices))
    ymax = float(np.nanmax(all_prices))
    pad = max((ymax - ymin) * 0.08, 1.0)
    ymin -= pad
    ymax += pad

    def ymap(value: float, panel_top: float) -> float:
        return panel_top + panel_h - (value - ymin) / (ymax - ymin) * panel_h

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, f"K-line comparison on {plot_date}: fixed-time vs TURNOVER-RESAMPLED", 20, "middle", "700"),
        svg_text(
            w / 2,
            52,
            "Blue top panel is fixed-time; orange bottom panel is TURNOVER-RESAMPLED.",
            12,
            "middle",
        ),
    ]

    for panel_idx, (title, data, accent) in enumerate(panels):
        panel_top = top + panel_idx * (panel_h + panel_gap)
        elems.append(svg_text(left, panel_top - 16, title, 15, "start", "700"))
        elems.append(f'<rect x="{left}" y="{panel_top:.1f}" width="{plot_w}" height="{panel_h:.1f}" fill="#ffffff" stroke="#d7dee8"/>')
        for tick in np.linspace(ymin, ymax, 5):
            y = ymap(float(tick), panel_top)
            elems.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#edf1f5"/>')
            elems.append(svg_text(left - 10, y + 4, f"{tick:.1f}", 11, "end"))

        n = len(data)
        slot = plot_w / max(n, 1)
        candle_w = max(3.0, min(10.0, slot * 0.58))
        for idx, row in data.iterrows():
            x = left + slot * (idx + 0.5)
            o, hi, lo, c = (float(row[col]) for col in ["open", "high", "low", "close"])
            up = c >= o
            fill = "#ffffff" if up else accent
            stroke = "#16a34a" if up else accent
            y_hi, y_lo = ymap(hi, panel_top), ymap(lo, panel_top)
            y_o, y_c = ymap(o, panel_top), ymap(c, panel_top)
            body_y = min(y_o, y_c)
            body_h = max(abs(y_c - y_o), 1.2)
            elems.append(f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="{stroke}" stroke-width="1.2"/>')
            elems.append(
                f'<rect x="{x - candle_w / 2:.1f}" y="{body_y:.1f}" width="{candle_w:.1f}" '
                f'height="{body_h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
            )

        first_label = str(pd.to_datetime(data["start_time"].iloc[0]).time())[:5]
        last_label = str(pd.to_datetime(data["end_time"].iloc[-1]).time())[:5]
        elems.append(svg_text(left, panel_top + panel_h + 26, first_label, 11, "start"))
        elems.append(svg_text(left + plot_w, panel_top + panel_h + 26, last_label, 11, "end"))
        elems.append(svg_text(left + plot_w / 2, panel_top + panel_h + 26, f"{n} bars", 11, "middle"))

    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def minute_return_frame(minutes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, day in minutes.groupby(minutes["trade_date"].dt.date, sort=True):
        day = day.sort_values("datetime").reset_index(drop=True)
        log_price = np.log(day["price"].astype(float).to_numpy())
        returns = np.diff(log_price)
        rows.append(
            pd.DataFrame(
                {
                    "trade_date": day["trade_date"].dt.date.astype(str).iloc[1:].to_numpy(),
                    "timestamp": day["datetime"].iloc[1:].to_numpy(),
                    "return": returns,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def bar_return_frame(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, day in bars.groupby("trade_date", sort=True):
        day = day.sort_values("end_time").reset_index(drop=True)
        log_close = np.log(day["close"].astype(float).to_numpy())
        returns = np.diff(log_close)
        rows.append(
            pd.DataFrame(
                {
                    "trade_date": day["trade_date"].iloc[1:].to_numpy(),
                    "timestamp": day["end_time"].iloc[1:].to_numpy(),
                    "return": returns,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def return_distribution_samples(minutes: pd.DataFrame, turnover_bars: pd.DataFrame) -> pd.DataFrame:
    samples = []
    for label, frame, price_col, time_col in [
        ("raw_1min", minutes, "price", "datetime"),
        ("turnover_bar", turnover_bars, "close", "end_time"),
    ]:
        for _, day in frame.groupby("trade_date", sort=True):
            day = day.sort_values(time_col).reset_index(drop=True)
            price = day[price_col].astype(float).to_numpy()
            if len(price) < 2:
                continue
            simple_returns = price[1:] / price[:-1] - 1.0
            log_returns = np.diff(np.log(price))
            samples.append(
                pd.DataFrame(
                    {
                        "sample": label,
                        "trade_date": day["trade_date"].astype(str).iloc[1:].to_numpy(),
                        "simple_return": simple_returns,
                        "log_return": log_returns,
                    }
                )
            )
    return pd.concat(samples, ignore_index=True)


def normal_cdf(values: np.ndarray) -> np.ndarray:
    vectorized_erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + vectorized_erf(values / math.sqrt(2.0)))


def distribution_metric_row(sample: str, return_type: str, values: np.ndarray) -> dict[str, float | str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if n < 4 or std <= 0:
        raise ValueError(f"Not enough variation to analyze {sample} {return_type}.")
    z = (values - mean) / std
    skew = float(np.mean(z**3))
    excess_kurtosis = float(np.mean(z**4) - 3.0)
    jarque_bera = float(n / 6.0 * (skew**2 + excess_kurtosis**2 / 4.0))
    sorted_z = np.sort(z)
    empirical = np.arange(1, n + 1) / n
    fitted_normal = normal_cdf(sorted_z)
    ks_distance = float(np.max(np.abs(empirical - fitted_normal)))
    normal_2sigma = 2.0 * (1.0 - 0.9772498680518208)
    normal_3sigma = 2.0 * (1.0 - 0.9986501019683699)
    tail_2sigma = float(np.mean(np.abs(z) > 2.0))
    tail_3sigma = float(np.mean(np.abs(z) > 3.0))
    center_10bp = float(np.mean(np.abs(values) < 0.001))
    return {
        "sample": sample,
        "return_type": return_type,
        "count": n,
        "mean": mean,
        "std": std,
        "skew": skew,
        "excess_kurtosis": excess_kurtosis,
        "jarque_bera": jarque_bera,
        "jarque_bera_per_obs": jarque_bera / n,
        "ks_distance": ks_distance,
        "tail_prob_abs_gt_2sigma": tail_2sigma,
        "tail_ratio_2sigma_vs_normal": tail_2sigma / normal_2sigma,
        "tail_prob_abs_gt_3sigma": tail_3sigma,
        "tail_ratio_3sigma_vs_normal": tail_3sigma / normal_3sigma,
        "center_prob_abs_lt_10bp": center_10bp,
    }


def return_distribution_metrics(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample in ["raw_1min", "turnover_bar"]:
        subset = samples[samples["sample"].eq(sample)]
        for return_type, column in [("simple", "simple_return"), ("log", "log_return")]:
            rows.append(distribution_metric_row(sample, return_type, subset[column].to_numpy()))
    return pd.DataFrame(rows)


def save_return_distribution_chart(samples: pd.DataFrame, path: Path) -> None:
    panels = [
        ("raw_1min", "simple_return", "Raw 1min simple return (not resampled)", "#2563eb"),
        ("turnover_bar", "simple_return", "TURNOVER-RESAMPLED simple return", "#d97706"),
        ("raw_1min", "log_return", "Raw 1min log return (not resampled)", "#2563eb"),
        ("turnover_bar", "log_return", "TURNOVER-RESAMPLED log return", "#d97706"),
    ]
    w, h = 1180, 780
    margin_x, top = 78, 68
    panel_w, panel_h = 500, 255
    gap_x, gap_y = 58, 78
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "Return distributions: raw 1min vs TURNOVER-RESAMPLED", 20, "middle", "700"),
        svg_text(w / 2, 52, "Returns are standardized by each sample's own mean and standard deviation; tails are clipped to 0.5%-99.5% for readability.", 12, "middle"),
    ]

    for idx, (sample, column, title, color) in enumerate(panels):
        row, col = divmod(idx, 2)
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = top + row * (panel_h + gap_y)
        values = samples.loc[samples["sample"].eq(sample), column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        z = (values - values.mean()) / values.std(ddof=1)
        lo, hi = np.percentile(z, [0.5, 99.5])
        bins = np.linspace(float(lo), float(hi), 42)
        hist, edges = np.histogram(z, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        normal_pdf = 1.0 / math.sqrt(2.0 * math.pi) * np.exp(-0.5 * centers**2)
        ymax = max(float(hist.max()), float(normal_pdf.max())) * 1.2

        def xmap(value: float) -> float:
            return x0 + (value - lo) / (hi - lo) * panel_w

        def ymap(value: float) -> float:
            return y0 + panel_h - value / ymax * panel_h

        elems.append(svg_text(x0, y0 - 18, title, 14, "start", "700"))
        elems.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#ffffff" stroke="#d7dee8"/>')
        for tick in np.linspace(0, ymax, 5):
            y = ymap(float(tick))
            elems.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" stroke="#edf1f5"/>')
        for i, density in enumerate(hist):
            bx0, bx1 = xmap(float(edges[i])), xmap(float(edges[i + 1]))
            y = ymap(float(density))
            elems.append(
                f'<rect x="{bx0:.1f}" y="{y:.1f}" width="{max(1.0, bx1 - bx0 - 1):.1f}" '
                f'height="{y0 + panel_h - y:.1f}" fill="{color}" opacity="0.58"/>'
            )
        points = " ".join(f"{xmap(float(x)):.1f},{ymap(float(y)):.1f}" for x, y in zip(centers, normal_pdf))
        elems.append(f'<polyline points="{points}" fill="none" stroke="#111827" stroke-width="2.0"/>')
        for tick in [-4, -2, 0, 2, 4]:
            if lo <= tick <= hi:
                x = xmap(float(tick))
                elems.append(f'<line x1="{x:.1f}" y1="{y0 + panel_h}" x2="{x:.1f}" y2="{y0 + panel_h + 5}" stroke="#2c3e50"/>')
                elems.append(svg_text(x, y0 + panel_h + 22, tick, 10, "middle"))
        elems.append(svg_text(x0 + panel_w / 2, y0 + panel_h + 42, "standardized return", 11, "middle"))
        elems.append(f'<rect x="{x0 + 14}" y="{y0 + 13}" width="14" height="14" fill="{color}" opacity="0.65"/>')
        elems.append(svg_text(x0 + 36, y0 + 25, "empirical", 11))
        elems.append(f'<line x1="{x0 + 112}" y1="{y0 + 20}" x2="{x0 + 146}" y2="{y0 + 20}" stroke="#111827" stroke-width="2"/>')
        elems.append(svg_text(x0 + 154, y0 + 25, "fitted normal", 11))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def save_return_distribution_metric_chart(metrics: pd.DataFrame, path: Path) -> None:
    rows = []
    for return_type in ["simple", "log"]:
        raw = metrics[(metrics["sample"].eq("raw_1min")) & (metrics["return_type"].eq(return_type))].iloc[0]
        turnover = metrics[(metrics["sample"].eq("turnover_bar")) & (metrics["return_type"].eq(return_type))].iloc[0]
        for label, column in [
            ("Abs excess kurtosis", "excess_kurtosis"),
            ("JB per obs", "jarque_bera_per_obs"),
            ("KS distance", "ks_distance"),
            ("3-sigma tail ratio", "tail_ratio_3sigma_vs_normal"),
        ]:
            raw_value = abs(float(raw[column])) if column == "excess_kurtosis" else float(raw[column])
            turnover_value = abs(float(turnover[column])) if column == "excess_kurtosis" else float(turnover[column])
            rows.append((f"{return_type}: {label}", turnover_value / raw_value, turnover_value, raw_value))

    w, h = 1080, 570
    left, right, top = 250, 230, 64
    row_h = 56
    plot_w = w - left - right
    xmax = max(1.35, max(row[1] for row in rows if np.isfinite(row[1])) * 1.12)

    def xmap(value: float) -> float:
        return left + value / xmax * plot_w

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "TURNOVER-RESAMPLED distribution metric / raw 1min metric", 18, "middle", "700"),
        svg_text(w / 2, 50, "Values below 1.0 mean the TURNOVER-RESAMPLED distribution is closer to normal on that metric.", 12, "middle"),
        f'<line x1="{xmap(1.0):.1f}" y1="{top - 8}" x2="{xmap(1.0):.1f}" y2="{top + row_h * len(rows)}" stroke="#7f8c8d" stroke-dasharray="4,4"/>',
    ]
    for idx, (label, ratio, turnover_value, raw_value) in enumerate(rows):
        y = top + idx * row_h
        color = "#1f9d55" if ratio < 1.0 else "#c0392b"
        elems.append(svg_text(24, y + 28, label, 12))
        elems.append(f'<rect x="{left}" y="{y + 10}" width="{max(1, xmap(ratio) - left):.1f}" height="24" rx="3" fill="{color}" opacity="0.78"/>')
        elems.append(svg_text(xmap(ratio) + 8, y + 28, f"{ratio:.2f}x", 12))
        elems.append(svg_text(w - 18, y + 28, f"turnover {turnover_value:.4g} | raw {raw_value:.4g}", 11, "end"))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def write_return_distribution_report(path: Path, metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Return Distribution Comparison",
        "",
        "## Scope",
        "",
        "- `raw_1min`: original 1-minute close-to-close returns, without resampling.",
        "- `turnover_bar`: returns from bars reconstructed by accumulated turnover.",
        "- Both samples use the same original date window.",
        "- Both simple returns and log returns are computed.",
        f"- Date window: `{args.start}` to `{args.end}`.",
        "",
        "## Metrics",
        "",
        markdown_table(metrics),
        "",
        "## How To Read",
        "",
        "- `excess_kurtosis`: above zero means sharper peak and fatter tails than normal.",
        "- `jarque_bera_per_obs`: lower means closer to normal after adjusting for sample size.",
        "- `ks_distance`: lower means the empirical CDF is closer to the fitted normal CDF.",
        "- `tail_ratio_3sigma_vs_normal`: above 1 means more extreme 3-sigma observations than a normal distribution.",
        "",
        "## Quick Conclusion",
        "",
    ]
    for return_type in ["simple", "log"]:
        raw = metrics[(metrics["sample"].eq("raw_1min")) & (metrics["return_type"].eq(return_type))].iloc[0]
        turnover = metrics[(metrics["sample"].eq("turnover_bar")) & (metrics["return_type"].eq(return_type))].iloc[0]
        lines.append(
            f"- {return_type}: turnover bars reduce excess kurtosis from `{raw['excess_kurtosis']:.6g}` "
            f"to `{turnover['excess_kurtosis']:.6g}`, and JB/obs from `{raw['jarque_bera_per_obs']:.6g}` "
            f"to `{turnover['jarque_bera_per_obs']:.6g}`."
        )
        lines.append(
            f"- {return_type}: 3-sigma tail ratio changes from `{raw['tail_ratio_3sigma_vs_normal']:.6g}` "
            f"to `{turnover['tail_ratio_3sigma_vs_normal']:.6g}`."
        )
    lines.extend(
        [
            "",
            "Overall, turnover resampling improves the normality diagnostics in this sample, but the resampled returns still have positive excess kurtosis and elevated 3-sigma tails. In plain terms: the peak-and-fat-tail problem is reduced, not eliminated.",
            "",
            "## Figures",
            "",
            "![Return distribution comparison](return_distribution_comparison.svg)",
            "",
            "![Return distribution metric ratios](return_distribution_metric_ratios.svg)",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def garch_nll(params: tuple[float, float, float], returns_bps: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return float("inf")
    values = np.asarray(returns_bps, dtype=float)
    var = float(np.var(values))
    if not np.isfinite(var) or var <= 0:
        return float("inf")
    sigma2 = max(var, omega / max(1.0 - alpha - beta, 1e-6))
    nll = 0.0
    for value in values:
        sigma2 = max(sigma2, 1e-12)
        nll += math.log(sigma2) + (value * value) / sigma2
        sigma2 = omega + alpha * value * value + beta * sigma2
    return 0.5 * nll


def fit_garch_11(returns: pd.Series, seed: int = 7) -> dict[str, float]:
    returns_bps = returns.dropna().to_numpy(dtype=float) * 10000.0
    returns_bps = returns_bps[np.isfinite(returns_bps)]
    if len(returns_bps) < 100:
        raise ValueError("Need at least 100 returns to fit GARCH(1,1).")
    sample_var = float(np.var(returns_bps))
    rng = np.random.default_rng(seed)

    candidates: list[tuple[float, float, float]] = []
    for alpha in [0.02, 0.04, 0.06, 0.08, 0.12, 0.18]:
        for beta in [0.65, 0.75, 0.85, 0.90, 0.94, 0.97]:
            if alpha + beta < 0.995:
                for scale in [0.5, 1.0, 2.0]:
                    omega = max(sample_var * (1.0 - alpha - beta) * scale, 1e-9)
                    candidates.append((omega, alpha, beta))
    for _ in range(450):
        alpha = float(rng.uniform(0.005, 0.22))
        beta = float(rng.uniform(0.55, 0.985))
        if alpha + beta >= 0.995:
            beta = 0.995 - alpha - float(rng.uniform(0.001, 0.02))
        if beta <= 0:
            continue
        omega = sample_var * max(1.0 - alpha - beta, 0.002) * float(np.exp(rng.uniform(-1.0, 1.0)))
        candidates.append((max(omega, 1e-9), alpha, beta))

    best = min(candidates, key=lambda params: garch_nll(params, returns_bps))
    best_score = garch_nll(best, returns_bps)

    for round_idx in range(6):
        omega, alpha, beta = best
        scale = 0.45 / (round_idx + 1)
        local_candidates = []
        for _ in range(220):
            new_omega = max(omega * float(np.exp(rng.normal(0.0, scale))), 1e-9)
            new_alpha = float(np.clip(alpha + rng.normal(0.0, 0.05 / (round_idx + 1)), 0.001, 0.35))
            new_beta = float(np.clip(beta + rng.normal(0.0, 0.08 / (round_idx + 1)), 0.05, 0.995))
            if new_alpha + new_beta >= 0.998:
                overflow = new_alpha + new_beta - 0.998
                new_beta = max(0.05, new_beta - overflow)
            local_candidates.append((new_omega, new_alpha, new_beta))
        for params in local_candidates:
            score = garch_nll(params, returns_bps)
            if score < best_score:
                best, best_score = params, score

    omega, alpha, beta = best
    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": float(alpha + beta),
        "train_nll": float(best_score),
        "train_return_count": int(len(returns_bps)),
    }


def filtered_garch_variance(returns: pd.Series, params: dict[str, float], init_variance: float | None = None) -> np.ndarray:
    values = returns.to_numpy(dtype=float) * 10000.0
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]
    sample_var = float(np.nanvar(values)) if init_variance is None else float(init_variance)
    sigma2_prev = max(sample_var, omega / max(1.0 - alpha - beta, 1e-6), 1e-9)
    sigma2 = np.empty(len(values), dtype=float)
    prev_return = 0.0
    for idx, value in enumerate(values):
        sigma2_t = omega + alpha * prev_return * prev_return + beta * sigma2_prev
        sigma2_t = max(sigma2_t, 1e-9)
        sigma2[idx] = sigma2_t
        if np.isfinite(value):
            prev_return = float(value)
            sigma2_prev = sigma2_t
    return sigma2


def garch_daily_predictions(
    returns: pd.DataFrame,
    train_dates: set[str],
    test_dates: list[str],
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    train_mask = returns["trade_date"].isin(train_dates)
    params = fit_garch_11(returns.loc[train_mask, "return"], seed=seed)
    train_var = float(np.var(returns.loc[train_mask, "return"].to_numpy(dtype=float) * 10000.0))
    sigma2 = filtered_garch_variance(returns["return"], params, init_variance=train_var)
    fitted = returns.copy()
    fitted["pred_var_bps2"] = sigma2
    daily = (
        fitted[fitted["trade_date"].isin(test_dates)]
        .groupby("trade_date", as_index=False)["pred_var_bps2"]
        .sum()
        .rename(columns={"pred_var_bps2": "predicted_rv_bps2"})
    )
    return params, daily


def garch_comparison(minutes: pd.DataFrame, turnover_bars: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    minute_returns = minute_return_frame(minutes)
    turnover_returns = bar_return_frame(turnover_bars)
    all_dates = sorted(minutes["trade_date"].dt.date.astype(str).unique())
    split_idx = int(len(all_dates) * train_ratio)
    split_idx = min(max(split_idx, 10), len(all_dates) - 5)
    train_dates = set(all_dates[:split_idx])
    test_dates = all_dates[split_idx:]

    actual_daily = (
        minute_returns[minute_returns["trade_date"].isin(test_dates)]
        .assign(actual_rv_bps2=lambda data: (data["return"] * 10000.0) ** 2)
        .groupby("trade_date", as_index=False)["actual_rv_bps2"]
        .sum()
    )
    raw_params, raw_daily = garch_daily_predictions(minute_returns, train_dates, test_dates, seed=11)
    turnover_params, turnover_daily = garch_daily_predictions(turnover_returns, train_dates, test_dates, seed=29)

    comparison = actual_daily.merge(
        raw_daily.rename(columns={"predicted_rv_bps2": "raw_1min_pred_rv_bps2"}),
        on="trade_date",
        how="inner",
    ).merge(
        turnover_daily.rename(columns={"predicted_rv_bps2": "turnover_pred_rv_bps2"}),
        on="trade_date",
        how="inner",
    )

    def metrics(column: str, label: str) -> dict[str, float | str]:
        actual = comparison["actual_rv_bps2"].to_numpy(dtype=float)
        pred = comparison[column].to_numpy(dtype=float)
        eps = 1e-12
        err = pred - actual
        corr = float(np.corrcoef(actual, pred)[0, 1]) if len(actual) > 2 and np.std(pred) > 0 else float("nan")
        qlike = float(np.mean(np.log(np.maximum(pred, eps)) + actual / np.maximum(pred, eps)))
        return {
            "model": label,
            "test_days": int(len(comparison)),
            "mae_bps2": float(np.mean(np.abs(err))),
            "rmse_bps2": float(np.sqrt(np.mean(err**2))),
            "mape": float(np.mean(np.abs(err) / np.maximum(actual, eps))),
            "qlike": qlike,
            "corr": corr,
        }

    metrics_frame = pd.DataFrame(
        [
            metrics("raw_1min_pred_rv_bps2", "raw_1min_garch"),
            metrics("turnover_pred_rv_bps2", "turnover_bar_garch"),
        ]
    )
    params_frame = pd.DataFrame(
        [
            {"model": "raw_1min_garch", **raw_params},
            {"model": "turnover_bar_garch", **turnover_params},
        ]
    )
    return comparison, metrics_frame, params_frame


def save_garch_variance_chart(comparison: pd.DataFrame, path: Path) -> None:
    w, h = 1180, 560
    left, right, top, bottom = 86, 36, 62, 82
    plot_w, plot_h = w - left - right, h - top - bottom
    dates = comparison["trade_date"].astype(str).tolist()
    series = [
        ("Actual daily RV", "actual_rv_bps2", "#111827"),
        ("Raw 1min GARCH", "raw_1min_pred_rv_bps2", "#2563eb"),
        ("TURNOVER-RESAMPLED bar GARCH", "turnover_pred_rv_bps2", "#d97706"),
    ]
    ymax = float(comparison[[col for _, col, _ in series]].to_numpy(dtype=float).max()) * 1.12
    ymin = 0.0

    def xmap(idx: int) -> float:
        return left + idx / max(len(dates) - 1, 1) * plot_w

    def ymap(value: float) -> float:
        return top + plot_h - (value - ymin) / max(ymax - ymin, 1e-9) * plot_h

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "GARCH(1,1) daily realized variance comparison", 19, "middle", "700"),
        svg_text(w / 2, 51, "Both models are evaluated on the same test trading days; target is raw 1-minute daily RV.", 12, "middle"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#2c3e50"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#2c3e50"/>',
    ]
    for tick in np.linspace(ymin, ymax, 6):
        y = ymap(float(tick))
        elems.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#edf1f5"/>')
        elems.append(svg_text(left - 10, y + 4, f"{tick:.0f}", 11, "end"))
    for label, column, color in series:
        values = comparison[column].to_numpy(dtype=float)
        points = " ".join(f"{xmap(idx):.1f},{ymap(float(value)):.1f}" for idx, value in enumerate(values))
        elems.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        for idx, value in enumerate(values):
            elems.append(f'<circle cx="{xmap(idx):.1f}" cy="{ymap(float(value)):.1f}" r="2.6" fill="{color}"/>')
    for idx in np.linspace(0, len(dates) - 1, min(6, len(dates))).astype(int):
        x = xmap(int(idx))
        elems.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#2c3e50"/>')
        elems.append(svg_text(x, top + plot_h + 24, dates[int(idx)][5:], 11, "middle"))
    elems.append(svg_text(left + plot_w / 2, h - 24, "test date", 12, "middle"))
    elems.append(svg_text(18, top + plot_h / 2, "daily RV, bps^2", 12, "middle"))
    for idx, (label, _, color) in enumerate(series):
        x = left + 18 + idx * 230
        elems.append(f'<rect x="{x}" y="{top + 10}" width="16" height="16" fill="{color}" opacity="0.85"/>')
        elems.append(svg_text(x + 24, top + 23, label, 12))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def save_garch_metrics_chart(metrics: pd.DataFrame, path: Path) -> None:
    raw = metrics[metrics["model"].eq("raw_1min_garch")].iloc[0]
    turnover = metrics[metrics["model"].eq("turnover_bar_garch")].iloc[0]
    rows = [
        ("MAE", "mae_bps2", True),
        ("RMSE", "rmse_bps2", True),
        ("MAPE", "mape", True),
        ("QLIKE", "qlike", True),
        ("Correlation", "corr", False),
    ]
    w, h = 980, 430
    left, right, top = 180, 160, 66
    row_h = 58
    plot_w = w - left - right
    ratios = []
    for label, column, lower_better in rows:
        base = float(raw[column])
        value = float(turnover[column])
        ratio = value / base if lower_better else base / value
        ratios.append((label, column, ratio, lower_better, value, base))
    xmax = max(1.45, max(r[2] for r in ratios if np.isfinite(r[2])) * 1.15)

    def xmap(value: float) -> float:
        return left + value / xmax * plot_w

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fbfcfd"/>',
        svg_text(w / 2, 30, "GARCH metric comparison", 18, "middle", "700"),
        svg_text(w / 2, 50, "Ratio below 1.0 favors TURNOVER-RESAMPLED bar GARCH. For correlation, ratio is raw / resampled.", 12, "middle"),
        f'<line x1="{xmap(1.0):.1f}" y1="{top - 8}" x2="{xmap(1.0):.1f}" y2="{top + row_h * len(rows)}" stroke="#7f8c8d" stroke-dasharray="4,4"/>',
    ]
    for idx, (label, column, ratio, lower_better, value, base) in enumerate(ratios):
        y = top + idx * row_h
        color = "#1f9d55" if ratio < 1.0 else "#c0392b"
        elems.append(svg_text(28, y + 28, label, 13))
        elems.append(f'<rect x="{left}" y="{y + 10}" width="{max(1, xmap(ratio) - left):.1f}" height="24" rx="3" fill="{color}" opacity="0.78"/>')
        elems.append(svg_text(xmap(ratio) + 8, y + 28, f"{ratio:.2f}x", 12))
        elems.append(svg_text(w - 18, y + 28, f"turnover {value:.4g} | raw {base:.4g}", 11, "end"))
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def write_garch_report(
    path: Path,
    comparison: pd.DataFrame,
    metrics: pd.DataFrame,
    params: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# GARCH(1,1) Comparison Report",
        "",
        "## Design",
        "",
        "- `raw_1min_garch`: fit GARCH(1,1) directly on un-resampled 1-minute intraday log returns.",
        "- `turnover_bar_garch`: first rebuild bars by accumulated turnover, then fit GARCH(1,1) on turnover-bar log returns.",
        "- Both models use the same original date window and the same train/test trading-day split.",
        "- Evaluation target is the same for both models: test-day realized variance computed from raw 1-minute returns.",
        "- Bar-level conditional variances are summed within each test day to produce a daily variance forecast.",
        f"- Train ratio: `{args.train_ratio}`. Test days: `{len(comparison)}`.",
        "",
        "## GARCH Parameters",
        "",
        markdown_table(params),
        "",
        "## Evaluation Metrics",
        "",
        markdown_table(metrics),
        "",
        "Lower is better for MAE, RMSE, MAPE, and QLIKE. Higher is better for correlation.",
        "",
        "## Daily Forecast Table",
        "",
        markdown_table(comparison),
        "",
        "## Figures",
        "",
        "![Daily variance comparison](garch_daily_variance.svg)",
        "",
        "![Metric comparison](garch_metric_comparison.svg)",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_inspection_report(
    path: Path,
    files: list[Path],
    minutes: pd.DataFrame,
    time_bars: pd.DataFrame,
    turnover_bars: pd.DataFrame,
    threshold: float,
    stats: pd.DataFrame,
    args: argparse.Namespace,
    plot_date: str,
) -> None:
    stock_list_note = (
        "`ashare_2026-06-15_1min_tushare/stock_list.csv` only contains the A-share stock list; "
        "the progress file shows zero data rows were written for that A-share minute download. "
        "Therefore this runnable prototype uses the available HS300 1-minute files."
    )
    field_rows = ["| Column | Meaning |", "|---|---|"]
    for column, meaning in FIELD_DICTIONARY:
        field_rows.append(f"| `{column}` | {meaning} |")

    time_row = stats[stats["sample"].str.contains("time")].iloc[0]
    turn_row = stats[stats["sample"].eq("turnover")].iloc[0]
    lines = [
        "# Resampling Inspection Report",
        "",
        "## What I Did",
        "",
        "- Wrote `main.py` as the direct entry point for the experiment.",
        "- Loaded the available HS300 1-minute monthly Excel files for a three-month sample.",
        "- Cleaned timestamps, price, volume, and turnover fields into a normalized minute table.",
        f"- Built normal fixed-time K bars using `{args.time_step}` minute rows per bar.",
        f"- Built reconstructed turnover bars by closing a bar once cumulative turnover reaches `{threshold:,.0f}`.",
        "- Saved both reconstructed bar tables, diagnostic statistics, and SVG figures.",
        "",
        "## Data Location",
        "",
        f"- Minute data used: `{args.data_dir}`",
        f"- Files used: {', '.join(f'`{p.name}`' for p in files)}",
        f"- Sample window: `{args.start}` to `{args.end}`",
        f"- Clean minute rows: `{len(minutes):,}`",
        f"- Trading days: `{minutes['trade_date'].dt.date.nunique()}`",
        f"- Note: {stock_list_note}",
        "",
        "## Field Dictionary",
        "",
        "\n".join(field_rows),
        "",
        "## Resampling Logic",
        "",
        "- Normal K line: sort each trading day by minute, group every fixed number of rows, then compute open/high/low/close/volume/turnover.",
        "- Turnover K line: sort each trading day by minute, accumulate `TURNOVER`, close the bar when the threshold is reached, then compute the same OHLC fields.",
        f"- Threshold rule: median daily turnover divided by `{args.target_bars_per_day}` target bars per day, unless `--turnover-threshold` is provided.",
        "- Each trading day resets independently, so no overnight bar is formed.",
        "",
        "## Main Diagnostics",
        "",
        markdown_table(stats),
        "",
        "## Quick Conclusion",
        "",
        f"- Turnover per bar is more even: `{turn_row['turnover_cv']:.6g}` vs `{time_row['turnover_cv']:.6g}`.",
        f"- Absolute-return ACF1 is lower: `{turn_row['abs_return_acf1']:.6g}` vs `{time_row['abs_return_acf1']:.6g}`.",
        f"- JB per return is lower: `{turn_row['jarque_bera_per_return']:.6g}` vs `{time_row['jarque_bera_per_return']:.6g}`.",
        "- In this sample, turnover bars look statistically cleaner on the selected diagnostics, at the cost of variable bar counts per day.",
        "",
        "## Figures",
        "",
        f"![K-line comparison](kline_comparison_{plot_date}.svg)",
        "",
        "![Diagnostic ratios](diagnostic_ratios.svg)",
        "",
        "![Return histogram](return_histogram.svg)",
        "",
        "![Absolute return ACF](abs_return_acf.svg)",
        "",
        "## Files Generated",
        "",
        "- `clean_minutes_sample.csv`: cleaned minute-level sample.",
        "- `time_bars.csv`: normal fixed-time K bars.",
        "- `turnover_bars.csv`: reconstructed equal-turnover K bars.",
        "- `diagnostics.csv`: summary statistics used in this report.",
        "- `field_dictionary.md`: column interpretation.",
        "- `kline_comparison_*.svg`: visual K-line comparison for one trading day.",
        "- `return_histogram.svg`, `diagnostic_ratios.svg`, `abs_return_acf.svg`: statistical comparison charts.",
        "- `garch_report.md`: GARCH(1,1) comparison on raw 1-minute data vs turnover bars.",
        "- `garch_daily_variance.svg`, `garch_metric_comparison.svg`: GARCH evaluation charts.",
        "",
        "## How To Re-run",
        "",
        "```bash",
        "python3 main.py",
        "```",
        "",
        "Optional example:",
        "",
        "```bash",
        "python3 main.py --start 2024-04-01 --end 2024-06-30 --time-step 5 --target-bars-per-day 48 --plot-date 2024-04-01",
        "```",
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
    stats = pd.DataFrame(
        [
            summarize_bars(f"{args.time_step}min_time", time_bars),
            summarize_bars("turnover", turnover_bars),
        ]
    )
    plot_date = pick_plot_date(minutes, args.plot_date)

    minutes.to_csv(output_dir / "clean_minutes_sample.csv", index=False)
    time_bars.to_csv(output_dir / "time_bars.csv", index=False)
    turnover_bars.to_csv(output_dir / "turnover_bars.csv", index=False)
    stats.to_csv(output_dir / "diagnostics.csv", index=False)
    write_field_dictionary(output_dir / "field_dictionary.md")

    time_returns = returns_from_bars(time_bars)
    turnover_returns = returns_from_bars(turnover_bars)
    save_kline_comparison(time_bars, turnover_bars, plot_date, output_dir / f"kline_comparison_{plot_date}.svg")
    save_return_histogram(time_returns, turnover_returns, output_dir / "return_histogram.svg")
    save_metric_ratios(stats, output_dir / "diagnostic_ratios.svg")
    save_abs_return_acf(time_returns, turnover_returns, output_dir / "abs_return_acf.svg")

    distribution_samples = return_distribution_samples(minutes, turnover_bars)
    distribution_metrics = return_distribution_metrics(distribution_samples)
    distribution_samples.to_csv(output_dir / "return_distribution_samples.csv", index=False)
    distribution_metrics.to_csv(output_dir / "return_distribution_metrics.csv", index=False)
    save_return_distribution_chart(distribution_samples, output_dir / "return_distribution_comparison.svg")
    save_return_distribution_metric_chart(distribution_metrics, output_dir / "return_distribution_metric_ratios.svg")
    write_return_distribution_report(output_dir / "return_distribution_report.md", distribution_metrics, args)

    if not args.skip_garch:
        garch_daily, garch_metrics, garch_params = garch_comparison(minutes, turnover_bars, args.train_ratio)
        garch_daily.to_csv(output_dir / "garch_daily_predictions.csv", index=False)
        garch_metrics.to_csv(output_dir / "garch_metrics.csv", index=False)
        garch_params.to_csv(output_dir / "garch_params.csv", index=False)
        save_garch_variance_chart(garch_daily, output_dir / "garch_daily_variance.svg")
        save_garch_metrics_chart(garch_metrics, output_dir / "garch_metric_comparison.svg")
        write_garch_report(output_dir / "garch_report.md", garch_daily, garch_metrics, garch_params, args)

    write_inspection_report(
        output_dir / "inspection_report.md",
        files,
        minutes,
        time_bars,
        turnover_bars,
        threshold,
        stats,
        args,
        plot_date,
    )

    print(f"Loaded {len(minutes):,} minute rows from {len(files)} files.")
    print(f"Built {len(time_bars):,} fixed-time bars and {len(turnover_bars):,} turnover bars.")
    print(f"Turnover threshold: {threshold:,.0f}")
    print(f"K-line comparison date: {plot_date}")
    if not args.skip_garch:
        print(f"GARCH report: {(output_dir / 'garch_report.md').resolve()}")
    print(f"Inspection report: {(output_dir / 'inspection_report.md').resolve()}")
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
