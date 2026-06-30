#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_resampling import load_minutes, make_time_bars, make_turnover_bars, markdown_table
from conditional_probability_experiment import (
    ConditionalKDE,
    evaluate_model,
    prepare_model_data,
    save_conditional_slices,
    save_metric_comparison,
    save_pit_histogram,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standard chronological out-of-sample test for conditional probability K-line models."
    )
    parser.add_argument("--data-dir", default="hs300_minute_20200101_20240618")
    parser.add_argument("--output-dir", default="conditional_oos_results")
    parser.add_argument("--train-start", default="2024-04-01")
    parser.add_argument("--train-end", default="2024-05-31")
    parser.add_argument("--test-start", default="2024-06-01")
    parser.add_argument("--test-end", default="2024-06-30")
    parser.add_argument("--time-step", type=int, default=5)
    parser.add_argument("--target-bars-per-day", type=int, default=48)
    parser.add_argument("--turnover-threshold", type=float, default=None)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=501)
    return parser.parse_args()


def in_window(series: pd.Series, start: str, end: str) -> pd.Series:
    values = pd.to_datetime(series.astype(str))
    return (values >= pd.Timestamp(start)) & (values <= pd.Timestamp(end))


def build_bars_with_train_threshold(minutes: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    train_minutes = minutes[in_window(minutes["trade_date"].dt.date.astype(str), args.train_start, args.train_end)]
    if train_minutes.empty:
        raise ValueError("No minute rows found in the requested training window.")
    threshold = (
        float(args.turnover_threshold)
        if args.turnover_threshold is not None
        else float(train_minutes.groupby(train_minutes["trade_date"].dt.date)["turnover"].sum().median() / args.target_bars_per_day)
    )
    return make_time_bars(minutes, args.time_step), make_turnover_bars(minutes, threshold), threshold


def split_model_data(data: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data[in_window(data["trade_date"], args.train_start, args.train_end)].copy()
    test = data[in_window(data["trade_date"], args.test_start, args.test_end)].copy()
    if train.empty or test.empty:
        raise ValueError("Training or test set is empty after feature construction.")
    return train, test


def bar_summary(time_bars: pd.DataFrame, turnover_bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for name, bars in [("time_5min", time_bars), ("turnover", turnover_bars)]:
        for period, start, end in [
            ("train", args.train_start, args.train_end),
            ("test", args.test_start, args.test_end),
        ]:
            subset = bars[in_window(bars["trade_date"], start, end)]
            rows.append(
                {
                    "sample": name,
                    "period": period,
                    "bars": int(len(subset)),
                    "trading_days": int(subset["trade_date"].nunique()),
                    "mean_bars_per_day": float(subset.groupby("trade_date").size().mean()),
                    "mean_minutes_per_bar": float(subset["minutes"].mean()),
                    "mean_turnover_per_bar": float(subset["turnover"].mean()),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    args: argparse.Namespace,
    files: list[Path],
    threshold: float,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    time_row = metrics[metrics["sample"].eq("time_5min")].iloc[0]
    turn_row = metrics[metrics["sample"].eq("turnover")].iloc[0]
    checks = [
        ("avg_log_score", "higher", "average log score"),
        ("crps_bps", "lower", "CRPS"),
        ("pit_ks", "lower", "PIT KS distance"),
        ("interval90_abs_error", "lower", "90% interval coverage error"),
        ("direction_brier", "lower", "direction Brier score"),
        ("left_tail_brier", "lower", "left-tail Brier score"),
    ]
    quick = []
    for column, direction, label in checks:
        time_value = float(time_row[column])
        turn_value = float(turn_row[column])
        turnover_better = turn_value > time_value if direction == "higher" else turn_value < time_value
        winner = "turnover-resampled bars" if turnover_better else "fixed-time bars"
        quick.append(f"- {label}: {winner} (turnover `{turn_value:.6g}` vs fixed-time `{time_value:.6g}`).")

    lines = [
        "# Standard Out-of-Sample Conditional Probability Test",
        "",
        "## Dataset Design",
        "",
        "- This is a chronological out-of-sample test, not a random split.",
        f"- Training window: `{args.train_start}` to `{args.train_end}`.",
        f"- Out-of-sample test window: `{args.test_start}` to `{args.test_end}`.",
        f"- Source files: {', '.join(p.name for p in files)}.",
        "- The turnover-bar threshold is estimated using training data only.",
        f"- Turnover threshold: `{threshold:,.0f}`.",
        f"- Fixed-time K line: `{args.time_step}` minute bars.",
        f"- Rolling volatility window: `{args.vol_window}` bars.",
        "",
        "## Method",
        "",
        "- For each K-line type, compute current rolling volatility and next-bar log return.",
        "- Fit `f(next return | current volatility)` using a two-dimensional Gaussian-kernel conditional density on the training window.",
        "- Evaluate the fitted conditional distribution only on the future OOS test window.",
        "",
        "## Bar Summary",
        "",
        markdown_table(summary),
        "",
        "## OOS Metrics",
        "",
        markdown_table(metrics),
        "",
        "Metric interpretation: higher is better for `avg_log_score`; lower is better for `crps_bps`, `pit_ks`, `interval90_abs_error`, `direction_brier`, and `left_tail_brier`.",
        "",
        "## Quick Read",
        "",
        *quick,
        "",
        "## Figures",
        "",
        "![Conditional density slices](conditional_density_slices.svg)",
        "",
        "![PIT histogram](conditional_pit_histogram.svg)",
        "",
        "![Metric comparison](conditional_metric_comparison.svg)",
        "",
        "## Generated Files",
        "",
        "- `oos_model_metrics.csv`",
        "- `oos_model_diagnostics.csv`",
        "- `oos_bar_summary.csv`",
        "- `oos_conditional_probability_report.md`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_start = min(pd.Timestamp(args.train_start), pd.Timestamp(args.test_start)).date().isoformat()
    load_end = max(pd.Timestamp(args.train_end), pd.Timestamp(args.test_end)).date().isoformat()
    minutes, files = load_minutes(Path(args.data_dir), load_start, load_end)
    time_bars, turnover_bars, threshold = build_bars_with_train_threshold(minutes, args)

    datasets = {
        "time_5min": prepare_model_data(time_bars, "time_5min", args.vol_window),
        "turnover": prepare_model_data(turnover_bars, "turnover", args.vol_window),
    }
    train_sets: dict[str, pd.DataFrame] = {}
    test_sets: dict[str, pd.DataFrame] = {}
    for name, data in datasets.items():
        train_sets[name], test_sets[name] = split_model_data(data, args)

    models = {name: ConditionalKDE.fit(train, args.grid_size) for name, train in train_sets.items()}
    metric_rows = []
    diagnostics = []
    for name in ["time_5min", "turnover"]:
        row, diag = evaluate_model(models[name], train_sets[name], test_sets[name], name)
        metric_rows.append(row)
        diagnostics.append(diag)

    metrics = pd.DataFrame(metric_rows)
    diagnostics_df = pd.concat(diagnostics, ignore_index=True)
    summary = bar_summary(time_bars, turnover_bars, args)

    metrics.to_csv(output_dir / "oos_model_metrics.csv", index=False)
    diagnostics_df.to_csv(output_dir / "oos_model_diagnostics.csv", index=False)
    summary.to_csv(output_dir / "oos_bar_summary.csv", index=False)
    time_bars.to_csv(output_dir / "time_5min_bars.csv", index=False)
    turnover_bars.to_csv(output_dir / "turnover_bars.csv", index=False)

    save_conditional_slices(models, train_sets, output_dir / "conditional_density_slices.svg")
    save_pit_histogram(diagnostics_df, output_dir / "conditional_pit_histogram.svg")
    save_metric_comparison(metrics, output_dir / "conditional_metric_comparison.svg")
    write_report(output_dir / "oos_conditional_probability_report.md", args, files, threshold, metrics, summary)

    print(f"Loaded {len(minutes):,} minute rows from {len(files)} file(s).")
    print(f"Train: {args.train_start} to {args.train_end}; OOS test: {args.test_start} to {args.test_end}.")
    print(f"Turnover threshold estimated from train only: {threshold:,.0f}")
    print(metrics.to_string(index=False))
    print(f"Report: {(output_dir / 'oos_conditional_probability_report.md').resolve()}")


if __name__ == "__main__":
    main()
