#!/usr/bin/env python3
import argparse
import calendar
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


WIND_SKILL_DIR = Path("/Users/sedol/.agents/skills/wind-mcp-skill")
ROOT = Path("/Users/sedol/Desktop/高频数据")
OUTPUT_COLUMNS = ["MATCH", "AVGPRICE", "VOLUME", "TURNOVER", "TIME", "_DATE"]
REQUEST_TIMEOUT_SECONDS = 90
DAILY_REQUEST_ATTEMPTS = 3

INDICES = {
    "csi500": {"windcode": "000905.SH", "cn": "中证500"},
    "csi1000": {"windcode": "000852.SH", "cn": "中证1000"},
    "csi800": {"windcode": "中证800", "cn": "中证800"},
    "chinext": {"windcode": "创业板指", "cn": "创业板指"},
}


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def month_ranges(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        first = date(y, m, 1)
        last = date(y, m, calendar.monthrange(y, m)[1])
        yield max(start, first), min(end, last)
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1


def day_ranges(start: date, end: date):
    current = start
    while current <= end:
        yield current, current
        current += timedelta(days=1)


def call_wind_quote(windcode: str, begin: date, end: date) -> dict:
    params = {
        "windcode": windcode,
        "begin": begin.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
    }
    cmd = [
        "node",
        "scripts/cli.mjs",
        "call",
        "index_data",
        "get_index_quote",
        json.dumps(params, ensure_ascii=False),
    ]
    proc = subprocess.run(
        cmd,
        cwd=WIND_SKILL_DIR,
        text=True,
        capture_output=True,
        check=False,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout.strip() or proc.stderr.strip())

    payload = json.loads(proc.stdout)
    if payload.get("isError"):
        raise RuntimeError(json.dumps(payload, ensure_ascii=False))

    text = payload["content"][0]["text"]
    inner = json.loads(text)
    if inner.get("error"):
        raise RuntimeError(json.dumps(inner["error"], ensure_ascii=False))
    return inner["data"]


def data_to_frame(data: dict) -> pd.DataFrame:
    columns = [item["name"] for item in data["columns"]]
    frame = pd.DataFrame(data["rows"], columns=columns)
    frame = frame.reindex(columns=OUTPUT_COLUMNS)

    frame["MATCH"] = pd.to_numeric(frame["MATCH"], errors="coerce")
    for col in ["VOLUME", "TURNOVER", "_DATE"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
    return frame


def output_filename(prefix: str, begin: date, end: date) -> str:
    suffix = "_mtd" if begin.month == end.month and end.day != calendar.monthrange(end.year, end.month)[1] else ""
    return f"{prefix}_{begin:%Y-%m}_1min{suffix}.xlsx"


def fetch_daily_frame(prefix: str, info: dict, begin: date, month_end: date) -> pd.DataFrame:
    frames = []
    for day, _ in day_ranges(begin, month_end):
        print(f"fetch daily {info['cn']} {info['windcode']} {day:%Y%m%d}", flush=True)
        daily_data = None
        for attempt in range(1, DAILY_REQUEST_ATTEMPTS + 1):
            try:
                daily_data = call_wind_quote(info["windcode"], day, day)
                break
            except (RuntimeError, subprocess.TimeoutExpired) as daily_exc:
                print(f"daily failed {day:%Y%m%d} attempt {attempt}: {daily_exc}", file=sys.stderr, flush=True)
        if daily_data is None:
            continue
        daily_frame = data_to_frame(daily_data)
        if not daily_frame.empty:
            frames.append(daily_frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)


def fetch_one_index(
    prefix: str,
    info: dict,
    start: date,
    end: date,
    overwrite: bool,
    daily: bool,
    output_start: date,
    output_end: date,
):
    out_dir = ROOT / f"{prefix}_minute_{output_start:%Y%m%d}_{output_end:%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "fetch_log.jsonl"

    for begin, month_end in month_ranges(start, end):
        target = out_dir / output_filename(prefix, begin, month_end)
        if target.exists() and not overwrite:
            print(f"skip existing {target}", flush=True)
            continue

        print(f"fetch {info['cn']} {info['windcode']} {begin:%Y%m%d}-{month_end:%Y%m%d}", flush=True)
        if daily:
            frame = fetch_daily_frame(prefix, info, begin, month_end)
        else:
            try:
                data = call_wind_quote(info["windcode"], begin, month_end)
                frame = data_to_frame(data)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                print(f"monthly failed, fallback to daily: {exc}", file=sys.stderr, flush=True)
                frame = fetch_daily_frame(prefix, info, begin, month_end)

        if frame.empty:
            status = {"file": str(target), "rows": 0, "status": "empty"}
        else:
            frame = frame.sort_values(["_DATE", "TIME"]).reset_index(drop=True)
            frame.to_excel(target, index=False)
            status = {
                "file": str(target),
                "rows": int(len(frame)),
                "min_date": int(frame["_DATE"].min()),
                "max_date": int(frame["_DATE"].max()),
                "status": "ok",
            }

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(status, ensure_ascii=False) + "\n")
        print(json.dumps(status, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20240618")
    parser.add_argument("--indices", nargs="+", default=list(INDICES))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--daily", action="store_true")
    parser.add_argument("--output-start")
    parser.add_argument("--output-end")
    args = parser.parse_args()

    unknown = sorted(set(args.indices) - set(INDICES))
    if unknown:
        raise SystemExit(f"unknown indices: {', '.join(unknown)}")

    start = parse_date(args.start)
    end = parse_date(args.end)
    output_start = parse_date(args.output_start) if args.output_start else start
    output_end = parse_date(args.output_end) if args.output_end else end
    if end < start:
        raise SystemExit("--end must be >= --start")

    for prefix in args.indices:
        fetch_one_index(prefix, INDICES[prefix], start, end, args.overwrite, args.daily, output_start, output_end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
