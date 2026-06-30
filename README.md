# CSI Resample Analyzer

Chinese A-share index (中证系列) high-frequency data analysis comparing **fixed-time K-lines** vs **turnover-resampled K-lines**.

## Indices

| Key | Name | Wind Code |
|---|---|---|
| hs300 | 沪深300 | 000300.SH |
| csi500 | 中证500 | 000905.SH |
| csi1000 | 中证1000 | 000852.SH |
| csi800 | 中证800 | 000906.SH |
| chinext | 创业板指 | 399006.SZ |

## Scripts

| Script | Description |
|---|---|
| `analyze_resampling.py` | Core library: `make_time_bars()`, `make_turnover_bars()`, statistics utilities |
| `factor_ml_comparison.py` | Full pipeline: stability analysis → volume-price factors → RF ML prediction |
| `vix_ml_comparison.py` | VIX-enhanced multi-model comparison (RF/XGBoost/LSTM/Attention) |
| `cross_universe_resampling.py` | Cross-index resampling analysis |
| `conditional_probability_experiment.py` | Conditional KDE density estimation for P(next ret \| current vol) |
| `conditional_oos_experiment.py` | Chronological out-of-sample test |
| `fetch_index_minute_wind.py` | Wind terminal data acquisition (MCP-based) |
| `main.py` | Original HS300 resampling + GARCH(1,1) comparison |

## Sample Data

The `sample_data/` directory contains 1 month (2024-01) of HS300 bar data:

- `hs300_5min_bars_2024-01.csv` — Fixed 5-minute K-lines (1,078 bars)
- `hs300_turnover_bars_2024-01.csv` — Turnover-resampled K-lines (988 bars)

Each bar file includes: open, high, low, close, volume, turnover, bar duration (minutes).

## Quick Start

```bash
# Full pipeline with 5 indices, RF only
python3 factor_ml_comparison.py --start 2020-01-01 --end 2024-06-18

# VIX-enhanced multi-model comparison
python3 vix_ml_comparison.py --start 2020-01-01 --end 2024-06-18

# Use sample data directly
python3 -c "
from analyze_resampling import load_minutes, markdown_table
from pathlib import Path
import pandas as pd

# Load sample bars and inspect
b5 = pd.read_csv('sample_data/hs300_5min_bars_2024-01.csv')
bt = pd.read_csv('sample_data/hs300_turnover_bars_2024-01.csv')
print(f'5min: {len(b5)} bars, avg {b5[\"turnover\"].mean():.0f} turnover/bar')
print(f'Turnover: {len(bt)} bars, avg {bt[\"minutes\"].mean():.1f} min/bar')
"
```

## Key Results

### Return Stability
Turnover-resampled bars produce **more stable return statistics** (lower volatility of rolling skewness, kurtosis, ACF) than fixed-time bars.

### ML Prediction (next-bar direction)
- **RF** (mean AUC 0.524) and **XGBoost** (0.518) outperform neural models
- LSTM and Attention models underperform (~0.50 AUC) on this tabular data
- Fixed-time vs turnover-resampled bars: **no consistent winner** (differences < 0.01 AUC)
- Limited predictive signal for short-term direction (AUC 0.50-0.54 across all models)

## Requirements

```
numpy pandas scikit-learn xgboost tensorflow openpyxl
```

## Data Source

Minute-level data acquired via Wind Financial Terminal (Alice Agent MCP). Not included in this repo due to size — use `fetch_index_minute_wind.py` to fetch your own copy.
