# Genetic Alpha Miner (GA Walk-Forward)

Industry-grade genetic programming / genetic algorithm pipeline for mining cross-sectional equity factors under strict walk-forward evaluation.  
Designed for research-to-trade consistency: mine on past data only, trade on *t* using signals computed from *t-1* information.

## What this repo does

This project implements a **walk-forward connected** factor mining workflow:

1. **Candidate terminal bank**: build a library of factor “terminals” (raw signals / transforms).
2. **Pre-screen + LASSO selection**: reduce the terminal set to a small, interpretable subset (e.g., K terminals) for stability and speed.
3. **GA/GP factor composition**: run a genetic search to combine terminals into composite formulas.
4. **Industry-grade inference**: score candidates using **Newey–West HAC t-stat** on daily IC (information coefficient) means.
5. **Strict holdout backtest**: evaluate out-of-sample on a forward holdout window, then roll forward on a fixed schedule.

The workflow repeats across time slices, producing a full research track record (not just a single lucky period).

## Key design choices (why this is “research-to-trade” aligned)

- **No look-ahead**: signals are aligned so that trading at day *t* uses factor values computed from day *t-1* (or earlier).
- **Walk-forward connected**: the training window and OOS holdout advance through time (e.g., 3Y train + 3M holdout).
- **Purged split inside GA**: GA_train / GA_val split is time-ordered with a purge gap ≥ (H-1)+DELAY to avoid leakage.
- **Fitness = statistical evidence**: primary objective is Newey–West t-stat of daily IC mean, optionally penalized by complexity.
- **Barra-style neutralization (optional)**: daily cross-sectional regression vs beta/size/industry to remove known exposures.

## Inputs / expected data

Typical inputs are daily, aligned matrices:

- `features`: dict of factor/terminal matrices (DataFrame: date × symbol)
- `logret`: daily log returns matrix (date × symbol)
- `spy_aggs` / benchmark series (optional): for beta estimation / market aggregation
- universe / sample-space filters (liquidity, listing age, price sanity, etc.)

All matrices must share the same date index and symbol columns (exact match).

## Outputs

Per walk-forward slice, the pipeline stores:

- selected terminals (LASSO result)
- best GA formulas + per-generation logs (no cutting)
- in-sample / validation / out-of-sample IC stats
- backtest performance metrics and equity curve artifacts
- debug-friendly alignment checks (index/columns, NaNs, leakage guards)

## Typical workflow

1. Prepare aligned `features` + `logret`.
2. Define the walk-forward schedule (train lookback, rebalance frequency, holdout length).
3. Run terminal screening + LASSO.
4. Run GA mining per slice.
5. Build a signal bank and run strict OOS backtest.

> The goal is not one “global factor” that magically works forever;  
> it’s a **repeatable process** that remains statistically defensible across regimes.

## Notes on practical usage

- This repo is built for **research correctness first**, then speed (NumPy/Numba acceleration where appropriate).
- For production trading, integrate with your execution layer (broker API) and add:
  - portfolio constraints
  - slippage/impact model calibration
  - monitoring + kill-switch logic
  - data integrity checks (corporate actions, survivorship bias)

## Disclaimer

This repository is for research and educational purposes only.  
Nothing here is financial advice, and performance in backtests does not guarantee future results.



UNAUTHORIZED USAGE OF THE CONTENTS IS PROHIBITED AND MAY RESULT IN LEGAL ACTIONS
