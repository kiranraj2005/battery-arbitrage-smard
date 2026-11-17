# Battery Arbitrage on German Power Prices

**Status: in progress** — forecasting day-ahead electricity prices and using the forecast
to schedule a battery's charging and discharging.

## Why this project
Battery storage earns money by charging when electricity is cheap and discharging when it is
expensive. How much it earns depends on two things: how well you forecast prices, and how well
you plan the battery around that forecast. This project builds both, step by step, on public data.

## Plan
- [x] Automated download of hourly day-ahead prices (DE-LU) from the SMARD API (Bundesnetzagentur)
- [x] Exploratory analysis: daily and weekly price patterns, negative-price hours
- [x] Baseline forecast: "same hour last week" (seasonal naive), measured with MAE
- [ ] Machine-learning forecast (gradient boosting on calendar and lagged-price features), compared to the baseline
- [ ] Battery optimization with linear programming (PuLP): 1 MWh battery, charge/discharge limits,
      efficiency losses; compared to a simple rule-based strategy
- [ ] Results: forecast error vs. baseline, revenue vs. rule-based strategy

## Built on Job Market Radar
This project reuses the approach of my earlier project
[Job Market Radar](https://github.com/kiranraj2005/job-market-radar): automated ingestion from a
public data source, cleaning and validation, automated tests and a reproducible setup.
The forecasting and optimization methods are new to me. Learning them is the point of this project.

## Data
Day-ahead wholesale electricity prices, bidding zone Germany/Luxembourg, hourly, in EUR/MWh.
Source: [SMARD.de](https://www.smard.de) (Bundesnetzagentur), filter 4169.

## First findings (52 weeks, Oct 2025 – Oct 2026)
- Prices are lowest around 13:00 (avg. ~49 EUR/MWh), when solar peaks, and highest at 19–20:00
  (avg. ~159 EUR/MWh). This ~110 EUR/MWh daily spread is what a battery earns from.
- A smaller morning peak around 7:00 (~131 EUR/MWh) suggests a possible second cycle per day.
- Weekends are 25–35 EUR/MWh cheaper than weekdays, so the weekday matters for forecasting.
- 471 hours (~5%) had negative prices, down to −499 EUR/MWh: a battery gets paid to charge.

## Baseline forecast

A seasonal-naive baseline - predicting each hour's price as the price at the same hour, exactly
one week earlier - gives a **mean absolute error of 53.53 EUR/MWh** on an 8-week chronological
test split (37.1% of the mean test price). Evaluated on a time-based split, not a random one:
the model can only ever see the past when forecasting the future, so the test set has to come
strictly after the training history, the same way it would in real use.

The baseline tracks the general daily shape well (it "remembers" that evenings are expensive and
midday is cheap), but visibly misses on days where the actual price breaks from the prior week's
pattern - see `02_baseline_forecast.ipynb` for a week-by-week comparison chart. That gap is what
the gradient-boosting model (next step) has to close to be worth using over this much simpler
approach.

## How to run
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # then open 01_explore_prices.ipynb or 02_baseline_forecast.ipynb