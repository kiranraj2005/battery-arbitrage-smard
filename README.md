# Battery Arbitrage on German Power Prices

**Status: complete** — forecasting day-ahead electricity prices and using the forecast
to schedule a battery's charging and discharging.

## Why this project
Battery storage earns money by charging when electricity is cheap and discharging when it is
expensive. How much it earns depends on two things: how well you forecast prices, and how well
you plan the battery around that forecast. This project builds both, step by step, on public data.

## Plan
- [x] Automated download of hourly day-ahead prices (DE-LU) from the SMARD API (Bundesnetzagentur)
- [x] Exploratory analysis: daily and weekly price patterns, negative-price hours
- [x] Baseline forecast: "same hour last week" (seasonal naive), measured with MAE
- [x] Machine-learning forecast (gradient boosting on calendar and lagged-price features), compared to the baseline
- [x] Battery optimization with linear programming (PuLP): 1 MWh battery, charge/discharge limits,
      efficiency losses; compared to a simple rule-based strategy
- [x] Results: forecast error vs. baseline, revenue vs. rule-based strategy

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

## Machine learning forecast

A gradient-boosting model (scikit-learn's `GradientBoostingRegressor`) on calendar features
(hour, day of week, month, weekend flag) plus three price lags brings the MAE down to
**41.19 EUR/MWh — a 23% improvement over the 53.53 EUR/MWh seasonal-naive baseline**, on the
exact same 8-week chronological test split.

**A deliberate feature-engineering decision:** the lags used are 24h, 48h, and 168h (yesterday,
two days ago, and last week, all at the same hour) - NOT a 1-hour-ago lag, even though that would
be a more obviously "predictive" feature. SMARD day-ahead prices are auctioned once per day for
all 24 hours of the next day at once. At the moment a real day-ahead forecast is actually made,
"the price one hour ago" relative to a target hour tomorrow doesn't exist yet as usable
information - tomorrow hasn't happened. Including it would be leakage: training the model on
information it could never actually have at real prediction time. Every lag used here is always
at least a full day old relative to its target hour, so the model only ever sees what a real
forecast could see.

Feature importances confirm the model is learning something real, not just memorizing noise:
`lag_24h` (yesterday, same hour) accounts for 68% of the model's decisions - more than double
`lag_168h` (last week, same hour) at 13%. That's a genuine finding the seasonal-naive baseline
had no way to surface, since it only ever looks a week back: yesterday's price is actually a
stronger predictor of today's than last week's price is. See `03_ml_forecast.ipynb` for the
full feature importance chart and a side-by-side comparison of both forecasts against the same
sample week used in the baseline notebook.

## Battery optimization

A linear program (PuLP, CBC solver) decides each hour's charge/discharge to maximize revenue,
given a 1 MWh battery, a 1 MW charge/discharge rate (a full cycle takes one hour), and a 90%
round-trip efficiency (a typical lithium-ion figure, not fitted to real hardware). Given the
actual, realized prices over the same 8-week test window used for forecasting, this gives a
perfect-foresight revenue ceiling - the best any schedule could possibly do - of
**13,103.51 EUR**, against **9,310.87 EUR** for a naive rule-based strategy that always charges
at 13:00 and discharges at 19:00 (the historically cheapest and most expensive hours, from the
exploratory analysis), with no awareness of what any particular day actually looks like. The LP
captures **40.7% more revenue** than the fixed-schedule rule, using the exact same battery
physics for both, so the comparison is about strategy, not about different assumed hardware.

**A finding worth being honest about, not hiding:** in about 4% of hours, the optimal schedule
charges and discharges in the same hour - which looks like a bug at first (why would charging
and then immediately discharging ever be worth it?). Checked rather than assumed: every single
one of these hours happens during a negative electricity price, where charging earns money and
discharging costs money, so the usual intuition doesn't hold. Some are the battery hitting its
capacity ceiling and venting a little to keep charging at the max rate; others are the LP
deliberately freeing up capacity during a merely-cheap hour to fully capture an even cheaper one
an hour or two later - a genuine cross-hour lookahead trick a simple rule could never do. The
total effect is tiny (0.08% of revenue) but real, not solver noise. See `04_battery_optimization.ipynb`
for the full verification. This also surfaces a real model limitation worth naming: no cost for
battery degradation/cycling wear is included, which a production system would need.

## Results: does a better forecast actually make more money?

Bringing the forecasting and optimization halves of this project together, scored against the
same 8-week holdout of real, realized prices throughout:

| Strategy | Revenue (8 weeks) |
|---|---|
| Rule-based (fixed 13:00 charge / 19:00 discharge, no model) | 9,310.87 EUR |
| **Forecast-driven LP (gradient boosting predictions)** | **10,405.21 EUR** |
| Perfect foresight (theoretical ceiling, cannot be beaten) | 13,103.51 EUR |

The gradient-boosting model cut forecast error by 23% over the seasonal-naive baseline
(53.53 -> 41.19 EUR/MWh MAE). Translated into money, running the battery on that forecast
instead of the naive fixed schedule captures **11.8% more revenue**, and reaches **79.4% of the
theoretical maximum** that a schedule with perfect knowledge of the future could ever achieve.
That's the real answer to "was building a better forecast worth it": yes, by a meaningful,
quantified amount - not just a better error metric on paper, but money a real battery operator
would actually see. See `05_results.ipynb` for the full calculation.

(These figures are from a real run of the pinned `requirements.txt` environment. Re-running the
gradient boosting step can shift the forecast-driven number by a small amount - under 0.2% in
testing - since scikit-learn's tree-building isn't perfectly reproducible across library
versions even with a fixed random seed. The rule-based and perfect-foresight figures, which
don't depend on the ML model, are exactly reproducible.)

## How to run
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # notebooks run in order:
    #   01_explore_prices.ipynb      - download data, exploratory analysis
    #   02_baseline_forecast.ipynb   - seasonal-naive forecast baseline
    #   03_ml_forecast.ipynb         - gradient boosting forecast
    #   04_battery_optimization.ipynb - LP battery scheduling vs. rule-based
    #   05_results.ipynb             - forecast error and revenue, brought together
