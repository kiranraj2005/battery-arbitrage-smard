"""Battery charge/discharge scheduling: the profit-maximizing LP, and a naive rule-based baseline."""
import pandas as pd
import pulp

CAPACITY_MWH = 1.0           # a 1 MWh battery, as scoped in the project plan
MAX_RATE_MW = 1.0            # can fully charge or discharge in one hour (a "1C" battery)
ROUND_TRIP_EFFICIENCY = 0.90  # typical for lithium-ion storage; an assumption, not fitted to real hardware


def optimize_schedule(prices, capacity_mwh=CAPACITY_MWH, max_rate_mw=MAX_RATE_MW,
                       efficiency=ROUND_TRIP_EFFICIENCY, initial_soc=0.0):
    """Solve for the profit-maximizing charge/discharge schedule against a KNOWN price series.

    Deliberately named around "known" prices: calling this with actual, realized prices gives
    the perfect-foresight revenue ceiling - the best any schedule could possibly do, which no
    real strategy can beat, only approach. The same function can later be called with a
    forecast standing in for `prices`, and the resulting schedule scored against real prices
    (see `schedule_revenue`) to measure how much of that ceiling a realistic, forecast-driven
    strategy actually captures.

    No explicit "don't charge and discharge in the same hour" constraint is added, and this
    needs a more careful claim than "it's never profitable" - that turns out to be false in a
    small but real set of cases. During NEGATIVE price hours (around 5% of hours in this
    dataset), charging earns money (you're paid to take delivery) while discharging costs money
    (you're paying to push power out) - so simultaneous charge+discharge during a negative-price
    hour isn't automatically wasteful the way it is at a positive price. Verified against the
    actual solved schedule: roughly 4% of hours do both at once, every single one during a
    negative price, and every single one adds strictly positive value versus not discharging -
    some because the capacity ceiling genuinely forces a bit of discharge to keep charging at
    the maximum rate, others because venting a little now (at a merely-cheap negative price)
    keeps capacity free to fully capture an even-more-negative price one or two hours later, a
    cross-hour lookahead trick that a hour-by-hour rule could never find. Total economic size is
    tiny (well under 1% of total revenue) but genuinely positive, not solver noise - confirmed
    by checking, not assumed. A binary "no simultaneous activity" constraint would eliminate
    this and turn the problem into a slower mixed-integer program for essentially zero revenue
    benefit, so it's left as-is and documented here instead.

    The final state of charge is pinned back to `initial_soc`. Without that constraint, the
    optimizer could end the window with a full battery and effectively "bank" stored value for
    free, since nothing after the evaluation window is ever scored - inflating the result.
    """
    hours = list(prices.index)
    n = len(hours)
    price_values = prices.values

    problem = pulp.LpProblem("battery_arbitrage", pulp.LpMaximize)
    charge = pulp.LpVariable.dicts("charge", range(n), lowBound=0, upBound=max_rate_mw)
    discharge = pulp.LpVariable.dicts("discharge", range(n), lowBound=0, upBound=max_rate_mw)
    soc = pulp.LpVariable.dicts("soc", range(n), lowBound=0, upBound=capacity_mwh)

    problem += pulp.lpSum(price_values[t] * (discharge[t] - charge[t]) for t in range(n))

    for t in range(n):
        previous_soc = initial_soc if t == 0 else soc[t - 1]
        problem += soc[t] == previous_soc + efficiency * charge[t] - discharge[t]

    problem += soc[n - 1] == initial_soc

    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    return pd.DataFrame({
        "charge_mwh": [charge[t].value() for t in range(n)],
        "discharge_mwh": [discharge[t].value() for t in range(n)],
        "soc_mwh": [soc[t].value() for t in range(n)],
    }, index=prices.index)


def rule_based_schedule(prices, capacity_mwh=CAPACITY_MWH, max_rate_mw=MAX_RATE_MW,
                         efficiency=ROUND_TRIP_EFFICIENCY, charge_hour=13, discharge_hour=19,
                         initial_soc=0.0):
    """A naive, non-optimized strategy: charge at a fixed hour every day (13:00, the
    historically cheapest hour on average - see the exploratory analysis), discharge at a
    fixed hour every day (19:00, the historically most expensive). No forecasting, no
    optimization, no adapting to what a given day actually looks like - this is the benchmark
    a real forecast-driven strategy has to beat to be worth building at all.

    Uses the exact same capacity/rate/efficiency assumptions as `optimize_schedule` - if the
    two strategies used different battery physics, a revenue difference wouldn't mean anything;
    it would be comparing different batteries, not different control strategies.
    """
    rows = []
    soc = initial_soc
    for t in prices.index:
        charge_mwh = 0.0
        discharge_mwh = 0.0
        if t.hour == charge_hour:
            room_for_more_charge = (capacity_mwh - soc) / efficiency
            charge_mwh = min(max_rate_mw, room_for_more_charge)
        elif t.hour == discharge_hour:
            discharge_mwh = min(max_rate_mw, soc)
        soc = soc + efficiency * charge_mwh - discharge_mwh
        rows.append((t, charge_mwh, discharge_mwh, soc))

    return pd.DataFrame(rows, columns=["time", "charge_mwh", "discharge_mwh", "soc_mwh"]).set_index("time")


def schedule_revenue(schedule, actual_prices):
    """Revenue a given charge/discharge schedule actually earns against REAL prices.

    Kept separate from `optimize_schedule` so a schedule built from a forecast (or any
    heuristic) can be scored against what really happened - distinct from whatever prices
    were used to build the schedule in the first place.
    """
    aligned_prices = actual_prices.loc[schedule.index]
    return (aligned_prices * (schedule["discharge_mwh"] - schedule["charge_mwh"])).sum()
