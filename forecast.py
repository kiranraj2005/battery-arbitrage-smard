"""Baseline (and later, ML) forecasting for day-ahead electricity prices."""
import pandas as pd

LAG_HOURS = 24 * 7   # one week, in hours - the basis of the "same hour last week" baseline


def load_prices(path="data/prices_hourly.csv"):
    """Load the saved price history with a correctly timezone-aware hourly index.

    Reading the CSV with pandas' own `parse_dates` leaves the index as plain Python
    objects, not a real datetime64 index - because the timestamps carry explicit
    UTC offsets that change across the year (+02:00 in summer/CEST, +01:00 in
    winter/CET, from Europe's daylight-saving switch). pandas can't build one
    uniform tz-aware column out of two different offsets by itself, so it silently
    falls back to object dtype. The fix: parse as UTC first (one single, unambiguous
    offset for every row), then convert the *display* timezone to Europe/Berlin.
    """
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Berlin")
    return df


def train_test_split_time(df, test_weeks=8):
    """Split chronologically: the last `test_weeks` weeks are the test set, everything
    before that is "training" history.

    This is NOT the usual random train/test split. A random split would let the model
    (or, here, the baseline) see prices from *after* the test period while being
    evaluated on prices from *before* it - which can never happen in real use, since
    you only ever forecast forward in time from what's already happened. A chronological
    split is what keeps the evaluation honest, and it's the same split the gradient
    boosting model in the next step will use, so the baseline and the model are compared
    on exactly the same test window.
    """
    cutoff = df.index.max() - pd.Timedelta(weeks=test_weeks)
    train = df[df.index <= cutoff]
    test = df[df.index > cutoff]
    return train, test


def seasonal_naive_forecast(df, column="price_eur_mwh"):
    """Predict each hour's price as the price at the same hour, exactly one week earlier.

    Returns a Series aligned to the FULL input `df`, not just a test slice. This matters:
    shifting has to happen on the whole, continuous price history before you cut out a
    test window, not after. If you sliced to the test period first and then shifted,
    the first week of the test set would have no prior week to look back on *within the
    slice* and would come out as NaN - even though the real prior week exists just before
    the test window, in the training data. Shifting first keeps that history available.
    """
    return df[column].shift(LAG_HOURS)


def mae(actual, predicted):
    """Mean absolute error, in the same units as price (EUR/MWh).

    MAE over RMSE on purpose: this dataset has real negative-price hours down to
    -499 EUR/MWh (see the exploratory analysis). Squared-error metrics like RMSE
    blow up disproportionately on those extreme hours, making a handful of unusual
    hours dominate the whole score. MAE just says "on average, we're off by X EUR/MWh" -
    a number a non-technical reader (or an interviewer) can sanity-check immediately.
    """
    aligned = pd.concat([actual, predicted], axis=1, keys=["actual", "predicted"]).dropna()
    return (aligned["actual"] - aligned["predicted"]).abs().mean()
