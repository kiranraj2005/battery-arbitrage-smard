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


def build_features(df, column="price_eur_mwh"):
    """Calendar + lag features for the gradient-boosting forecast.

    All lags here are multiples of 24 hours (yesterday, two days ago, last week) -
    deliberately NOT a 1-hour or few-hours-back lag. SMARD day-ahead prices are
    auctioned once per day for all 24 hours of the next day at once, around midday.
    At the moment that real forecast is actually made, "the price one hour ago"
    relative to a given target hour tomorrow doesn't exist yet as usable information -
    tomorrow hasn't happened. A short lag like that would let the model train on
    information it could never actually have at real prediction time: leakage,
    just a subtler form of it than usual. Lags of a full day or more sidestep this,
    since they're always already-settled history by the time a real day-ahead
    forecast would be made.

    Calendar fields (hour, day of week, month, weekend flag) are safe regardless -
    they're known in advance with certainty, no auction timing to worry about.

    Returned aligned to the FULL input index, same reasoning as seasonal_naive_forecast:
    shift on the whole continuous series first, slice into train/test afterward.
    """
    features = pd.DataFrame(index=df.index)
    features["hour"] = df.index.hour
    features["dayofweek"] = df.index.dayofweek
    features["month"] = df.index.month
    features["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    features["lag_24h"] = df[column].shift(24)
    features["lag_48h"] = df[column].shift(48)
    features["lag_168h"] = df[column].shift(LAG_HOURS)
    features[column] = df[column]
    return features


FEATURE_COLUMNS = ["hour", "dayofweek", "month", "is_weekend", "lag_24h", "lag_48h", "lag_168h"]


def gradient_boosting_forecast(df, test_weeks=8, column="price_eur_mwh"):
    """Fit the gradient boosting model on the chronological training split and return its
    predictions for the test window, plus the fitted model itself.

    This reuses the exact feature set and split validated in 03_ml_forecast.ipynb, so any other
    notebook that needs the model's actual PREDICTIONS (not just its error metric) - like scoring
    a battery schedule built from the forecast instead of from perfect-foresight prices - can
    call this once instead of re-deriving the pipeline inline.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    features = build_features(df, column=column)
    train_f, test_f = train_test_split_time(features, test_weeks=test_weeks)
    train_f = train_f.dropna(subset=FEATURE_COLUMNS + [column])

    model = GradientBoostingRegressor(random_state=42)
    model.fit(train_f[FEATURE_COLUMNS], train_f[column])
    predictions = pd.Series(model.predict(test_f[FEATURE_COLUMNS]), index=test_f.index)
    return predictions, model
