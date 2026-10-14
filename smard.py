"""Download hourly day-ahead electricity prices (Germany/Luxembourg) from the SMARD API."""
import requests
import pandas as pd

BASE_URL = "https://www.smard.de/app/chart_data"
PRICE_FILTER = 4169   # day-ahead wholesale price, EUR/MWh
REGION = "DE-LU"      # German-Luxembourg bidding zone
RESOLUTION = "hour"


def get_week_timestamps():
    """SMARD stores data in weekly files. This index lists the start timestamp of every week."""
    url = f"{BASE_URL}/{PRICE_FILTER}/{REGION}/index_{RESOLUTION}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()["timestamps"]


def get_week(timestamp):
    """One week of hourly [timestamp_ms, price] pairs."""
    url = f"{BASE_URL}/{PRICE_FILTER}/{REGION}/{PRICE_FILTER}_{REGION}_{RESOLUTION}_{timestamp}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()["series"]


def fetch_prices(weeks=52):
    """Download the last `weeks` weeks and return a clean, time-indexed DataFrame."""
    timestamps = get_week_timestamps()[-weeks:]
    rows = []
    for ts in timestamps:
        rows.extend(get_week(ts))

    df = pd.DataFrame(rows, columns=["timestamp_ms", "price_eur_mwh"])
    df["time"] = (pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
                    .dt.tz_convert("Europe/Berlin"))
    df = (df.dropna(subset=["price_eur_mwh"])   # current week has empty future hours
            .drop(columns="timestamp_ms")
            .set_index("time")
            .sort_index())
    return df[~df.index.duplicated()]