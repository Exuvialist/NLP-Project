import numpy as np
import pandas as pd


def build_market_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values("trading_date").reset_index(drop=True).copy()
    rate = df["usd_idr_rate"].astype(float)
    returns = rate.pct_change()

    # Semua fitur hanya memakai kurs <= t (JISDOR t terbit 00:00 WIB hari t).
    df["ret_1d"] = returns
    df["ret_2d"] = rate.pct_change(2)
    df["ret_3d"] = rate.pct_change(3)
    df["ret_5d"] = rate.pct_change(5)

    df["ma_5"] = rate.rolling(5).mean()
    df["ma_20"] = rate.rolling(20).mean()
    df["rate_to_ma5"] = rate / df["ma_5"] - 1.0
    df["rate_to_ma20"] = rate / df["ma_20"] - 1.0

    df["vol_5"] = returns.rolling(5).std()
    df["vol_20"] = returns.rolling(20).std()
    df["log_rate"] = np.log(rate)
    return df
