"""
Module: feature_market.py
Deskripsi: Rekayasa fitur teknikal (market-only) untuk data historis nilai tukar USD/IDR.
Seluruh fitur pada baris t dihitung HANYA menggunakan informasi hingga hari t untuk mencegah data leakage.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np


MARKET_FEATURE_COLS = [
    "return_lag1",
    "return_lag2",
    "return_lag3",
    "return_lag5",
    "sma5_ratio",
    "sma20_ratio",
    "volatility_5d",
    "volatility_20d",
    "momentum_5d",
    "momentum_20d",
    "day_of_week",
]


def extract_market_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Mengekstrak fitur teknikal dan target pergerakan kurs dari data harian.

    Parameters
    ----------
    df_input : pd.DataFrame
        DataFrame harian (misal dari daily_aligned_dataset.csv)
        Wajib memiliki kolom: 'trading_date', 'usd_idr_rate', 'daily_return'.

    Returns
    -------
    pd.DataFrame
        DataFrame yang telah dilengkapi fitur teknikal, target biner 'target_up',
        dan return hari berikutnya 'next_return'.
    """
    df = df_input.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df = df.sort_values("trading_date").reset_index(drop=True)

    close = df["usd_idr_rate"].astype(float)
    ret = df["daily_return"].astype(float)

    # 1. Target Construction (untuk hari t+1)
    # y_t = 1 jika Close_{t+1} > Close_t, 0 sebaliknya
    next_close = close.shift(-1)
    df["target_up"] = (next_close > close).astype(int)
    df["next_return"] = (next_close - close) / close

    # 2. Return Lags (informasi hingga hari t)
    # return_lag1 adalah return hari t vs t-1 (tersedia saat penutupan hari t)
    df["return_lag1"] = ret
    df["return_lag2"] = ret.shift(1)
    df["return_lag3"] = ret.shift(2)
    df["return_lag5"] = ret.shift(4)

    # 3. Simple Moving Average (SMA) Ratios
    sma5 = close.rolling(window=5).mean()
    sma20 = close.rolling(window=20).mean()
    df["sma5_ratio"] = (close / sma5) - 1.0
    df["sma20_ratio"] = (close / sma20) - 1.0

    # 4. Rolling Volatility (standar deviasi return bergulir)
    df["volatility_5d"] = ret.rolling(window=5).std()
    df["volatility_20d"] = ret.rolling(window=20).std()

    # 5. Price Momentum
    df["momentum_5d"] = (close / close.shift(5)) - 1.0
    df["momentum_20d"] = (close / close.shift(20)) - 1.0

    # 6. Fitur Kalender
    df["day_of_week"] = df["trading_date"].dt.dayofweek

    # Hapus warm-up period (20 hari pertama NaN karena rolling 20)
    # dan baris paling akhir (karena target hari t+1 belum diketahui)
    df_clean = df.iloc[20:-1].copy().reset_index(drop=True)

    return df_clean


def get_market_feature_cols() -> List[str]:
    """Mengembalikan daftar nama kolom fitur market."""
    return list(MARKET_FEATURE_COLS)

