"""
Module: data_preparation.py
Deskripsi: Memuat data harian kurs, mengekstrak fitur market, dan membagi dataset
secara kronologis (Train / Validation / Test) untuk mencegah data leakage pada time-series.
"""

import os
from pathlib import Path
from typing import Tuple
import pandas as pd

from src.feature_market import extract_market_features


# Definisi batas waktu pembagian temporal (Chronological Split)
TRAIN_END_DATE = "2024-12-31"
VAL_START_DATE = "2025-01-01"
VAL_END_DATE = "2025-12-31"
TEST_START_DATE = "2026-01-01"


def split_chronological(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Membagi DataFrame berdasarkan tanggal secara kronologis:
    - Train: s/d 2024-12-31
    - Validation: 2025-01-01 s/d 2025-12-31
    - Test: 2026-01-01 ke atas

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang sudah memiliki kolom 'trading_date'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    df = df.copy()
    dates = pd.to_datetime(df["trading_date"])

    train_df = df[dates <= TRAIN_END_DATE].copy().reset_index(drop=True)
    val_df = df[(dates >= VAL_START_DATE) & (dates <= VAL_END_DATE)].copy().reset_index(drop=True)
    test_df = df[dates >= TEST_START_DATE].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def prepare_market_dataset(
    input_path: str = "data/aligned/daily_aligned_dataset.csv",
    output_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pipeline lengkap ekstraksi fitur market dan penyimpanan split dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/3] Membaca dataset dari: {input_path}")
    raw_df = pd.read_csv(input_path)

    print("[2/3] Mengekstrak fitur market & target...")
    featured_df = extract_market_features(raw_df)

    print("[3/3] Membagi dataset secara kronologis...")
    train_df, val_df, test_df = split_chronological(featured_df)

    train_path = Path(output_dir) / "train_market.csv"
    val_path = Path(output_dir) / "val_market.csv"
    test_path = Path(output_dir) / "test_market.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n=== RINGKASAN DATASET MARKET-ONLY ===")
    for name, split in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        n_total = len(split)
        n_up = int(split["target_up"].sum())
        n_down = n_total - n_up
        d_min = split["trading_date"].min()
        d_max = split["trading_date"].max()
        print(
            f"{name:10}: {n_total:4d} baris | {d_min.strftime('%Y-%m-%d')} s/d {d_max.strftime('%Y-%m-%d')} | "
            f"UP={n_up:3d} ({n_up/n_total*100:.1f}%) | DOWN={n_down:3d} ({n_down/n_total*100:.1f}%)"
        )

    print(f"\nFile tersimpan di folder: {output_dir}/")
    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_market_dataset()

