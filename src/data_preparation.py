from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd

from src import feature_market, feature_sentiment
from src.config import (
    ALIGNED_DIR,
    AUDIT_JSON,
    DAILY_ALIGNED_CSV,
    FORBIDDEN_COLUMNS,
    MARKET_FEATURES,
    NEWS_ALIGNMENT_CSV,
    NEWS_CLEANED_CSV,
    SENTIMENT_FEATURES,
    TARGET_COLUMN,
    TEST_CSV,
    TRAIN_CSV,
    TRAIN_END,
    VALIDATION_CSV,
    VALIDATION_END,
    ensure_output_dirs,
)


def count_csv_records(path) -> int:
    # csv.reader menghitung record logis, bukan baris fisik: isi artikel
    # mengandung newline di dalam field yang dikutip.
    with open(path, encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def audit_dataset() -> dict:
    daily = pd.read_csv(DAILY_ALIGNED_CSV, parse_dates=["trading_date"])
    alignment = pd.read_csv(NEWS_ALIGNMENT_CSV, parse_dates=["publication_date", "effective_trading_date"])
    lag_days = (alignment["effective_trading_date"] - alignment["publication_date"]).dt.days

    audit = {
        "exchange_rate": {
            "rows": int(len(daily)),
            "date_start": str(daily["trading_date"].min().date()),
            "date_end": str(daily["trading_date"].max().date()),
            "columns": list(daily.columns),
            "missing_values": {c: int(n) for c, n in daily.isna().sum().items() if n > 0},
            "duplicate_dates": int(daily["trading_date"].duplicated().sum()),
            "zero_change_days": int((daily["usd_idr_rate"].diff() == 0).sum()),
            "max_gap_days": int(daily["trading_date"].diff().dt.days.max()),
        },
        "news": {
            "cleaned_articles": count_csv_records(NEWS_CLEANED_CSV),
            "cleaned_columns": list(pd.read_csv(NEWS_CLEANED_CSV, nrows=0).columns),
            "published_wib_start": str(alignment["publication_date"].min().date()),
            "published_wib_end": str(alignment["publication_date"].max().date()),
            "articles_without_next_trading_day": int(lag_days.isna().sum()),
        },
        "alignment": {
            "rows": int(len(alignment)),
            "rule": "next_trading_day (WIB)",
            "all_effective_after_publication": bool((alignment["effective_trading_date"] > alignment["publication_date"]).all()),
            "lag_days_distribution": {int(k): int(v) for k, v in lag_days.value_counts().sort_index().items()},
            "articles_per_trading_day": {
                "days_with_news": int((daily["geopolitical_news_count"] > 0).sum()),
                "days_without_news": int((daily["geopolitical_news_count"] == 0).sum()),
                "mean": round(float(daily["geopolitical_news_count"].mean()), 2),
                "median": float(daily["geopolitical_news_count"].median()),
                "max": int(daily["geopolitical_news_count"].max()),
            },
        },
        "conventions": {
            "market_timezone": "Asia/Jakarta (JISDOR terbit 00:00 WIB untuk hari tersebut)",
            "news_timezone": "Asia/Jakarta (dikonversi dari published_at_utc)",
            "information_cutoff": "fitur hari t hanya memakai kurs <= t dan berita terbit sebelum hari t",
            "target": f"{TARGET_COLUMN} = 1 jika kurs hari perdagangan berikutnya > kurs t; flat = 0",
        },
    }
    return audit


def build_target(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values("trading_date").reset_index(drop=True).copy()
    next_rate = df["usd_idr_rate"].shift(-1)
    df["usd_idr_rate_next"] = next_rate
    # Target mengikuti AGENTS.md: arah kurs hari berikutnya; hari flat dihitung 0 (non-kenaikan).
    # Kolom ini hanya dipakai untuk membentuk label, tidak pernah menjadi fitur.
    target = (next_rate > df["usd_idr_rate"]).astype(float)
    target[next_rate.isna()] = np.nan
    df[TARGET_COLUMN] = target
    return df


def build_modeling_frame(force_sentiment: bool = False) -> pd.DataFrame:
    daily = pd.read_csv(DAILY_ALIGNED_CSV, parse_dates=["trading_date"])
    frame = build_target(daily)
    frame = feature_market.build_market_features(frame)

    sentiment = feature_sentiment.build_daily_sentiment(force=force_sentiment)
    frame = frame.merge(sentiment, on="trading_date", how="left")
    frame[SENTIMENT_FEATURES] = frame[SENTIMENT_FEATURES].fillna(0.0)

    before = len(frame)
    frame = frame.dropna(subset=MARKET_FEATURES + [TARGET_COLUMN]).reset_index(drop=True)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    print(f"baris terbuang (histori < 20 hari atau label hari berikutnya tidak ada): {before - len(frame)}")
    return frame


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame["trading_date"] <= TRAIN_END].reset_index(drop=True)
    validation = frame[(frame["trading_date"] > TRAIN_END) & (frame["trading_date"] <= VALIDATION_END)].reset_index(drop=True)
    test = frame[frame["trading_date"] > VALIDATION_END].reset_index(drop=True)
    return train, validation, test


def verify_frame(frame: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    features = MARKET_FEATURES + SENTIMENT_FEATURES
    leaked = FORBIDDEN_COLUMNS.intersection(features)
    assert not leaked, f"kolom terlarang dipakai sebagai fitur: {leaked}"
    assert frame[features].notna().all().all(), "fitur mengandung NaN"
    assert train["trading_date"].max() < validation["trading_date"].min(), "split train/validation tidak kronologis"
    assert validation["trading_date"].max() < test["trading_date"].min(), "split validation/test tidak kronologis"
    recomputed = (frame["usd_idr_rate_next"] > frame["usd_idr_rate"]).astype(int)
    assert (recomputed == frame[TARGET_COLUMN]).all(), "target tidak konsisten dengan kurs hari berikutnya"

    alignment = pd.read_csv(NEWS_ALIGNMENT_CSV, parse_dates=["publication_date", "effective_trading_date"])
    assert (alignment["publication_date"] < alignment["effective_trading_date"]).all(), "ada berita yang dipakai sebelum terbit"
    print("verifikasi leakage: lolos (fitur aman, split kronologis, cutoff berita benar)")


def summarize_split(name: str, split: pd.DataFrame) -> dict:
    return {
        "split": name,
        "rows": int(len(split)),
        "date_start": str(split["trading_date"].min().date()),
        "date_end": str(split["trading_date"].max().date()),
        "up_ratio": round(float(split[TARGET_COLUMN].mean()), 3),
    }


def main() -> pd.DataFrame:
    ensure_output_dirs()
    audit = audit_dataset()
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"audit dataset -> {AUDIT_JSON.name}")

    frame = build_modeling_frame()
    train, validation, test = split_frame(frame)
    verify_frame(frame, train, validation, test)

    save_columns = ["trading_date", "usd_idr_rate"] + MARKET_FEATURES + SENTIMENT_FEATURES + [TARGET_COLUMN]
    train[save_columns].to_csv(TRAIN_CSV, index=False)
    validation[save_columns].to_csv(VALIDATION_CSV, index=False)
    test[save_columns].to_csv(TEST_CSV, index=False)

    summary = pd.DataFrame([summarize_split("train", train), summarize_split("validation", validation), summarize_split("test", test)])
    print(summary.to_string(index=False))
    print(f"dataset tersimpan: {TRAIN_CSV.name}, {VALIDATION_CSV.name}, {TEST_CSV.name}")
    return frame


if __name__ == "__main__":
    main()
